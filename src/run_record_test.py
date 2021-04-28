# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
import sys
import numpy as np
from tqdm import tqdm, trange
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_gather
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


from model.model_utils import configure_tokenizer_model, load_model_from_checkpoint
from kgs_retrieve.kg_utils import initialize_kg_retriever
from utils.args import ArgumentGroup
from text_processor.record import RecordResult, RecordProcessor, create_input, DefinitionInfo
from dgl import save_graphs, load_graphs
import pickle
from nltk.corpus import wordnet as wn
from kgs_retrieve.baseretriever import KGRetriever, read_concept_embedding, run_strip_accents
from tqdm import tqdm

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

sys.path.append('..')
sys.path.append('.')
# os.environ['CUDA_VISIBLE_DEVICES']='5, 6'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


logger.info("running on GPU: {}".format(torch.cuda.current_device()))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, model, processor, tokenizer, global_step, input_dir, prefix=""):
    retrievers = dict()
    for kg in args.use_kgs:
        logger.info("Initialize kg:{}".format(kg))
        kg_path = os.path.join(input_dir, args.kg_paths[kg])
        data_path = os.path.join(args.data_dir, args.kg_paths[kg])

        if not os.path.exists(kg_path):
            logger.warning("need prepare training dataset firstly, program exit")
            exit()

        retrievers[kg] = initialize_kg_retriever(kg, kg_path, data_path, args.cache_file_suffix)

    dataset, examples_tokenized, features, wn_synset_graphs, wn_synset_graphs_label_dict = \
        load_and_cache_examples(args,
                                processor,
                                retrievers,
                                relation_list=args.relation_list,
                                input_dir = input_dir,
                                evaluate=True,
                                output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.mkdir(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # synset_graphs_batch = []
    # for batch_index in eval_dataloader.batch_sampler:
    #     synset_graphs_batch.append([i for i in batch_index])

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1 and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
                )


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Dataset size = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    if args.local_rank == -1:
        all_results = []
        start_time = timeit.default_timer()
        epoch_iterator = tqdm(eval_dataloader, desc="Evaluating Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            batch_synset_graphs = batch[3]
            # batch_synset_graphs = torch.tensor(synset_graphs_batch[step], device=args.device)

            with torch.no_grad():
                inputs = create_input(args, batch, global_step, batch_synset_graphs=batch_synset_graphs, wn_synset_graphs=wn_synset_graphs, evaluate=True)
                feature_indices = batch[3]

                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = RecordResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))


        predictions = RecordProcessor.compute_predictions_logits(
            examples_tokenized,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            args.verbose_logging,
            os.path.join(args.data_dir, args.predict_file),
            tokenizer,
            is_testing=args.test
        )

        # Compute the F1 and exact scores.
        results = RecordProcessor.record_evaluate(examples_tokenized, predictions, args.data_dir)
        return results
    else:
        # all_results = []
        all_start_logits = torch.tensor([], dtype=torch.float32, device=args.device)
        all_end_logits = torch.tensor([], dtype=torch.float32, device=args.device)
        all_unique_ids = []
        # start_time = timeit.default_timer()
        epoch_iterator = tqdm(eval_dataloader, desc="Evaluating Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            batch_synset_graphs = batch[3]
            # batch_synset_graphs = torch.tensor(synset_graphs_batch[step], device=args.device)
            with torch.no_grad():
                inputs = create_input(args, batch, global_step, batch_synset_graphs=batch_synset_graphs, wn_synset_graphs=wn_synset_graphs, evaluate=True)
                feature_indices = batch[3]

                outputs = model(**inputs)

            # if step == 0:
            #     all_logits = torch.Tensor([o for o in outputs], device=args.device)
            # else:
            all_start_logits = torch.cat((all_start_logits, outputs[0]), dim=0)
            all_end_logits = torch.cat((all_end_logits, outputs[1]), dim=0)
            # all_logits = torch.cat((all_logits, torch.cat((outputs[0].unsqueeze(0), outputs[1].unsqueeze(0)), dim=0).unsqueeze(0)), dim=0)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_unique_ids.append(unique_id)
                #
                # output = [to_list(output[i]) for output in outputs]
                # start_logits, end_logits = output
                # result = RecordResult(unique_id, start_logits, end_logits)
                #
                # all_results.append(result)
        all_unique_ids = torch.tensor(all_unique_ids, dtype=torch.long, device=args.device)

        start_time = timeit.default_timer()

        all_start_logits_list = [torch.zeros_like(all_start_logits, device=args.device) for _ in range(torch.distributed.get_world_size())]
        all_end_logits_list = [torch.zeros_like(all_end_logits, device=args.device) for _ in range(torch.distributed.get_world_size())]
        all_unique_ids_list = [torch.zeros_like(all_unique_ids, device=args.device) for _ in range(torch.distributed.get_world_size())]

        all_gather(all_start_logits_list, all_start_logits)
        all_gather(all_end_logits_list, all_end_logits)
        all_gather(all_unique_ids_list, all_unique_ids)

        logger.info("time for gather communication:{} in rank {}".format(timeit.default_timer() - start_time, args.local_rank))

        if args.local_rank == 0:
            start_time = timeit.default_timer()
            all_results = []
            all_unique_ids_list = all_unique_ids_list
            all_start_logits_list = all_start_logits_list
            all_end_logits_list = all_end_logits_list

            for batch_idx, batch_unique_ids in enumerate(all_unique_ids_list):
                batch_start_logits = all_start_logits_list[batch_idx]
                batch_end_logits = all_end_logits_list[batch_idx]
                for i, unique_id in enumerate(batch_unique_ids):
                    # output = [to_list(output[i]) for output in batch_logits]
                    start_logits, end_logits = to_list(batch_start_logits[i]), to_list(batch_end_logits[i])
                    result = RecordResult(int(unique_id.cpu().numpy()), start_logits, end_logits)

                    all_results.append(result)

            evalTime = timeit.default_timer() - start_time
            logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

            # Compute predictions
            output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
            output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))


            predictions = RecordProcessor.compute_predictions_logits(
                examples_tokenized,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                args.verbose_logging,
                os.path.join(args.data_dir, args.predict_file),
                tokenizer,
                is_testing=args.test,
            )

            # Compute the F1 and exact scores.
            if not args.test:
                results = RecordProcessor.record_evaluate(examples_tokenized, predictions, args.data_dir)
                return results
        else:
            return None

def load_and_cache_examples(args, processor, retrievers, relation_list, input_dir, evaluate=False, output_examples=False):
    """
    :param args: arguments. Here use "local_rank", "cache_dir", "model_type", "max_seq_length", "data_dir",
    "train_file", "tokenization_train_filepath", "predict_file", "tokenization_dev_filepath", "retrieved_nell_concept_filepath",
    :param tokenizer: the predefined tokenzier, correpsonding to the type of model. Each model has its own tokenizer.
    :param evaluate: bool. An indicator for loading train file or dev file.
    :param output_examples: bool. To decide whether to output  examples.
    :return:
    """

    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "test",
            args.model_type,
            args.speed_up_version,
            str(args.cache_file_suffix),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples_tokenized = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
        if args.model_type in ["bert", "roberta", "roberta_base"]:
            logger.info("not have to load graph data")
            all_kgs_graphs, all_kgs_graphs_label_dict = None, {}
        else:
            all_kgs_graphs, all_kgs_graphs_label_dict = load_graphs(cached_features_file + "_all_kgs_graphs.bin")
            # defid2defembed = torch.load(os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_embedding")["defid2defembed"]
    else:
        logger.error("dataset not exist and program exits")
        exit()
    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    logger.info("{} load data is done".format(args.local_rank))

    if output_examples:
        return dataset, examples_tokenized, features, all_kgs_graphs, all_kgs_graphs_label_dict

    # exit()
    return dataset, all_kgs_graphs, all_kgs_graphs_label_dict

def create_dataset(args, processor, retrievers, relation_list, evaluate, input_dir):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    definition_info = DefinitionInfo()
    tokenizer, _ = configure_tokenizer_model(args, logger, retrievers, is_preprocess=True)

    logger.info("tokenizer: {}".format(tokenizer))

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "test",
            args.model_type,
            args.speed_up_version,
            str(args.cache_file_suffix),
        ),
    )

    if os.path.exists(cached_features_file):
        logger.warning("cache file exist and exit program")
        exit()

    logger.info("Creating features from dataset file at %s", input_dir)

    if not os.path.exists(cached_features_file + "_example"):
        if args.test:
            examples = processor.get_test_examples(args.data_dir, filename=args.predict_file)
        else:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        torch.save(examples, cached_features_file + "_example")
    else:
        logger.info("Loading examples from cached files.")
        examples = torch.load(cached_features_file + "_example")

    if not os.path.exists(cached_features_file + "_example_tokenized"):
        examples_tokenized = processor.tokenization_on_examples(examples, tokenizer, is_testing=args.test)

    if not os.path.exists(cached_features_file + '_features_no_pad'):
        features = processor.convert_examples_to_features(args, examples_tokenized, tokenizer, retrievers, not evaluate, debug=args.debug)

    features, dataset, all_kgs_graphs = processor.pad_and_index_features_all(
        features, retrievers, args, tokenizer, relation_list, encoder=None, definition_info=definition_info, is_training=not evaluate, debug=args.debug)

    if args.local_rank in [-1, 0]:
        # wn_synset_graphs = all_kgs_synset_graphs["wordnet"]
        if args.model_type in ["bert", "roberta", "roberta_base"]:
            all_kgs_graphs_label_dict = None
        else:
            all_kgs_graphs_label_dict = {"glabel": torch.tensor([i for i in range(len(all_kgs_graphs))])}
            save_graphs(cached_features_file+"_all_kgs_graphs.bin", all_kgs_graphs, all_kgs_graphs_label_dict)
            logger.info("complete data preprocessing")

        logger.info("Saving features into cached file %s", cached_features_file)

        for f in features:
            del f.kgs_conceptids2synset
        torch.save({"features": features, "dataset": dataset, "examples": examples_tokenized}, cached_features_file)

        logger.info("Saving knowledge graph retrievers")
        for kg, retriever in retrievers.items():
            if not os.path.exists(os.path.join(input_dir, args.kg_paths[kg])):
                os.mkdir(os.path.join(input_dir, args.kg_paths[kg]))
            torch.save(retriever, os.path.join(input_dir, args.kg_paths[kg], kg + args.cache_file_suffix))

        logger.info("data create is done")

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

def read_wn18_embedding(filepath):
    concept_embedding_path = os.path.join(filepath, "wn_concept2vec.txt")

    id2concept, concept2id, concept_embedding_mat = read_concept_embedding(
        concept_embedding_path)


    offset_to_wn18name_dict = {}
    fin = open(os.path.join(filepath, 'wordnet-mlj12-definitions.txt'))
    for line in fin:
        info = line.strip().split('\t')
        offset_str, synset_name = info[0], info[1]
        offset_to_wn18name_dict[offset_str] = synset_name

    return id2concept, concept2id, concept_embedding_mat, offset_to_wn18name_dict


def create_definition_table(args, filepath):
    tokenizer, encoder = configure_tokenizer_model(args, logger, None, is_preprocess=True)
    logger.info("tokenizer: {}".format(tokenizer))
    logger.info("encoder: {}".format(encoder))
    for param in encoder.parameters():
        param.requires_grad = False

    id2concept, concept2id, concept_embedding_mat, offset_to_wn18name_dict = read_wn18_embedding(filepath)
    repeated_id_path = os.path.join(filepath, "repeated_id.npy")
    repeated_id = np.load(repeated_id_path, allow_pickle='TRUE').item()

    wn18_dir = os.path.join(filepath, "wn18/text")
    synset_name_set_path = os.path.join(wn18_dir, "synset_name.txt")
    with open(synset_name_set_path, "rb") as fp:
        synset_name_set = set(pickle.load(fp))

    definition_embedding_mat = torch.zeros(concept_embedding_mat.shape[0], 1024)

    for item in tqdm(synset_name_set):
        try:
            synset = wn.synset(item)
        except:
            logger.warning("{} can't find synset".format(item))
        check_synset(item, synset)
        offset_str = str(synset.offset()).zfill(8)

        if offset_str in offset_to_wn18name_dict:
            synset_nnnnname = offset_to_wn18name_dict[offset_str]
            id = concept2id[synset_nnnnname]
            try:
                definition_sentence = synset.definition()
            except:
                logger.warning("{} can't find definition".format(synset))
                continue

            if id in repeated_id:
                continue

            definition_embedding_mat[id, :] = encoder(**tokenizer(definition_sentence, return_tensors="pt", padding=True))[1]

    return definition_embedding_mat

def check_synset(synset, synset_name):
    if len(str(synset_name).split("'")) == 3:
        tmp = str(synset_name).split("'")[1]
    else:
        tmp = str(synset_name).replace("')", "('").split("('")[1]

    assert tmp == synset

def main():
    parser = argparse.ArgumentParser()

    model_g = ArgumentGroup(parser, "model", "model configuration and pafths.")

    model_g.add_arg("test", bool, False, "test")
    model_g.add_arg("use_wn", bool, True, "wn")
    model_g.add_arg("use_nell", bool, True, "nell")
    model_g.add_arg("use_context_graph", bool, True, "use_context_graph")


    model_g.add_arg("schedule_strategy", str, "linear", "schedule_strategy")
    model_g.add_arg("tokenizer_path", str, "", "tokenizer_path")
    model_g.add_arg("save_model", bool, True, "whether save model")
    model_g.add_arg("speed_up_version", str, "v2", "choose speed up strategy")
    model_g.add_arg("data_preprocess", bool, False, "data process")
    model_g.add_arg("data_preprocess_evaluate", bool, False, "data_preprocess_evaluate")

    # multi-relational part
    model_g.add_arg("relation_agg", str, "sum", "the method to aggeregate multi-relational neoghbor")

    model_g.add_arg("is_lemma", bool, False, "whether trigger lemma")
    model_g.add_arg("is_filter", bool, True, "weather filter node not in wn18")
    model_g.add_arg("is_clean", bool, True, "weather filter node not in repeated_id")
    model_g.add_arg("is_morphy", bool, False, "weather morphy")
    model_g.add_arg("fewer_label", bool, False, "weather fewer_label")
    model_g.add_arg("label_rate", float, 0.1, "label rate")

    model_g.add_arg("relation_list", list, ["_hyponym", "_hypernym", "_derivationally_related_form", "_member_meronym", "_member_holonym",
                     "_part_of", "_has_part", "_member_of_domain_topic", "_synset_domain_topic_of", "_instance_hyponym",
                     "_instance_hypernym", "_also_see", "_verb_group", "_member_of_domain_region",
                     "_synset_domain_region_of", "_member_of_domain_usage", "_synset_domain_usage_of", "_similar_to"], "The used relation.")
    model_g.add_arg("is_all_relation", bool, True, "use all relations")
    model_g.add_arg("selected_relation", str, "_hyponym,_hypernym,_derivationally_related_form", "relations")
    model_g.add_arg("is_parallel_for_dgl_preprocess", bool, False, "is_parallel_for_dgl_preprocess")
    model_g.add_arg("chunksize_for_graph", int, 40, "The chunksize for multiprocessing to convert examples to features.")
    model_g.add_arg("wn18_dir", str, "",
                    "wn18 dir")

    model_g.add_arg("mark", str, "test1", "mark")
    model_g.add_arg("tensorboard_dir", str, "./", "tensorboard_dir")
    model_g.add_arg("debug", bool, False, "debug")


    model_g.add_arg("model_name_or_path", str, "", "Path to pretrained model or model identifier from huggingface.co/models")
    model_g.add_arg("config_name", str, "", "Pretrained config name or path if not the same as model_name")
    model_g.add_arg("model_type", str, "kelm", "The classification model to be used.")
    model_g.add_arg("text_embed_model", str, "bert", "The model for embedding texts in KELM model.")
    model_g.add_arg("output_dir", str, "../outputs/test", "Path to save checkpoints.")
    model_g.add_arg("overwrite_output_dir", bool, True, "Overwrite the content of the output directory.")
    model_g.add_arg(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    model_g.add_arg("per_gpu_train_batch_size", int, 6, "Batch size per GPU/CPU for training.")
    model_g.add_arg("per_gpu_eval_batch_size", int, 4, "Batch size per GPU/CPU for evaluation.")
    model_g.add_arg("max_steps", int, -1, "If > 0: set total number of training steps to perform. Override num_train_epochs.")
    model_g.add_arg("gradient_accumulation_steps", int, 1, "Number of updates steps to accumulate before performing a backward/update pass.")
    model_g.add_arg("num_train_epochs", float, 10, "Total number of training epochs to perform.")
    model_g.add_arg("weight_decay", float, 0.01, "Weight decay if we apply some.")
    model_g.add_arg("learning_rate", float, 3e-4, "The initial learning rate for Adam.")
    model_g.add_arg("adam_epsilon", float, 1e-8, "Epsilon for Adam optimizer.")
    model_g.add_arg("warmup_steps", int, 10, "Linear warmup over warmup_steps.")
    model_g.add_arg("max_grad_norm", float, 1.0, "Max gradient norm.")
    model_g.add_arg("evaluate_steps", int, 2, "Evaluate every X updates steps.")
    model_g.add_arg("evaluate_epoch", float, 0.0, "evaluate every X update epoch")

    model_g.add_arg("save_steps", int, 1, "Save every X updates steps.")
    model_g.add_arg("evaluate_during_training", bool, True, "Run evaluation during training at each logging step.")
    model_g.add_arg("n_best_size", int, 20,
                    "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    model_g.add_arg("verbose_logging", bool, False, "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.")
    model_g.add_arg("init_dir", str, "", "The path of loading pre-trained model.")
    model_g.add_arg("mem_method", str, "raw", "The KG nodes embedding methods.")
    model_g.add_arg("initializer_range", float, 0.02, "The initializer range for KELM")
    model_g.add_arg("cat_mul", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_sub", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_twotime", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_twotime_mul", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_twotime_sub", bool, False, "The output part of vector in KELM")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("train_file", str, "record/train_0831.json", "ReCoRD json for training. E.g., train.json.")
    data_g.add_arg("predict_file", str, "record/dev_0831.json", "ReCoRD json for predictions. E.g. dev.json.")
    # data_g.add_arg("train_file", str, "record/train.json", "ReCoRD json for training. E.g., train.json.")
    # data_g.add_arg("predict_file", str, "record/dev.json", "ReCoRD json for predictions. E.g. dev.json.")
    data_g.add_arg("cache_file_suffix", str, "IDE_tt", "The suffix of cached file.")
    data_g.add_arg("cache_dir", str, "", "The cached data path.")
    data_g.add_arg("cache_store_dir", str, "", "The cached data path.")
    data_g.add_arg("data_dir", str, "", "The input data dir. Should contain the .json files for the task."
                   + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")

    data_g.add_arg("vocab_path", str, "vocab.txt", "Vocabulary path.")
    data_g.add_arg("do_lower_case", bool, False,
                   "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    data_g.add_arg("seed", int, 42, "Random seed.")
    data_g.add_arg("kg_paths", dict, {"wordnet":"kgs/", "nell": "kgs/"}, "The paths of knowledge graph files.")
    data_g.add_arg("wn_concept_embedding_path", str, "embedded/wn_concept2vec.txt",
                   "The embeddings of concept in knowledge graph : Wordnet.")
    data_g.add_arg("nell_concept_embedding_path", str, "embedded/nell_concept2vec.txt",
                   "The embeddings of concept in knowledge graph : Nell.")
    data_g.add_arg("use_kgs", list, ['nell', 'wordnet'], "The used knowledge graphs.")
    data_g.add_arg("doc_stride", int, 128,
                   "When splitting up a long document into chunks, how much stride to take between chunks.")
    data_g.add_arg("max_seq_length", int, 384, "Number of words of the longest seqence.")
    data_g.add_arg("max_query_length", int, 64, "Max query length.")
    data_g.add_arg("max_answer_length", int, 30, "Max answer length.")
    data_g.add_arg("no_stopwords", bool, True, "Whether to include stopwords.")
    data_g.add_arg("ignore_length", int, 0, "The smallest size of token.")
    data_g.add_arg("print_loss_step", int, 100, "The steps to print loss.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
    run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
    run_type_g.add_arg("max_n_gpu", int, 100, "The maximum number of GPU to use.")
    run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
    run_type_g.add_arg("num_iteration_per_drop_scope", int, 1,
                       "Ihe iteration intervals to clean up temporary variables.")
    run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
    run_type_g.add_arg("do_eval", bool, False, "Whether to perform evaluation during training.")
    run_type_g.add_arg("do_predict", bool, False, "Whether to perform prediction.")
    run_type_g.add_arg("freeze", bool, True, "freeze bert parameters")
    run_type_g.add_arg("server_ip", str, "", "Can be used for distant debugging.")
    run_type_g.add_arg("chunksize", int, 1024, "The chunksize for multiprocessing to convert examples to features.")
    run_type_g.add_arg("server_port", str, "", "Can be used for distant debugging.")
    run_type_g.add_arg("local_rank", int, -1, "Index for distributed training on gpus.")
    run_type_g.add_arg("threads", int, 50, "multiple threads for converting example to features")
    run_type_g.add_arg("overwrite_cache", bool, False, "Overwrite the cached training and evaluation sets")
    run_type_g.add_arg("eval_all_checkpoints", bool, False, "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    run_type_g.add_arg("min_diff_steps", int, 50, "The minimum saving steps before the last maximum steps.")
    args = parser.parse_args()

    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)  # Reduce model loading logs

    if not args.is_all_relation:
        args.relation_list = args.selected_relation.split(",")
        logger.info("not use all relation, relation_list: {}".format(args.relation_list))

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )


    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or not args.use_cuda:# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        args.n_gpu = 0 if not args.use_cuda else min(args.max_n_gpu, torch.cuda.device_count())
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    if args.local_rank in [-1, 0] and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.use_fp16,
    )

    # logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

    # Set seed
    set_seed(args)

    logger.info("Parameters from arguments are:\n{}".format(args))

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.use_fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.use_fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    processor = RecordProcessor(args)

    input_dir = os.path.join(args.cache_store_dir, "cached_{}_{}_{}".format(
            args.model_type,
            args.speed_up_version,
            str(args.cache_file_suffix),
    )
                             )

    args.wn_def_embed_mat_dir = os.path.join("./cache", "record_test_script_test_definition_embedding_mat")

    # if not os.path.exists(args.wn_def_embed_mat_dir):
    #     data_path = os.path.join(args.data_dir, args.kg_paths["wordnet"])
    #     definition_embedding_mat = create_definition_table(args, data_path)
    #
    #     torch.save({"definition_embedding_mat": definition_embedding_mat}, args.wn_def_embed_mat_dir)
    #
    #     logger.info("definition embedding is done. program exits.")
    #     exit()

    ## create data
    retrievers = dict()
    for kg in args.use_kgs:
        logger.info("Initialize kg:{}".format(kg))
        kg_path = os.path.join(input_dir, args.kg_paths[kg])
        data_path = os.path.join(args.data_dir, args.kg_paths[kg])

        retrievers[kg] = initialize_kg_retriever(kg, kg_path, data_path, args.cache_file_suffix)

    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        logger.info("begin preprocess")
        create_dataset(args, processor, retrievers, relation_list=args.relation_list, evaluate=True, input_dir=input_dir)

        logger.info("data preprocess is done")

    # Load pretrained model and tokenizers
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    tokenizer, model = configure_tokenizer_model(args, logger, retrievers)
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # model.update_definition_embedding_mat(
    #     torch.load(args.wn_def_embed_mat_dir)["definition_embedding_mat"])
    model.to(args.device)
    results = evaluate(args, model, processor, tokenizer, 100, input_dir)

    if args.local_rank in [-1, 0]:
        logger.info("results: {}".format(results))

    logger.info("eval is done")

if __name__ == "__main__":
    main()