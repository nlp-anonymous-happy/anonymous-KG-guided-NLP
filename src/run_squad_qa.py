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

from model.model_utils import configure_tokenizer_model
from kgs_retrieve.kg_utils import initialize_kg_retriever
from utils.args import ArgumentGroup
from text_processor.squad import SquadResult, SquadProcessor, create_input, DefinitionInfo
from dgl import save_graphs, load_graphs

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

sys.path.append('..')
sys.path.append('.')
# os.environ['CUDA_VISIBLE_DEVICES']='5, 6'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logger.info("running on GPU: {}".format(torch.cuda.current_device()))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, processor, tokenizer, retrievers, wn_synset_graphs, wn_synset_graphs_label_dict,
          input_dir):
    """ Train the model """
    logger.info("Training the model {}".format(args.model_type))
    if args.local_rank in [-1, 0] and args.mark != "test":
        tb_writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, args.mark), filename_suffix=args.mark)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # synset_graphs_batch = []
    # for batch_index in train_dataloader.batch_sampler:
    #     synset_graphs_batch.append([i for i in batch_index])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    named_parameters = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.schedule_strategy == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.schedule_strategy == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        logger.error("unknown schedule, program exits")
        exit()
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.output_dir, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.output_dir, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        map_location = {'cuda:%d' % 0: 'cuda:%d' % torch.distributed.get_rank()}
        logger.info("map_location when loading optimizer and scheduler: {}".format(map_location))
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt"), map_location=map_location))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt"), map_location=map_location))

    if args.use_fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.use_fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Dataset size = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.output_dir):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.output_dir.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_dic, logging_loss_dic = {}, {}

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_evals = dict()
    # epoch_counts = 0
    all_train_size = args.per_gpu_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    if args.evaluate_epoch:
        num_train_iteration = math.floor(len(train_dataloader) / (10 ** (len(str(len(train_dataloader))) - 1))) * (
                    10 ** (len(str(len(train_dataloader))) - 1))
        train_loss_record_steps = int(num_train_iteration / 8)
        first_record_point = int(num_train_iteration / 8)
        args.evaluate_steps = int(num_train_iteration * args.evaluate_epoch)
    else:
        train_loss_record_steps = int(500 * (24 / all_train_size))
        first_record_point = int(500 * (24 / all_train_size))
        args.evaluate_steps = int(args.evaluate_steps * (24 / all_train_size))

    for _ in train_iterator:

        if epochs_trained >= 1:
            if args.evaluate_epoch:
                args.evaluate_steps = int(num_train_iteration * (args.evaluate_epoch / 2))
            else:
                args.evaluate_steps = int(1000 * (24 / all_train_size))

        # shuffle dataset per epoch
        if args.local_rank != -1:
            train_sampler.set_epoch(epochs_trained)

        logger.info("args.evaluate_steps: {}".format(args.evaluate_steps))
        logger.info("epochs_trained: {}".format(epochs_trained))

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            def training_step(batch):
                model.train()

                batch = tuple(t.to(args.device) for t in batch)
                batch_synset_graphs = batch[3]

                inputs = create_input(args, batch, global_step, batch_synset_graphs=batch_synset_graphs,
                                      wn_synset_graphs=wn_synset_graphs)

                outputs = model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]
                if args.model_type in ["bert", "roberta", "roberta_base"]:
                    loss_dic = {
                        'loss_wn': torch.tensor(0.0, device=args.device),
                        'loss_cls': torch.tensor(0.0, device=args.device),
                    }
                else:
                    loss_dic = outputs[3]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    for k in loss_dic.keys():
                        loss_dic[k] = loss_dic[k].mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    for k in loss_dic.keys():
                        loss_dic[k] = loss_dic[k] / args.gradient_accumulation_steps

                if args.use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                return loss.detach(), loss_dic

            if (step + 1) % args.gradient_accumulation_steps != 0:
                with model.no_sync():
                    loss, loss_dic = training_step(batch)
            else:
                loss, loss_dic = training_step(batch)

            tr_loss += loss.item()
            if args.local_rank in [-1, 0] and (step + 1) % args.gradient_accumulation_steps == 0:
                logger.info('total loss during training {} at {}'.format(loss.item(), global_step))

            for k, v in loss_dic.items():
                if k not in tr_loss_dic:
                    tr_loss_dic[k] = 0.0
                if k not in logging_loss_dic:
                    logging_loss_dic[k] = 0.0
                tr_loss_dic[k] += v.item()
                if global_step >= args.warm_up and args.use_consistent_loss_wn and (
                        step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info('{} during training {} at {}'.format(k, v.item(), global_step))
                # if args.local_rank in [-1, 0] and (global_step == 1 or global_step == args.warm_up):
                #     tb_writer.add_scalar(f'loss/{k}', v.item(), global_step)

            if (step + 1) % args.gradient_accumulation_steps == 0:

                if args.local_rank in [-1, 0] and global_step == 1:
                    tb_writer.add_scalar("loss", tr_loss, int(global_step * all_train_size / 24))
                if args.local_rank in [-1, 0] and (global_step == 1 or global_step == args.warm_up):
                    for k, v in tr_loss_dic.items():
                        tb_writer.add_scalar(f'loss/{k}', v, int(global_step * all_train_size / 24))

                if args.use_fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Evaluate metrics
                if args.evaluate_steps > 0 and (global_step % args.evaluate_steps == 0 or (
                        global_step == first_record_point and epochs_trained == 0)) \
                        and args.evaluate_during_training:
                    logger.info("Evaluation during training:")
                    logger.info("Loss during training is {}".format(loss.item()))
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # if args.local_rank in [-1, 0] and args.evaluate_during_training:
                    # try:
                    results = evaluate(args, model, processor, tokenizer, global_step, input_dir)

                    if args.local_rank in [-1, 0]:
                        for key, value in results.items():
                            logger.info("eval_{}. Value: {}. In Step: {}".format(key, value, global_step))
                            if args.mark != "test":
                                tb_writer.add_scalar("eval_{}".format(key), value,
                                                     int(global_step * all_train_size / 24))
                            if key not in best_evals:
                                best_evals[key] = (value, global_step)
                            elif value > best_evals[key][0]:
                                if global_step - best_evals[key][1] > args.min_diff_steps and args.save_model:
                                    logger.info(
                                        "Saving the model since {} is improved from {} to {} at step {}".format(key,
                                                                                                                best_evals[
                                                                                                                    key][
                                                                                                                    0],
                                                                                                                value,
                                                                                                                global_step))
                                    # Save model checkpoint
                                    try:
                                        output_dir = os.path.join(args.output_dir,
                                                                  "checkpoint-{}".format(global_step))
                                        if not os.path.exists(output_dir):
                                            os.mkdir(output_dir)

                                        # Take care of distributed/parallel training
                                        model_to_save = model.module if hasattr(model, "module") else model
                                        if args.model_type in ["bert", "kelm", "roberta", "roberta_base"]:
                                            model_to_save.save_pretrained(output_dir)
                                            # tokenizer.save_pretrained(output_dir)
                                        else:
                                            logger.warning("Have no idea how to save the model.")

                                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                        logger.info("Saving arguments, optimizer and scheduler states to %s",
                                                    output_dir)
                                    except:
                                        logger.warning("Cannot save the checkpoints.")
                                    best_evals[key] = (value, global_step)
                        # # except:
                        # logger.info("To be processed Error for evaluate within train.")

                if global_step % train_loss_record_steps == 0:
                    if args.mark != "test" and args.local_rank in [-1, 0]:
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], int(global_step * all_train_size / 24))
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / train_loss_record_steps,
                                             int(global_step * all_train_size / 24))
                        for k, v in tr_loss_dic.items():
                            tb_writer.add_scalar(f'loss/{k}', (v - logging_loss_dic[k]) / train_loss_record_steps,
                                                 int(global_step * all_train_size / 24))

                    logging_loss = tr_loss
                    for k, v in tr_loss_dic.items():
                        logging_loss_dic[k] = v

                # if args.memory_bank_update and args.use_context_graph and args.speed_up_version == "v2" and global_step % args.memory_bank_update_steps == 0:
                #     model.eval()
                #     with torch.no_grad():
                #         start_time = timeit.default_timer()
                #         logger.info("cuda: {} updating entity description text embedding by the latest encoder".format(args.local_rank))
                #         t_list = torch.load(os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_info")["t_list"]
                #
                #         encoder = model.module.text_embed_model
                #         t_list_input_ids = t_list["input_ids"].to(encoder.device)
                #         if args.text_embed_model == "bert":
                #             t_list_token_type_ids = t_list["token_type_ids"].to(encoder.device)
                #         t_list_attention_mask = t_list["attention_mask"].to(encoder.device)
                #
                #         defid2defembed = torch.Tensor().to(encoder.device)
                #         c_size = 1024
                #         start_point = 0
                #         end_point = start_point + c_size
                #         total_size = len(t_list_input_ids)
                #         while True:
                #             if start_point > total_size:
                #                 break
                #             if end_point > total_size:
                #                 end_point = total_size
                #
                #             if args.text_embed_model == "bert":
                #                 tmp = encoder(input_ids=t_list_input_ids[start_point:end_point, :],
                #                               token_type_ids=t_list_token_type_ids[start_point:end_point, :],
                #                               attention_mask=t_list_attention_mask[start_point:end_point, :])[1]
                #             elif args.text_embed_model == "roberta" or args.text_embed_model == "roberta_base":
                #                 tmp = encoder(input_ids=t_list_input_ids[start_point:end_point, :],
                #                               attention_mask=t_list_attention_mask[start_point:end_point, :])[1]
                #             else:
                #                 logger.warning("not available LM, exit program")
                #                 exit()
                #             defid2defembed = torch.cat([defid2defembed, tmp], dim=0)
                #
                #             start_point += c_size
                #             end_point += c_size
                #
                #         model.module.update_defid2defembed(defid2defembed, args.memory_bank_keep_coef)
                #         logger.info("cuda: {} time for updating is {}".format(args.local_rank, timeit.default_timer() - start_time))
                #         logger.info("cuda: {} update is done".format(args.local_rank))

                # if args.memory_bank_update and global_step % args.memory_bank_update_steps == 0 and args.speed_up_version == "v2":
                #     with torch.no_grad():
                #         start_time = timeit.default_timer()
                #         logger.info("cuda: {} updating entity description text embedding by the latest encoder".format(
                #             args.local_rank))
                #         t_list = torch.load(os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_info")[
                #             "t_list"]
                #         logger.info("t_list size: {}".format(t_list["input_ids"].shape))
                #
                #         encoder = model.module.text_embed_model.cpu()
                #         t_list_input_ids = t_list["input_ids"]
                #         if args.text_embed_model == "bert":
                #             t_list_token_type_ids = t_list["token_type_ids"]
                #         t_list_attention_mask = t_list["attention_mask"]
                #
                #         defid2defembed = torch.Tensor()
                #         c_size = 10000
                #         start_point = 0
                #         end_point = start_point + c_size
                #         total_size = len(t_list_input_ids)
                #         while True:
                #             if start_point > total_size:
                #                 break
                #             if end_point > total_size:
                #                 end_point = total_size
                #
                #             if args.text_embed_model == "bert":
                #                 tmp = encoder(input_ids=t_list_input_ids[start_point:end_point, :],
                #                               token_type_ids=t_list_token_type_ids[start_point:end_point, :],
                #                               attention_mask=t_list_attention_mask[start_point:end_point, :])[1]
                #             elif args.text_embed_model == "roberta" or args.text_embed_model == "roberta_base":
                #                 tmp = encoder(input_ids=t_list_input_ids[start_point:end_point, :],
                #                               attention_mask=t_list_attention_mask[start_point:end_point, :])[1]
                #             else:
                #                 logger.warning("not available LM, exit program")
                #                 exit()
                #             defid2defembed = torch.cat([defid2defembed, tmp], dim=0)
                #
                #             start_point += c_size
                #             end_point += c_size
                #
                #         model.module.update_defid2defembed(defid2defembed.to(model.module.text_embed_model.device))
                #         logger.info("cuda: {} time for updating is {}".format(args.local_rank,
                #                                                               timeit.default_timer() - start_time))
                #         logger.info("cuda: {} update is done".format(args.local_rank))

            if args.max_steps > 0 and global_step > args.max_steps:
                logger.info("to max steps and stop iterator")
                epoch_iterator.close()
                break

        # if args.local_rank in [-1, 0]:
        #     logger.info("To check layers' parameters.")
        #     model_to_save = model.module if hasattr(model, "module") else model
        #     if args.model_type == "kelm":
        #         logger.info(model_to_save.text_embed_model.encoder.layer[0].output.dense.weight[:100, :100])
        #     elif args.model_type == "bert":
        #         logger.info(model_to_save.bert.encoder.layer[0].output.dense.weight[:100, :100])
        #
        #     if args.max_steps > 0 and global_step > args.max_steps:
        #         train_iterator.close()
        #         break
        #
        #     logger.info("Saving the model since an epoch ends")
        #     try:
        #         output_dir = os.path.join(args.output_dir,
        #                                   "epoch-{}-checkpoint-{}".format(epochs_trained, global_step))
        #         if not os.path.exists(output_dir):
        #             os.mkdir(output_dir)
        #
        #         # Take care of distributed/parallel training
        #         model_to_save = model.module if hasattr(model, "module") else model
        #         if args.model_type in ["bert", "kelm", "roberta", "roberta_base"]:
        #             model_to_save.save_pretrained(output_dir)
        #             # tokenizer.save_pretrained(output_dir)
        #         else:
        #             logger.warning("Have no idea how to save the model.")
        #
        #         torch.save(args, os.path.join(output_dir, "training_args.bin"))
        #         torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        #         torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        #         logger.info("Saving arguments, optimizer and scheduler states to %s",
        #                     output_dir)
        #     except:
        #         logger.warning("Cannot save the checkpoints.")
        epochs_trained += 1
    if args.local_rank in [-1, 0] and args.mark != "test":
        tb_writer.close()

    return global_step, tr_loss / global_step


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
                                input_dir=input_dir,
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
        logger.warning("program exits and please use pytorch DDP framework")
        exit()
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
            with torch.no_grad():
                inputs = create_input(args, batch, global_step, batch_synset_graphs=batch_synset_graphs,
                                      wn_synset_graphs=wn_synset_graphs, evaluate=True)
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

        all_start_logits_list = [torch.zeros_like(all_start_logits, device=args.device) for _ in
                                 range(torch.distributed.get_world_size())]
        all_end_logits_list = [torch.zeros_like(all_end_logits, device=args.device) for _ in
                               range(torch.distributed.get_world_size())]
        all_unique_ids_list = [torch.zeros_like(all_unique_ids, device=args.device) for _ in
                               range(torch.distributed.get_world_size())]

        all_gather(all_start_logits_list, all_start_logits)
        all_gather(all_end_logits_list, all_end_logits)
        all_gather(all_unique_ids_list, all_unique_ids)

        logger.info(
            "time for gather communication:{} in rank {}".format(timeit.default_timer() - start_time, args.local_rank))

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
                    result = SquadResult(int(unique_id.cpu().numpy()), start_logits, end_logits)

                    all_results.append(result)

            evalTime = timeit.default_timer() - start_time
            logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

            # Compute predictions
            output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
            output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

            predictions = SquadProcessor.compute_predictions_logits(
                examples_tokenized,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                args.verbose_logging,
                os.path.join(args.data_dir, args.predict_file),
                tokenizer
            )

            # Compute the F1 and exact scores.
            results = SquadProcessor.squad_evaluate(examples_tokenized, predictions, args.data_dir)
            return results
        else:
            return None


def load_and_cache_examples(args, processor, retrievers, relation_list, input_dir, evaluate=False,
                            output_examples=False):
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
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            args.model_type,
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
    tokenizer, encoder = configure_tokenizer_model(args, logger, retrievers, is_preprocess=True)
    logger.info("tokenizer: {}".format(tokenizer))
    logger.info("encoder: {}".format(encoder))

    encoder.to(args.device)
    for param in encoder.parameters():
        param.requires_grad = False

    if not evaluate:
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                args.model_type,
                str(args.cache_file_suffix),
            ),
        )

        if os.path.exists(cached_features_file):
            logger.warning("cache file exist and exit program")
            exit()

        logger.info("Creating features from dataset file at %s", input_dir)
        if not os.path.exists(cached_features_file + "_example"):
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
            logger.info("saving example")
            torch.save(examples, cached_features_file + "_example")

            # torch.save(examples, cached_features_file + "_example")
        # else:
        #     logger.info("Loading examples from cached files.")
        #     examples = torch.load(cached_features_file + "_example")

        if not os.path.exists(cached_features_file + "_example_tokenized"):
            examples_tokenized = processor.tokenization_on_examples(examples, tokenizer)
            logger.info("saving examples_tokenized")
            torch.save(examples_tokenized, cached_features_file + "_examples_tokenized")
        #     torch.save(examples_tokenized, cached_features_file + "_example_tokenized")
        # else:
        #     logger.info("Loading tokenized examples from cached files.")
        #     examples_tokenized = torch.load(cached_features_file + "_example_tokenized")
        if not os.path.exists(cached_features_file + '_features_no_pad'):
            features = processor.convert_examples_to_features(args, examples_tokenized, tokenizer, retrievers,
                                                              not evaluate, debug=args.debug)
            logger.info("saving features")
            torch.save(features, cached_features_file + "_features")
            # logger.info("saving repeated_id...")
            # repeated_id_path = os.path.join(retrievers["wordnet"].filepath, "repeated_id_train.npy")
            # np.save(repeated_id_path, retrievers["wordnet"].repeated_id)
            # examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            # examples_tokenized = processor.tokenization_on_examples(examples, tokenizer)
            # features = processor.convert_examples_to_features(args, examples_tokenized, tokenizer, retrievers, False, debug=args.debug)
            # repeated_id_path = os.path.join(retrievers["wordnet"].filepath, "repeated_id.npy")
            # np.save(repeated_id_path, retrievers["wordnet"].repeated_id)
            # exit()

        #     torch.save(features, cached_features_file + '_features_no_pad')
        # else:
        #     logger.info("Loading no padded features from cached files.")
        #     features = torch.load(cached_features_file + '_features_no_pad')

        features, dataset, all_kgs_graphs = processor.pad_and_index_features_all(
            features, retrievers, args, tokenizer, relation_list, encoder=encoder, definition_info=definition_info,
            is_training=not evaluate, debug=args.debug)

        if args.local_rank in [-1, 0]:
            # wn_synset_graphs = all_kgs_synset_graphs["wordnet"]
            if args.model_type in ["bert", "roberta", "roberta_base"]:
                all_kgs_graphs_label_dict = None
            else:
                all_kgs_graphs_label_dict = {"glabel": torch.tensor([i for i in range(len(all_kgs_graphs))])}
                save_graphs(cached_features_file + "_all_kgs_graphs.bin", all_kgs_graphs, all_kgs_graphs_label_dict)
                logger.info("complete data preprocessing")

            logger.info("Saving features into cached file %s", cached_features_file)

            torch.save({"features": None, "dataset": dataset, "examples": examples_tokenized}, cached_features_file)

            logger.info("Saving knowledge graph retrievers")

            for kg, retriever in retrievers.items():
                if not os.path.exists(os.path.join(input_dir, args.kg_paths[kg])):
                    os.mkdir(os.path.join(input_dir, args.kg_paths[kg]))
                torch.save(retriever, os.path.join(input_dir, args.kg_paths[kg], kg + args.cache_file_suffix))

            logger.info("saving definition information ...")
            torch.save({"defid2def": definition_info.defid2def, "conceptid2defid": definition_info.conceptid2defid},
                       os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_info")
            assert len(definition_info.defid2def) == len(definition_info.conceptid2defid)

            logger.info("training data create is done")

    else:
        stored_definition_info = torch.load(os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_info")
        definition_info.defid2def, definition_info.conceptid2defid = stored_definition_info["defid2def"], \
                                                                     stored_definition_info["conceptid2defid"]
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                args.model_type,
                str(args.cache_file_suffix),
            ),
        )

        if os.path.exists(cached_features_file):
            logger.warning("cache file exist and exit program")
            exit()

        logger.info("Creating features from dataset file at %s", input_dir)

        if not os.path.exists(cached_features_file + "_example"):
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            torch.save(examples, cached_features_file + "_example")
        else:
            logger.info("Loading examples from cached files.")
            examples = torch.load(cached_features_file + "_example")

        if not os.path.exists(cached_features_file + "_example_tokenized"):
            examples_tokenized = processor.tokenization_on_examples(examples, tokenizer)
        #     torch.save(examples_tokenized, cached_features_file + "_example_tokenized")
        # else:
        #     logger.info("Loading tokenized examples from cached files.")
        #     examples_tokenized = torch.load(cached_features_file + "_example_tokenized")
        if not os.path.exists(cached_features_file + '_features_no_pad'):
            features = processor.convert_examples_to_features(args, examples_tokenized, tokenizer, retrievers,
                                                              not evaluate, debug=args.debug)

            # logger.info("saving repeated_id...")
            # repeated_id_path = os.path.join(retrievers["wordnet"].filepath, "repeated_id_train.npy")
            # np.save(repeated_id_path, retrievers["wordnet"].repeated_id)
            # examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            # examples_tokenized = processor.tokenization_on_examples(examples, tokenizer)
            # features = processor.convert_examples_to_features(args, examples_tokenized, tokenizer, retrievers, False, debug=args.debug)
            # repeated_id_path = os.path.join(retrievers["wordnet"].filepath, "repeated_id.npy")
            # np.save(repeated_id_path, retrievers["wordnet"].repeated_id)
            # exit()

        #     torch.save(features, cached_features_file + '_features_no_pad')
        # else:
        #     logger.info("Loading no padded features from cached files.")
        #     features = torch.load(cached_features_file + '_features_no_pad')

        features, dataset, all_kgs_graphs = processor.pad_and_index_features_all(
            features, retrievers, args, tokenizer, relation_list, encoder=encoder, definition_info=definition_info,
            is_training=not evaluate, debug=args.debug)

        if args.local_rank in [-1, 0]:
            # wn_synset_graphs = all_kgs_synset_graphs["wordnet"]
            if args.model_type in ["bert", "roberta", "roberta_base"]:
                all_kgs_graphs_label_dict = None
            else:
                all_kgs_graphs_label_dict = {"glabel": torch.tensor([i for i in range(len(all_kgs_graphs))])}
                save_graphs(cached_features_file + "_all_kgs_graphs.bin", all_kgs_graphs, all_kgs_graphs_label_dict)
                logger.info("complete data preprocessing")

            logger.info("Saving features into cached file %s", cached_features_file)

            for f in features:
                del f.kgs_conceptids2synset
            torch.save({"features": features, "dataset": dataset, "examples": examples_tokenized}, cached_features_file)

            if args.model_type == "kelm":
                logger.info("saving definition embedding")

                t_list = tokenizer(definition_info.defid2def, return_tensors="pt", padding=True)

                t_list_input_ids = t_list["input_ids"].to(encoder.device)
                if args.text_embed_model == "bert":
                    t_list_token_type_ids = t_list["token_type_ids"].to(encoder.device)
                t_list_attention_mask = t_list["attention_mask"].to(encoder.device)

                defid2defembed = torch.Tensor()
                c_size = 1024
                start_point = 0
                end_point = start_point + c_size
                total_size = len(t_list_input_ids)
                while True:
                    if start_point > total_size:
                        break
                    if end_point > total_size:
                        end_point = total_size

                    if args.text_embed_model == "bert":
                        tmp = encoder(input_ids=t_list_input_ids[start_point:end_point, :],
                                      token_type_ids=t_list_token_type_ids[start_point:end_point, :],
                                      attention_mask=t_list_attention_mask[start_point:end_point, :])[1].cpu()
                    elif args.text_embed_model == "roberta" or args.text_embed_model == "roberta_base":
                        tmp = encoder(input_ids=t_list_input_ids[start_point:end_point, :],
                                      attention_mask=t_list_attention_mask[start_point:end_point, :])[1].cpu()
                    else:
                        logger.warning("not available LM, exit program")
                        exit()
                    defid2defembed = torch.cat([defid2defembed, tmp], dim=0)

                    start_point += c_size
                    end_point += c_size

                assert defid2defembed.shape[0] == total_size
            else:
                t_list = []
                defid2defembed = []

            logger.info("Saving knowledge graph retrievers")
            for kg, retriever in retrievers.items():
                torch.save(retriever, os.path.join(input_dir, args.kg_paths[kg], kg + args.cache_file_suffix))

            logger.info("saving definition embedding ...")
            torch.save({"defid2defembed": defid2defembed},
                       os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_embedding")

            logger.info("saving definition information ...")
            torch.save({"defid2def": definition_info.defid2def, "conceptid2defid": definition_info.conceptid2defid,
                        "t_list": t_list},
                       os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_info")
            assert len(definition_info.defid2def) == len(definition_info.conceptid2defid)

            logger.info("data create is done")

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    exit()


def main():
    parser = argparse.ArgumentParser()

    model_g = ArgumentGroup(parser, "model", "model configuration and path.")

    model_g.add_arg("dataset", str, "squad", "used dataset")
    model_g.add_arg("is_update_max_concept", bool, False, "weather update max concept for kg retriver")
    model_g.add_arg("full_table", bool, False, "full_table")
    model_g.add_arg("test", bool, False, "weather load test set")
    model_g.add_arg("use_wn", bool, True, "wn")
    model_g.add_arg("use_nell", bool, True, "nell")

    model_g.add_arg("sentinel_trainable", bool, False, "sentinel_trainable")
    model_g.add_arg("memory_bank_update", bool, False, "memory_bank_update")
    model_g.add_arg("memory_bank_update_steps", int, 500, "memory_bank_update_steps")
    model_g.add_arg("memory_bank_keep_coef", float, 0.0, "what percent keep")
    model_g.add_arg("use_context_graph", bool, True, "use_context_graph")

    model_g.add_arg("schedule_strategy", str, "linear", "schedule_strategy")
    model_g.add_arg("tokenizer_path", str, "../cache/bert-large-cased/", "tokenizer_path")
    model_g.add_arg("save_model", bool, True, "whether save model")
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

    model_g.add_arg("relation_list", list,
                    ["_hyponym", "_hypernym", "_derivationally_related_form", "_member_meronym", "_member_holonym",
                     "_part_of", "_has_part", "_member_of_domain_topic", "_synset_domain_topic_of", "_instance_hyponym",
                     "_instance_hypernym", "_also_see", "_verb_group", "_member_of_domain_region",
                     "_synset_domain_region_of", "_member_of_domain_usage", "_synset_domain_usage_of", "_similar_to"],
                    "The used relation.")
    model_g.add_arg("is_all_relation", bool, False, "use all relations")
    model_g.add_arg("selected_relation", str, "_hyponym,_hypernym,_derivationally_related_form", "relations")
    model_g.add_arg("wn18_dir", str, "../data/kgs/wn18/text/", "wn18 dir")

    # SSL part
    model_g.add_arg("use_consistent_loss_wn", bool, False, "add consistent loss between entity embedding from WN.")
    model_g.add_arg("warm_up", int, 10000, "warm_up_iterations")
    model_g.add_arg("consistent_loss_wn_coeff", float, 2.0, "Weight decay if we apply some.")
    model_g.add_arg("consistent_loss_type", str, "kld", "consistent loss type")
    model_g.add_arg("mark", str, "test1", "mark")
    model_g.add_arg("tensorboard_dir", str, "./", "tensorboard_dir")
    model_g.add_arg("debug", bool, False, "debug")

    model_g.add_arg("model_name_or_path", str, "../cache/bert-large-cased/",
                    "Path to pretrained model or model identifier from huggingface.co/models")
    model_g.add_arg("config_name", str, "../cache/bert-large-cased/",
                    "Pretrained config name or path if not the same as model_name")
    model_g.add_arg("model_type", str, "kelm", "The classification model to be used.")
    model_g.add_arg("text_embed_model", str, "bert", "The model for embedding texts in kelm model.")
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
    model_g.add_arg("max_steps", int, -1,
                    "If > 0: set total number of training steps to perform. Override num_train_epochs.")
    model_g.add_arg("gradient_accumulation_steps", int, 1,
                    "Number of updates steps to accumulate before performing a backward/update pass.")
    model_g.add_arg("num_train_epochs", float, 10, "Total number of training epochs to perform.")
    model_g.add_arg("weight_decay", float, 0.0, "Weight decay if we apply some.")
    model_g.add_arg("learning_rate", float, 3e-4, "The initial learning rate for Adam.")
    model_g.add_arg("adam_epsilon", float, 1e-8, "Epsilon for Adam optimizer.")
    model_g.add_arg("warmup_steps", int, 0, "Linear warmup over warmup_steps.")
    model_g.add_arg("max_grad_norm", float, 1.0, "Max gradient norm.")
    model_g.add_arg("evaluate_steps", int, 2, "Evaluate every X updates steps.")
    model_g.add_arg("evaluate_epoch", float, 0.0, "evaluate every X update epoch")

    model_g.add_arg("save_steps", int, 1, "Save every X updates steps.")
    model_g.add_arg("evaluate_during_training", bool, True, "Run evaluation during training at each logging step.")
    model_g.add_arg("n_best_size", int, 20,
                    "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    model_g.add_arg("verbose_logging", bool, False,
                    "If true, all of the warnings related to data processing will be printed. "
                    "A number of warnings are expected for a normal SQuAD evaluation.")
    model_g.add_arg("init_dir", str, "../cache/bert-large-cased/", "The path of loading pre-trained model.")
    model_g.add_arg("initializer_range", float, 0.02, "The initializer range for KELM")
    model_g.add_arg("cat_mul", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_sub", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_twotime", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_twotime_mul", bool, True, "The output part of vector in KELM")
    model_g.add_arg("cat_twotime_sub", bool, False, "The output part of vector in KELM")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("train_file", str, "squad/train-v1.1.tagged.json", "SQUAD json for training. E.g., train.json.")
    data_g.add_arg("predict_file", str, "squad/dev-v1.1.tagged.json", "SQUAD json for predictions. E.g. dev.json.")

    data_g.add_arg("cache_file_suffix", str, "test", "The suffix of cached file.")
    data_g.add_arg("cache_dir", str, "../cache/", "The cached data path.")
    data_g.add_arg("cache_store_dir", str, "../cache/", "The cached data path.")
    data_g.add_arg("data_dir", str, "../data/", "The input data dir. Should contain the .json files for the task."
                   + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")

    data_g.add_arg("vocab_path", str, "vocab.txt", "Vocabulary path.")
    data_g.add_arg("do_lower_case", bool, False,
                   "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    data_g.add_arg("seed", int, 42, "Random seed.")
    data_g.add_arg("kg_paths", dict, {"wordnet": "kgs/", "nell": "kgs/"}, "The paths of knowledge graph files.")
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
    run_type_g.add_arg("eval_all_checkpoints", bool, False,
                       "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    run_type_g.add_arg("min_diff_steps", int, 50, "The minimum saving steps before the last maximum steps.")
    args = parser.parse_args()
    logger.info(args)
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
    if args.local_rank == -1 or not args.use_cuda:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
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

    processor = SquadProcessor(args)

    input_dir = os.path.join(args.cache_store_dir, "cached_{}_{}".format(
        args.model_type,
        str(args.cache_file_suffix),
    )
                             )
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    if args.data_preprocess:
        retrievers = dict()
        for kg in args.use_kgs:
            logger.info("Initialize kg:{}".format(kg))
            kg_path = os.path.join(input_dir, args.kg_paths[kg])
            data_path = os.path.join(args.data_dir, args.kg_paths[kg])

            if args.data_preprocess_evaluate and \
                    (not os.path.exists(kg_path) or not os.path.join(input_dir,
                                                                     args.cache_file_suffix) + "_" + "definition_info"):
                logger.warning("need prepare training dataset firstly, program exit")
                exit()

            retrievers[kg] = initialize_kg_retriever(kg, kg_path, data_path, args.cache_file_suffix)

        create_dataset(args, processor, retrievers, relation_list=args.relation_list,
                       evaluate=args.data_preprocess_evaluate, input_dir=input_dir)

        logger.info("data preprocess is done. program exits")
        exit()

    args.wn_def_embed_mat_dir = os.path.join(input_dir, args.cache_file_suffix) + "_" + "definition_embedding"
    # Training
    if args.do_train:
        retrievers = dict()
        for kg in args.use_kgs:
            logger.info("Initialize kg:{}".format(kg))
            kg_path = os.path.join(input_dir, args.kg_paths[kg])
            data_path = os.path.join(args.data_dir, args.kg_paths[kg])

            if not os.path.exists(kg_path):
                logger.warning("need prepare training dataset firstly, program exit")
                exit()

            retrievers[kg] = initialize_kg_retriever(kg, kg_path, data_path, args.cache_file_suffix)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()
        tokenizer, model = configure_tokenizer_model(args, logger, retrievers)
        if args.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        if args.do_eval:
            model.to(args.device)
            results = evaluate(args, model, processor, tokenizer, 100, input_dir)

            if args.local_rank in [-1, 0]:
                logger.info("results: {}".format(results))

            logger.info("eval is done")
            exit()

        train_dataset, wn_synset_graphs, wn_synset_graphs_label_dict = load_and_cache_examples(args,
                                                                                               processor,
                                                                                               retrievers,
                                                                                               relation_list=args.relation_list,
                                                                                               input_dir=input_dir,
                                                                                               evaluate=False,
                                                                                               output_examples=False)
        model.to(args.device)
        global_step, tr_loss = train(args, train_dataset, model, processor, tokenizer, retrievers, wn_synset_graphs,
                                     wn_synset_graphs_label_dict, input_dir)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving trained model to %s", args.output_dir)
        # Save a trained model, configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
