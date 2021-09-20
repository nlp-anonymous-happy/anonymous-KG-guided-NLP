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

import torch
import os
import copy
import unicodedata
from tqdm import tqdm
import collections
import logging
import math
import six
from functools import partial
from torch.multiprocessing import Pool, cpu_count

from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import DataProcessor

from text_processor import tokenization
from utils.squad_1_1_evaluate import *
from transformers import RobertaTokenizer, BertTokenizer
import networkx as nx
import dgl
from dgl.data.utils import save_graphs
from kgs_retrieve.baseretriever import KGRetriever, read_concept_embedding, run_strip_accents
from nltk.corpus import wordnet as wn
import numpy as np
from text_processor.wn_relational_graph_builder import RelationalGraphBuilder

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

_DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
)

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"]
)

# remove_special_punc_count = 0

class DefinitionInfo(object):
    def __init__(
            self,
            defid2def=[],
            conceptid2defid={},
            # defid2defembed=None,
    ):
        self.defid2def = defid2def
        self.conceptid2defid = conceptid2defid
        # self.defid2defembed = defid2defembed


class SquadExample(object):
    """
    A single training/test example for the squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        doc_text: The context string
        answer_entities: list of dict: [{'orig_text': entity_text, 'start_position': entity_start_offset, 'end_position': entity_end_offset})]
            entity_start/end_offset: the index of the start's/end's char of the entity in the text.
        passage_entities: list of dict: [{'orig_text': entity_text,
                                                'start_position': entity_start_offset,
                                                'end_position': entity_end_offset})]
        e.g. text: "Today is Obama's birthday.". the length of the text is: 25
            entity_text: Obama
            start_position: 9
            end_position: 13

    """

    def __init__(
            self,
            qas_id,
            question_text,
            question_entities_strset,
            doc_text,
            answer_entities,
            passage_entities,
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.question_entities_strset = question_entities_strset
        self.doc_text = doc_text
        self.answer_entities = answer_entities
        self.passage_entities = passage_entities
        self.is_impossible = is_impossible

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_text: [%s]" % (tokenization.printable_text(self.doc_text))
        s += ", answer entity: [%s]" % (" ".join(answer['orig_text'] for answer in self.answer_entities))
        return s

    def __str__(self):
        return self.__repr__()


class SquadExampleTokenized(object):
    def __init__(self,
                 id,
                 query_text,
                 query_tokens,
                 query_ori_to_tok_map,
                 query_tok_to_ori_map,
                 query_entities,
                 doc_text,
                 doc_ori_text,
                 doc_tokens,
                 doc_tok_to_ori_map,
                 doc_ori_to_tok_map,
                 doc_entities,
                 answer_entities):
        """
        :param id: qas-id.
        :param query_text: the processed question string.
            e.g.
        :param query_tokens: the basic units for embedding. If using Bert model, here means subtokens given by BertTokenizer.
        :param query_ori_to_tok_map: the map from the index of char in the text to the index of subtoken in the query_tokens.
            e.g.
        :param query_tok_to_ori_map:
        :param query_entities:
        :param doc_text:
        :param doc_ori_text:
        :param doc_tokens:
        :param doc_tok_to_ori_map:
        :param doc_ori_to_tok_map:
        :param doc_entities:
        :param answer_entities:
        """
        self.id = id
        self.query_text = query_text
        self.query_tokens = query_tokens
        self.query_ori_to_tok_map = query_ori_to_tok_map
        self.query_tok_to_ori_map = query_tok_to_ori_map
        self.query_entities = query_entities
        self.doc_text = doc_text
        self.doc_ori_text = doc_ori_text
        self.doc_tokens = doc_tokens
        self.doc_tok_to_ori_map = doc_tok_to_ori_map
        self.doc_ori_to_tok_map = doc_ori_to_tok_map
        self.doc_entities = doc_entities
        self.answer_entities = answer_entities

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.query_text))
        s += ", doc_text: [%s]" % (tokenization.printable_text(self.doc_ori_text))
        s += ", answer entity: [%s]" % (" ".join(answer['orig_text'] for answer in self.answer_entities))

        return s


class SquadProcessor(DataProcessor):
    """
    Processor for the squad data set.
    """

    def __init__(self, args):
        self.name = "squad"
        self.do_lower_case = args.do_lower_case
        self.doc_stride = args.doc_stride
        self.use_kgs = args.use_kgs
        self.max_query_length = args.max_query_length
        self.max_seq_length = args.max_seq_length
        self.no_stopwords = args.no_stopwords
        self.ignore_length = args.ignore_length
        self.is_filter = args.is_filter
        self.is_lemma = args.is_lemma
        self.is_clean = args.is_clean
        self.fewer_label = args.fewer_label
        self.label_rate = args.label_rate
        self.is_morphy = args.is_morphy

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""
        logger.info("****** Get train examples ******")
        filename = os.path.join(data_dir, filename)
        return self._create_examples(filename, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        logger.info("****** Get dev examples ******")
        filename = os.path.join(data_dir, filename)
        return self._create_examples(filename, "dev")

    def convert_example_to_features(self, example, tokenizer, retrievers, is_training):
        """
        :param example:
        :param args:
        :param tokenizer:
        :param is_training:
        :return:
        """
        features = []
        max_retrieved = dict()
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer_type = "roberta"
        elif isinstance(tokenizer, BertTokenizer):
            self.tokenizer_type = "bert"
        else:
            ValueError

        if example is None:
            return features, max_retrieved
        query_tokens = example.query_tokens
        doc_tokens = example.doc_tokens
        tok_len = len(query_tokens)
        if tok_len > self.max_query_length:
            query_tokens = query_tokens[:self.max_query_length]

        # get the query and document tokens' corresponding concepts.
        query_kgs_concepts = dict()
        doc_kgs_concepts = dict()
        kgs_query_conceptids2synset = dict()
        kgs_doc_conceptids2synset = dict()

        for kg_info in self.use_kgs:
            # extract tokens and their corresponding concept ids.
            KGRetriever = retrievers[kg_info]
            args_dict = dict()
            args_dict['do_lower_case'] = self.do_lower_case
            if kg_info == "wordnet":
                args_dict['no_stopwords'] = self.no_stopwords
                args_dict['ignore_length'] = self.ignore_length
                args_dict['is_filter'] = self.is_filter
                args_dict['is_lemma'] = self.is_lemma
                args_dict['is_clean'] = self.is_clean
                args_dict['is_morphy'] = self.is_morphy

                query_kg_concept_ids, doc_kg_concept_ids, max_concept_length, query_conceptids2synset, doc_conceptids2synset = \
                    KGRetriever.lookup_concept_ids(example, tokenizer, **args_dict)
                # to guarantee the length of tokens doesn't exceed the limitation.
                if tok_len > self.max_query_length:
                    query_kg_concept_ids = query_kg_concept_ids[: self.max_query_length]

                kgs_query_conceptids2synset[kg_info] = query_conceptids2synset
                kgs_doc_conceptids2synset[kg_info] = doc_conceptids2synset

            else:
                query_kg_concept_ids, doc_kg_concept_ids, max_concept_length = KGRetriever.lookup_concept_ids(example,
                                                                                                              **args_dict)

                # to guarantee the length of tokens doesn't exceed the limitation.
                if tok_len > self.max_query_length:
                    query_kg_concept_ids = query_kg_concept_ids[: self.max_query_length]
                kgs_query_conceptids2synset[kg_info] = []
                kgs_doc_conceptids2synset[kg_info] = []

            query_kgs_concepts[kg_info] = query_kg_concept_ids
            doc_kgs_concepts[kg_info] = doc_kg_concept_ids
            max_retrieved[kg_info] = max_concept_length

        # build the former part of tokens. Namely, add functional tokens in the query tokens..
        former_tokens, former_segment_ids, former_kgs_concept_ids = build_first_part_features(
            self.tokenizer_type,
            self.use_kgs,
            query_tokens,
            query_kgs_concepts,
            )


        # cut the long doc tokens into shorter spans. Distinguish the training mode and valiation mode.
        # In the training mode, only keeps the doc_span including answer entities with the requirement that
        # each span (1) only includes one answer and (2) the doc_span length is smaller than the max_length.
        # In the validatiaon mode, just cut the document into several overlapped spans.

        former_len = len(former_tokens)
        max_tokens_for_doc = self.max_seq_length - former_len - 1

        # get all the answers and their positions
        answer_spans = []  # item: tuple (answer_start, answer_end)
        for answer_entity in example.answer_entities:
            answer_spans.append((answer_entity[1], answer_entity[2]))
        answer_spans = sorted(answer_spans, key=lambda t: t[0])

        doc_spans = get_doc_spans(len(example.doc_tokens), max_tokens_for_doc, self.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            start_position, end_position = None, None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                start_position, end_position = get_answer_position(doc_span, answer_spans)
                if start_position == -1:
                    continue

            tokens = copy.deepcopy(former_tokens)
            segment_ids = copy.deepcopy(former_segment_ids)
            kgs_concept_ids = copy.deepcopy(former_kgs_concept_ids)
            tok_offset = len(tokens)

            # if (doc_span.start + doc_span.length) < len(doc_tokens):
            #     logger.warning("warning!  need to deal with graph data")
            #     exit()
            tokens.extend(doc_tokens[doc_span.start: (doc_span.start + doc_span.length)])
            segment_ids.extend([1] * doc_span.length)
            for kg in self.use_kgs:
                kgs_concept_ids[kg].extend(doc_kgs_concepts[kg][doc_span.start: (doc_span.start + doc_span.length)])

            if self.tokenizer_type == "roberta":
                tokens.append("</s>")
            else:
                tokens.append("[SEP]")
            segment_ids.append(1)
            for kg in self.use_kgs:
                kgs_concept_ids[kg].append([])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            if start_position is not None:
                start_position = start_position - doc_span.start + tok_offset
                end_position = end_position - doc_span.start + tok_offset

            kgs_conceptids2synset = {}
            for kg in self.use_kgs:
                conceptids2synset = {}
                if kg == "wordnet":
                    query_conceptids2synset = kgs_query_conceptids2synset[kg]
                    doc_conceptids2synset = kgs_doc_conceptids2synset[kg]

                    for k, v in query_conceptids2synset.items():
                        if k in conceptids2synset:
                            continue

                        if len(str(v).split("'")) == 3:
                            conceptids2synset[k] = str(v).split("'")[1]
                        else:
                            conceptids2synset[k] = str(v).replace("')", "('").split("('")[1]

                        try:
                            wn.synset(conceptids2synset[k])
                        except:
                            logger.warning("error!!!!!!!!!!!! wrong synset:{}".format(conceptids2synset[k]))
                            exit()

                    for k, v in doc_conceptids2synset.items():
                        if k in conceptids2synset:
                            continue

                        if len(str(v).split("'")) == 3:
                            conceptids2synset[k] = str(v).split("'")[1]
                        else:
                            conceptids2synset[k] = str(v).replace("')", "('").split("('")[1]

                        try:
                            wn.synset(conceptids2synset[k])
                        except:
                            logger.warning("error!!!!!!!!!!!! wrong synset:{}".format(conceptids2synset[k]))
                            exit()
                    # conceptids2synset = []
                    # for k, v in query_conceptids2synset.items():
                    #     conceptids2synset.append([k, "good"])
                kgs_conceptids2synset[kg] = conceptids2synset

            feature = SquadFeature(
                qas_id=example.id,
                example_index=0,
                unique_id=0,
                tokens=tokens,
                tokid_span_to_orig_map=(tok_offset, doc_span.start, doc_span.length),
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                kgs_concept_ids=kgs_concept_ids,
                kgs_conceptids2synset=kgs_conceptids2synset,
            )

            features.append(feature)
        return features, max_retrieved

    def convert_examples_to_features(self,
                                     args,
                                     examples,
                                     tokenizer,
                                     retrievers,
                                     is_training,
                                     tqdm_enabled=True,
                                     debug=False,
                                     ):
        threads = min(args.threads, cpu_count())

        if debug:
            results = []
            logger.info("testing convert_example_to_features function")
            for example in tqdm(examples):
                result = self.convert_example_to_features(example, tokenizer, retrievers, is_training)
                results.append(result)

        else:
            logger.info("Using {} threads to convert {} examples to features".format(threads,
                                                                                     "train" if is_training else "dev"))
            with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
                annotate_ = partial(
                    self.convert_example_to_features,
                    tokenizer=tokenizer,
                    is_training=is_training,
                    retrievers=retrievers,
                )
                results = list(
                    tqdm(
                        p.imap(annotate_, examples, chunksize=args.chunksize),
                        total=len(examples),
                        desc="convert squad examples to features",
                        disable=not tqdm_enabled,
                    )
                )
        features = [item[0] for item in results if len(item[0]) > 0]
        if len(features) == 0:
            return features

        for kg in self.use_kgs:
            val = retrievers[kg]
            kg_max = max([item[1][kg] for item in results if len(item[0]) > 0])
            if is_training or args.is_update_max_concept:
                val.update_max_concept_length(kg_max)

        return features

    def pad_and_index_features(self, features, retrievers):
        # Zero-pad up to the sequence length.
        new_features = []
        for feature in features:

            # w = feature.kgs_concept_ids["wordnet"]
            # check_concept_list(w)

            pad_len = self.max_seq_length - len(feature.input_ids)
            feature.input_ids.extend([0] * pad_len)
            feature.attention_mask.extend([0] * pad_len)
            feature.token_type_ids.extend([0] * pad_len)
            for kg in self.use_kgs:
                feature.kgs_concept_ids[kg].extend([[] for _ in range(pad_len)])
                if len(feature.kgs_concept_ids[kg]) != self.max_seq_length:
                    logger.warning(
                        "Feature qas-id: {} has {} different concepts length {} with max seq length {}".format(
                            feature.qas_id, kg, len(feature.kgs_concept_ids[kg]), self.max_seq_length))

            assert len(feature.input_ids) == self.max_seq_length
            assert len(feature.attention_mask) == self.max_seq_length
            assert len(feature.token_type_ids) == self.max_seq_length

            # w = feature.kgs_concept_ids["wordnet"]
            # check_concept_list(w)
            # pad kg concepts
            for kg in self.use_kgs:
                concept_ids = feature.kgs_concept_ids[kg]
                kg_max_len = retrievers[kg].get_concept_max_length()
                for cindex in range(self.max_seq_length):
                    expaned_concept_ids = concept_ids[cindex] + [0 for _ in range(
                        kg_max_len - len(concept_ids[cindex]))]
                    concept_ids[cindex] = expaned_concept_ids[:kg_max_len]
                assert all([len(id_list) == kg_max_len for id_list in concept_ids])

                feature.kgs_concept_ids[kg] = concept_ids

            new_features.append(feature)
        return new_features

    def pad_and_index_features_all(self, features, retrievers, args, tokenizer, relation_list, encoder, definition_info,
                                   is_training, tqdm_enabled=True, debug=False):
        threads = min(args.threads, cpu_count())

        logger.info("Using {} threads to pad features".format(threads))
        for kg in self.use_kgs:
            ret = retrievers[kg]
            logger.info("KG {}'s max retrieved concpet length {}".format(kg, ret.get_concept_max_length()))

        if debug:
            for feature in features:
                self.pad_and_index_features(feature, retrievers)

        with Pool(threads) as p:
            annotate_ = partial(
                self.pad_and_index_features,
                retrievers=retrievers,
            )
            padded_features = list(
                tqdm(
                    p.imap(annotate_, features, chunksize=args.chunksize),
                    total=len(features),
                    desc="pad features",
                    disable=not tqdm_enabled,
                )
            )

        new_features = []
        unique_id = 1000000000
        example_index = 0
        for example_features in tqdm(
                padded_features, total=len(padded_features), desc="add example index and unique id",
                disable=not tqdm_enabled
        ):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                new_features.append(example_feature)
                unique_id += 1
            example_index += 1
        features = new_features
        del new_features
        # Convert to Tensors and build dataset
        dataset, all_kgs_graphs = create_tensordataset(features, is_training, args, retrievers, tokenizer,
                                                       relation_list, encoder=encoder, definition_info=definition_info,
                                                       debug=args.debug)
        return features, dataset, all_kgs_graphs

    def tokenization_on_example(self, example, tokenizer):
        if isinstance(tokenizer, RobertaTokenizer):
            return self.tokenization_on_example_roberta(example, tokenizer)
        elif isinstance(tokenizer, BertTokenizer):
            return self.tokenization_on_example_bert(example, tokenizer)
        else:
            raise NotImplementedError

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x200e:
            return True
        cat = unicodedata.category(c)
        if cat == "Zs":
            return True
        return False

    def bert_from_string_to_subtokens(self, text, tokenizer):
        '''
        Given a text, using berttokenizer to get the token, subtoken, new_text converted from tokens and the index map from subtokens to new_text  char, and the index map from new text to subtokens.
        :param text:
        :return:
        '''
        cleaned_text = ""
        for c in text:
            if self.is_whitespace(c):
                cleaned_text += " "
            else:
                cleaned_text += c

        unpunc_tokens = []
        subtokens = []
        char_to_unpunc_map = []
        unpunc_to_char_map = []
        unpunc_to_sub_map = []
        char_to_sub_map = []
        sub_to_char_map = []
        prev_is_whitespace = True

        for i, c in enumerate(cleaned_text):
            if c == " ":
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    unpunc_tokens.append(c)
                    unpunc_to_char_map.append(i)
                else:
                    unpunc_tokens[-1] += c
                prev_is_whitespace = False

            char_to_unpunc_map.append(len(unpunc_tokens) - 1)

        for unpunc_index, unpunc_tokenized_token in enumerate(unpunc_tokens):
            tokens = tokenizer.basic_tokenizer.tokenize(unpunc_tokenized_token)  # do punctuation tokenization
            unpunc_to_sub_map.append(len(subtokens))
            char_end_index = unpunc_to_char_map[unpunc_index + 1] if unpunc_index < len(
                unpunc_to_char_map) - 1 else len(char_to_unpunc_map)
            char_start_index = unpunc_to_char_map[unpunc_index]
            char_to_sub_map.extend([len(subtokens)] * (char_end_index - char_start_index))

            for token in tokens:
                for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                    subtokens.append(sub_token)
                    sub_to_char_map.append(unpunc_to_char_map[unpunc_index])

        return cleaned_text, subtokens, char_to_sub_map, sub_to_char_map

    def roberta_from_string_to_subtokens(self, subtokens, tokenizer):
        ori_to_sub_map = []
        sub_to_ori_map = []
        cur_len = 0
        text_from_sub = ""
        to_check = False
        for i, token in enumerate(subtokens):
            if token == "âĢ":
                to_check = True
                sub_to_ori_map.append(cur_len)
                continue
            if to_check and token == 'Ļ':
                tmp = "'"
                string_from_token = tmp
            else:
                string_from_token = tokenizer.convert_tokens_to_string(token)

            text_from_sub += string_from_token
            cur_tok_len = len(string_from_token)
            ori_to_sub_map.extend([i] * cur_tok_len)
            sub_to_ori_map.append(cur_len)
            cur_len += cur_tok_len
            to_check = False
        return text_from_sub, ori_to_sub_map, sub_to_ori_map

    def tokenization_on_example_bert(self, example, tokenizer):
        # do tokenization on raw question text
        query_text, query_subtokens, query_char_to_sub_map, query_sub_to_char_map = self.bert_from_string_to_subtokens(
            example.question_text, tokenizer)
        doc_text, doc_subtokens, doc_char_to_sub_map, doc_sub_to_char_map = self.bert_from_string_to_subtokens(
            example.doc_text, tokenizer)

        # generate token-level document entity index
        document_entities = []
        for entity in example.passage_entities:
            entity_start_position = doc_char_to_sub_map[entity['start_position']]
            cur_end_position = entity['end_position']
            entity_end_position = doc_char_to_sub_map[cur_end_position]

            stop_count = 0
            while True:
                if stop_count > 100:
                    logger.warning(
                        "Somethging wrong in finding the entity span in document span of {}!".format(example.qas_id))
                    break
                stop_count += 1
                cur_end_position += 1
                if cur_end_position >= len(doc_char_to_sub_map):
                    entity_end_position = len(doc_subtokens) - 1
                    break
                if doc_char_to_sub_map[cur_end_position] != entity_end_position:
                    entity_end_position = doc_char_to_sub_map[cur_end_position] - 1
                    break

            entity_start_position, entity_end_position = _improve_answer_span(doc_subtokens, entity_start_position,
                                                                              entity_end_position, tokenizer,
                                                                              entity['orig_text'])
            document_entities.append(
                (entity['orig_text'], entity_start_position, entity_end_position))  # ('Trump', 10, 10)
        answer_entities = []
        for entity in example.answer_entities:
            entity_start_position = doc_char_to_sub_map[entity['start_position']]
            cur_end_position = entity['end_position']
            entity_end_position = doc_char_to_sub_map[cur_end_position]

            stop_count = 0
            while True:
                if stop_count > 100:
                    logger.warning(
                        "Somethging wrong in finding the entity span in answer span of {}!".format(example.qas_id))
                    break
                stop_count += 1
                cur_end_position += 1
                if cur_end_position >= len(doc_char_to_sub_map):
                    entity_end_position = len(doc_subtokens) - 1
                    break
                if doc_char_to_sub_map[cur_end_position] != entity_end_position:
                    entity_end_position = doc_char_to_sub_map[cur_end_position] - 1
                    break

            entity_start_position, entity_end_position = _improve_answer_span(doc_subtokens, entity_start_position,
                                                                              entity_end_position, tokenizer,
                                                                              entity['orig_text'], is_answer=True)

            answer_entities.append(
                (entity['orig_text'], entity_start_position, entity_end_position))  # ('Trump', 10, 10)
        

        ## select entity strings
        entity_strings = set()
        query_entity_str_list = example.question_entities_strset
        for item in query_entity_str_list:
            clean_text = ""
            for c in item:
                if is_whitespace(c):
                    clean_text += " "
                else:
                    clean_text += remove_special_punc(c)
            entity_strings.add(clean_text)

        for document_entity in document_entities:
            entity_strings.add(document_entity[0])


        # match query to passage entities
        query_entities = match_query_entities(query_text, document_entities, query_char_to_sub_map, query_subtokens,
                                              tokenizer, entity_strings)  # [('trump', 10, 10)]
        tokenized_example = SquadExampleTokenized(id=example.qas_id,
                                                   query_text=query_text,
                                                   query_tokens=query_subtokens,
                                                   query_ori_to_tok_map=query_char_to_sub_map,
                                                   query_tok_to_ori_map=query_sub_to_char_map,
                                                   query_entities=query_entities,
                                                   doc_text=doc_text,
                                                   doc_ori_text=example.doc_text,
                                                   doc_tokens=doc_subtokens,
                                                   doc_tok_to_ori_map=doc_sub_to_char_map,
                                                   doc_ori_to_tok_map=doc_char_to_sub_map,
                                                   doc_entities=document_entities,
                                                   answer_entities=answer_entities)

        return tokenized_example

    def tokenization_on_example_roberta(self, example, tokenizer):
        # build the mapping relation among the subtoken index and token index and char index

        # for query subtokens.
        query_subtokens = tokenizer.tokenize(example.question_text)
        query_text_from_sub, query_ori_to_sub_map, query_sub_to_ori_map = self.roberta_from_string_to_subtokens(
            query_subtokens, tokenizer)

        # for doc subtokens
        doc_subtokens = tokenizer.tokenize(example.doc_text)
        doc_text, doc_ori_to_sub_map, doc_sub_to_ori_map = self.roberta_from_string_to_subtokens(doc_subtokens,
                                                                                                 tokenizer)

        if len(doc_text) != len(example.doc_text):
            logger.warning("{} has inconsistent lengths of documents after tokenization.".format(example.qas_id))
            passage_entities = relocate_entities(example.passage_entities, doc_text)
            example.passage_entities = passage_entities

        # return None

        document_entities = []
        for entity in example.passage_entities:
            if len(entity['orig_text']) < 2:
                continue
            entity_start_position = doc_ori_to_sub_map[entity['start_position']]
            entity_end_position = doc_ori_to_sub_map[entity['end_position']]
            document_entities.append(
                (entity['orig_text'], entity_start_position, entity_end_position))  # ('Trump', 10, 10)

        answer_entities = []
        for entity in example.answer_entities:
            entity_start_position = doc_ori_to_sub_map[entity['start_position']]
            entity_end_position = doc_ori_to_sub_map[entity['end_position']]
            answer_entities.append(
                (entity['orig_text'], entity_start_position, entity_end_position))  # ('Trump', 10, 10)

        ## select entity strings
        entity_strings = set()
        query_entity_str_list = example.question_entities_strset
        for item in query_entity_str_list:
            clean_text = ""
            for c in item:
                if is_whitespace(c):
                    clean_text += " "
                else:
                    clean_text += remove_special_punc(c)
            entity_strings.add(clean_text)

        for document_entity in document_entities:
            entity_strings.add(document_entity[0])

            # match query to passage entities
        query_entities = match_query_entities(query_text_from_sub, document_entities,
                                              query_ori_to_sub_map, query_subtokens, tokenizer, entity_strings)  # [('trump', 10, 10)]

        tokenized_example = SquadExampleTokenized(id=example.qas_id,
                                                   query_text=query_text_from_sub,
                                                   query_tokens=query_subtokens,
                                                   query_ori_to_tok_map=query_ori_to_sub_map,
                                                   query_tok_to_ori_map=query_sub_to_ori_map,
                                                   query_entities=query_entities,
                                                   doc_text=doc_text,
                                                   doc_ori_text=example.doc_text,
                                                   doc_tokens=doc_subtokens,
                                                   doc_tok_to_ori_map=doc_sub_to_ori_map,
                                                   doc_ori_to_tok_map=doc_ori_to_sub_map,
                                                   doc_entities=document_entities,
                                                   answer_entities=answer_entities)

        return tokenized_example

    def tokenization_on_examples(self, examples, tokenizer):
        tokenization_result = []
        for example in tqdm(examples, desc='Tokenization on examples'):
            tokenization_result.append(self.tokenization_on_example(example, tokenizer))
        return tokenization_result

    def _create_examples(self, input_file, set_type):
        with open(input_file, "r") as reader:
            logger.info("Reading examples from {}".format(input_file))
            input_data = json.load(reader)["data"]
        if self.fewer_label:
            logger.info("using fewer label, label rate:{}".format(self.label_rate))
            input_data = input_data[:int(len(input_data) * self.label_rate)]

        # def is_whitespace(c):
        #     if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x200e:
        #         return True
        #     cat = unicodedata.category(c)
        #     if cat == "Zs":
        #         return True
        #     return False
        # 
        # def remove_special_punc(c):
        #     if c == "‘":
        #         return "\'"
        #     if c == "’":
        #         return "\'"
        #     if c == "“":
        #         return "\""
        #     if c == "”":
        #         return "\""
        #     return c

        examples = []
        for entry in tqdm(input_data, desc='Reading entries from json'):
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"].strip()

                doc_text = ""
                for c in paragraph_text:
                    if is_whitespace(c):
                        doc_text += " "
                    else:
                        doc_text += c

                # doc_text = paragraph_text
                # load entities in passage
                passage_entities = []
                for entity in paragraph['context_entities']:
                    entity_start_offset = entity['start']
                    entity_end_offset = entity['end']
                    og_entity_text = doc_text[entity_start_offset: entity_end_offset + 1]

                    # entity_text = entity['text']

                    entity_text = ""

                    for c in entity['text']:
                        if is_whitespace(c):
                            entity_text += " "
                        else:
                            entity_text += c

                    # for c in entity['text']:
                    #     if is_whitespace(c):
                    #         entity_text += " "
                    #     else:
                    #         entity_text += remove_special_punc(c)

                    if entity_text != og_entity_text:
                        entity_start_offset, entity_end_offset, entity_text = renew_offset(entity_start_offset, entity_end_offset, entity_text, doc_text)
                        if entity_end_offset is None:
                            logger.warning("doc text: {} differs from the positions {} in doc_text".format(entity_text,
                                                                                                              og_entity_text))
                            continue
                        # assert entity_text == entity['text']
                    passage_entities.append({'orig_text': entity_text,
                                             'start_position': entity_start_offset,
                                             'end_position': entity_end_offset})



                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    # question_text = qa["question"]
                    question_text = ""

                    for c in qa["question"].strip():
                        if is_whitespace(c):
                            question_text += " "
                        else:
                            question_text += c

                    question_entities_strset = set([entity_info["text"] for entity_info in qa["question_entities"]])
                    answers = qa["answers"]
                    answer_entities = []

                    if (len(qa["answers"]) == 0):
                        logger.warning("paragraph: {} \n q: {}".format(paragraph, qa))
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )

                    for entity in answers:
                        orig_answer_text = entity["text"]
                        answer_length = len(orig_answer_text)
                        entity_start_offset = entity['answer_start']
                        entity_end_offset =  entity_start_offset + answer_length - 1
                        if entity_end_offset < entity_start_offset:  # some error labeled entities in squad dataset
                            continue

                        # entity_text = entity['text']
                        entity_text = ""
                        for c in entity['text']:
                            if is_whitespace(c):
                                entity_text += " "
                            else:
                                entity_text += c

                        entity_doc_text = doc_text[entity_start_offset: entity_end_offset + 1]
                        if entity_doc_text != entity_text:
                            entity_start_offset, entity_end_offset, entity_text = renew_offset(entity_start_offset,
                                                                                               entity_end_offset,
                                                                                               entity_text,
                                                                                               doc_text)
                            logger.info("renew: from {} to {}".format(entity_doc_text, entity_text))

                            if entity_end_offset is None:
                                logger.warning("answer text: {} differs from the positions {} in doc_text".format(entity_text,
                                                                                                              entity_doc_text))
                                continue

                        answer_entities.append({'orig_text': entity_text,
                                                'start_position': entity_start_offset,
                                                'end_position': entity_end_offset})


                    # orig_answer_text = entity["text"]
                    # answer_length = len(orig_answer_text)
                    # entity_start_offset = entity['answer_start']
                    # entity_end_offset =  entity_start_offset + answer_length - 1
                    # if entity_end_offset < entity_start_offset:  # some error labeled entities in squad dataset
                    #     continue

                    # entity_text = ""
                    # for c in entity['text']:
                    #     if is_whitespace(c):
                    #         entity_text += " "
                    #     else:
                    #         entity_text += remove_special_punc(c)
                    #
                    # entity_doc_text = doc_text[entity_start_offset: entity_end_offset + 1]
                    # if entity_doc_text != entity_text:
                    #     entity_start_offset, entity_end_offset, entity_text = renew_offset(entity_start_offset,
                    #                                                                        entity_end_offset,
                    #                                                                        entity_text,
                    #                                                                        doc_text)
                    #     logger.info("renew: {}".format(entity_text))
                    #
                    #     if entity_end_offset is None:
                    #         logger.warning("answer text: {} differs from the positions {} in doc_text".format(entity_text,
                    #                                                                                       entity_doc_text))
                    #         continue

                    # answer_entities.append({'orig_text': orig_answer_text,
                    #                         'start_position': entity_start_offset,
                    #                         'end_position': entity_end_offset})

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        question_entities_strset=question_entities_strset,
                        doc_text=doc_text,
                        answer_entities=answer_entities,
                        passage_entities=passage_entities,
                    )
                    examples.append(example)
        # global remove_special_punc_count
        # logger.info(remove_special_punc_count)
        return examples

    @classmethod
    def compute_predictions_logits(
            cls,
            all_examples,
            all_features,
            all_results,
            n_best_size,
            max_answer_length,
            output_prediction_file,
            output_nbest_file,
            verbose_logging,
            predict_file,
            tokenizer
    ):
        """read in the dev.json file"""
        logger.info(" Find {}-best prediction for each qas-id.".format(n_best_size))
        with open(predict_file, "r", encoding='utf-8') as reader:
            predict_json = json.load(reader)["data"]
            # all_candidates = {}
            # for passage in predict_json:
            #     passage_text = passage['passage']['text']
            #     candidates = []
            #     for entity_info in passage['passage']['entities']:
            #         start_offset = entity_info['start']
            #         end_offset = entity_info['end']
            #         candidates.append(passage_text[start_offset: end_offset + 1])
            #     for qa in passage['qas']:
            #         all_candidates[qa['id']] = candidates

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        example_index = -1
        for example in tqdm(all_examples, desc='Compute best predictions'):
            example_index += 1
            features = example_index_to_features[example_index]

            prelim_predictions = []
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                cur_predictions = find_nbest_start_end_logit(result, n_best_size, max_answer_length, feature,
                                                             feature_index)

                prelim_predictions.extend(cur_predictions)

            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            nbest = find_nbest_prediction_for_example(prelim_predictions, n_best_size, example, features, tokenizer,
                                                      verbose_logging=verbose_logging)
            assert len(nbest) >= 1
            total_scores = []
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output['start_index'] = entry.start_index
                output['end_index'] = entry.end_index
                nbest_json.append(output)

            # best_pred = None
            # # for pred in nbest_json:
            # #     if any([exact_match_score(pred['text'], candidate) > 0. for candidate in all_candidates[example.id]]):
            # #         best_pred = pred
            # #         break
            # # if best_pred is None:
            # #     for pred in nbest_json:
            # #         if any([f1_score(pred['text'], candidate) > 0. for candidate in
            # #                 all_candidates[example.id]]):
            # #             best_pred = pred
            # #             break
            # if best_pred is None:
            #     best_pred = nbest_json[0]
            best_pred = nbest_json[0]

            all_predictions[example.id] = (best_pred['text'], best_pred['start_index'], best_pred['end_index'])
            all_nbest_json[example.id] = nbest_json

        """Write final predictions to the json file and log-odds of null if needed."""
        if output_prediction_file:
            logger.info(f"Writing predictions to: {output_prediction_file}")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

        # if output_nbest_file:
        #     logger.info(f"Writing nbest to: {output_nbest_file}")
        #     with open(output_nbest_file, "w") as writer:
        #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        return all_predictions

    @classmethod
    def squad_evaluate(cls, examples, predictions, relate_path="../data/"):
        evaluates = dict()

        f1 = exact_match = total = 0
        total = len(examples)
        correct_ids = []
        for example in examples:
            if example.id not in predictions:
                logger.warning('Unanswered question {} will receive score 0.'.format(example.id))
                continue

            ground_truths = list(map(lambda x: x[0], example.answer_entities))
            prediction_item = predictions[example.id]
            _exact_match = metric_max_over_ground_truths(exact_match_score, prediction_item[0], ground_truths)
            if int(_exact_match) == 1:
                correct_ids.append(example.id)
            exact_match += _exact_match

            f1 += metric_max_over_ground_truths(f1_score, prediction_item[0], ground_truths)
            evaluates[example.id] = {"predict": prediction_item, "true": example.answer_entities}

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        logger.info('* Exact_match: {}\n* F1: {}'.format(exact_match, f1))

        evaluate_file = os.path.join(relate_path, "evaluate.json")
        logger.info(f"Writing predictions to: {evaluate_file}")
        with open(evaluate_file, "w") as writer:
            writer.write(json.dumps(evaluates, indent=4) + "\n")

        return {'exact_match': exact_match, 'f1': f1}  # , correct_ids


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text, is_answer=False):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The squad annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in squad, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    if is_answer:
        logger.warning("no exact match answer entities")
    return (input_start, input_end)


class SquadFeature(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 qas_id,
                 example_index,
                 # doc_span_index,
                 tokens,
                 tokid_span_to_orig_map,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 kgs_concept_ids,
                 kgs_conceptids2synset,
                 kgs_definition_ids=None,
                 kgs_graphs=None,
                 start_position=None,
                 end_position=None,
                 ):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.example_index = example_index
        self.tokens = tokens
        self.tokid_span_to_orig_map = tokid_span_to_orig_map
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position
        self.kgs_concept_ids = kgs_concept_ids
        self.kgs_definition_ids = kgs_definition_ids
        self.kgs_graphs = kgs_graphs
        self.kgs_conceptids2synset = kgs_conceptids2synset


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index


def _is_real_subspan(start, end, other_start, other_end):
    return (start >= other_start and end < other_end) or (start > other_start and end <= other_end)


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_answer_position(doc_span, answer_spans):
    for answer_span in answer_spans:
        if answer_span[0] >= doc_span[0] and answer_span[1] <= doc_span[0] + doc_span[1]:
            return answer_span[0], answer_span[1]
    return -1, -1


def get_doc_spans(all_doc_tokens_len, max_tokens_for_doc, doc_stride):
    doc_spans = []
    start_offset = 0
    while start_offset < all_doc_tokens_len:
        length = all_doc_tokens_len - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == all_doc_tokens_len:
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def build_first_part_features(tokenizer_type, use_kg, query_tokens, query_kgs_concepts):
    tokens = []
    segment_ids = []
    kgs_concept_ids = dict()
    for kg in use_kg:
        kgs_concept_ids[kg] = []

    # load the query part.
    if tokenizer_type == "roberta":
        tokens.append("<s>")
    else:
        tokens.append("[CLS]")
    segment_ids.append(0)
    for kg in use_kg:
        kgs_concept_ids[kg].append([])
        for query_concept in query_kgs_concepts[kg]:
            kgs_concept_ids[kg].append(query_concept)

    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)

    if tokenizer_type == "roberta":
        tokens.extend(["</s>", "</s>"])
        segment_ids.extend([0, 1])
        for kg in use_kg:
            kgs_concept_ids[kg].extend([[], []])
    else:
        tokens.append("[SEP]")
        segment_ids.append(0)
        for kg in use_kg:
            kgs_concept_ids[kg].append([])

    return tokens, segment_ids, kgs_concept_ids


def match_query_entities(text, document_entities, ori_to_tok_map, subtokens, tokenizer, entity_strings):
    """
    Find the index of entities in the query.
    :param text:
    :param document_entities:
    :param ori_to_sub_map:
    :param if_roberta:
    :return:
    """

    # entity_strings = set()
    # for document_entity in document_entities:
    #     entity_strings.add(document_entity[0])  # .lower())

    # text_lower = text.lower()

    # do matching
    results = []
    for entity_string in entity_strings:
        start = 0
        while True:
            # pos = text_lower.find(entity_string, start)
            pos = text.find(entity_string, start)
            if pos == -1:
                break
            token_start = ori_to_tok_map[pos]
            cur_end_position = pos + len(entity_string) - 1
            token_end = ori_to_tok_map[cur_end_position]

            stop_count = 0
            while True:
                if stop_count > 100:
                    logger.warning("Somethging wrong in finding the entity span in matching query span")
                    break
                stop_count += 1
                cur_end_position += 1
                if cur_end_position >= len(ori_to_tok_map):
                    token_end = len(subtokens) - 1
                    break
                if ori_to_tok_map[cur_end_position] != token_end:
                    token_end = ori_to_tok_map[cur_end_position] - 1
                    break

            # assure the match is not partial match (eg. "ville" matches to "danville")
            token_start, token_end = _improve_answer_span(subtokens, token_start, token_end, tokenizer, entity_string)

            results.append((entity_string, token_start, token_end))
            start = cur_end_position

    # filter out a result span if it's a subspan of another span
    no_subspan_results = []
    for result in results:
        if not any(
                [_is_real_subspan(result[1], result[2], other_result[1], other_result[2]) for other_result in results]):
            no_subspan_results.append((result[0], result[1], result[2]))
    if len(no_subspan_results) != len(set(no_subspan_results)):
        logger.warning("Query subspan results is wrong! {}".format(no_subspan_results))
        for e_str in entity_strings:
            logger.warning("{}".format(e_str))

    return no_subspan_results


def create_tensordataset(features, is_training, args, retrievers, tokenizer, relation_list, encoder, definition_info,
                         tqdm_enabled=True, debug=False):
    wn18_dir = args.wn18_dir
    wn18_path = os.path.join(wn18_dir, "full.txt")
    if not os.path.exists(wn18_path):
        wn18_full = open(wn18_path, 'a')

        wn18_train = open(os.path.join(wn18_dir, "train.txt"), 'r')
        for line in wn18_train.readlines():
            wn18_full.writelines(line, )
        wn18_train.close()

        wn18_valid = open(os.path.join(wn18_dir, "valid.txt"), 'r')
        for line in wn18_valid.readlines():
            wn18_full.writelines(line, )
        wn18_valid.close()

        wn18_test = open(os.path.join(wn18_dir, "test.txt"), 'r')
        for line in wn18_test.readlines():
            wn18_full.writelines(line, )
        wn18_test.close()

        wn18_full.close()

    # wn_18 = open(wn18_path, 'r')

    # construct graphs
    # all_kgs_graphs = []
    kg_path = os.path.join(args.data_dir, args.kg_paths["wordnet"])
    concept_embedding_path = os.path.join(kg_path, "wn_concept2vec.txt")
    id2concept, concept2id, concept_embedding_mat = read_concept_embedding(concept_embedding_path)

    offset_to_wn18name_dict = {}
    fin = open(os.path.join("./data/kgs/",
                            'wordnet-mlj12-definitions.txt'))
    for line in fin:
        info = line.strip().split('\t')
        offset_str, synset_name = info[0], info[1]
        offset_to_wn18name_dict[offset_str] = synset_name
    fin.close()

    defid2def = definition_info.defid2def
    conceptid2defid = definition_info.conceptid2defid

    if args.model_type == "kelm":
        # get mapping relation between src to dst list each relation
        multi_relation_dict = retrive_multi_relation_dict(relation_list, wn18_dir)
        all_kgs_graphs = []
        logger.info("testing graph_collecter function")
        for f in tqdm(features, desc="build dgl graph", disable=not tqdm_enabled, ):
            g = graph_collecter(f, wn18_dir, offset_to_wn18name_dict, concept2id, relation_list, tokenizer,
                                multi_relation_dict, retrievers["wordnet"], encoder, defid2def, conceptid2defid,
                                )
            all_kgs_graphs.append(g)

    else:
        all_kgs_graphs = None

    assert len(conceptid2defid) == len(defid2def)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_kgs_concept_tuple = tuple()
    for kg in args.use_kgs:
        all_concept_ids = torch.tensor([f.kgs_concept_ids[kg] for f in features], dtype=torch.long)
        # all_definition_ids = torch.tensor([f.kgs_definition_ids[kg] for f in features], dtype=torch.long)
        # all_kgs_concept_tuple = all_kgs_concept_tuple+(all_concept_ids, all_definition_ids, )
        all_kgs_concept_tuple = all_kgs_concept_tuple + (all_concept_ids,)

    if not is_training:
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index,
                                *all_kgs_concept_tuple)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_feature_index,
            all_start_positions,
            all_end_positions,
            *all_kgs_concept_tuple,
        )

    return dataset, all_kgs_graphs


def retrive_multi_relation_dict(relation_list, wn18_dir):
    multi_relation_dict_path = os.path.join(wn18_dir, "multi_relation_dict.npy")
    multi_relation_dict = {}

    if os.path.exists(multi_relation_dict_path):
        multi_relation_dict = np.load('my_file.npy', allow_pickle='TRUE').item()
        return multi_relation_dict

    for relation_type in relation_list:
        multi_relation_dict[relation_type] = {}
        realation_path = os.path.join(wn18_dir, relation_type + ".txt")

        if not os.path.exists(realation_path):
            wn18_full = open(os.path.join(wn18_dir, "full.txt"), 'r')
            wn18_relation = open(realation_path, 'w')

            for line in wn18_full.readlines():
                relation = line.strip().split("\t")[1]
                if relation != relation_type:
                    continue
                wn18_relation.write(line)

            wn18_full.close()
            wn18_relation.close()

        wn18_relation = open(realation_path, 'r')
        for line in wn18_relation.readlines():
            src, relation, dst = line.strip().split("\t")
            if src not in multi_relation_dict[relation_type]:
                multi_relation_dict[relation_type][src] = []
            try:
                multi_relation_dict[relation_type][src].append(wn.synset(dst))
            except:
                logger.warning("{} can't switch to synset object".format(dst))
        wn18_relation.close()

    # np.save(multi_relation_dict_path, multi_relation_dict)
    return multi_relation_dict


def graph_collecter(f, wn18_dir, offset_to_wn18name_dict, concept2id, relation_list, tokenizer, multi_relation_dict,
                    retrievers, encoder, defid2def, conceptid2defid):
    nell_src = []
    nell_dst = []
    nell_conceptid2nodeid = {}
    nell_nodeid2conceptid = []

    builder_list = [RelationalGraphBuilder(f, relation_type, wn18_dir, offset_to_wn18name_dict, concept2id,
                                           multi_relation_dict[relation_type], retrievers) for relation_type in
                    relation_list]

    # wn_conceptids2synset = f.kgs_conceptids2synset["wordnet"]
    wn_concept_input = f.kgs_concept_ids["wordnet"]
    nell_concept_input = f.kgs_concept_ids["nell"]
    assert (len(f.input_ids) == len(wn_concept_input) == len(nell_concept_input))

    for token_id, concept_list in enumerate(wn_concept_input):
        if concept_list[0] == 0:
            continue
        for i, concept_id in enumerate(concept_list):
            if concept_id == 0:
                break
            RelationalGraphBuilder.update_wn_src(token_id)
            if RelationalGraphBuilder.is_update_relation_graph(concept_id):
                RelationalGraphBuilder.update_wn_conceptid2nodeid(concept_id)
                RelationalGraphBuilder.update_wn_nodeid2conceptid(concept_id)

                for builder in builder_list:
                    builder.update_dst_id_list(concept_id)

            RelationalGraphBuilder.update_wn_dst(concept_id)
    assert ([i for i in
             RelationalGraphBuilder.wn_conceptid2nodeid.keys()] == RelationalGraphBuilder.wn_nodeid2conceptid)
    # assert ([i for i in wn_hypernyms_conceptid2nodeid.keys()] == wn_hypernyms_nodeid2conceptid)
    # assert ([i for i in wn_hyponyms_conceptid2nodeid.keys()] == wn_hyponyms_nodeid2conceptid)
    #
    for token_id, concept_list in enumerate(nell_concept_input):
        if concept_list[0] == 0:
            continue
        for i, concept_id in enumerate(concept_list):
            if concept_id == 0:
                break
            # if type(token_id) is not numpy.int64:
            #     print(type(token_id))
            #     print(token_id)
            nell_src.append(token_id)
            if concept_id not in nell_conceptid2nodeid:
                nell_conceptid2nodeid[concept_id] = len(nell_conceptid2nodeid)
                # nell_nodeid2conceptid[len(conceptid2nodeid)-1] = concept_id
                nell_nodeid2conceptid.append(concept_id)
            # if type(nell_conceptid2nodeid[concept_id]) is not numpy.int64:
            #     print(type(nell_conceptid2nodeid[concept_id]))
            #     print(nell_conceptid2nodeid[concept_id])
            nell_dst.append(nell_conceptid2nodeid[concept_id])
    assert ([i for i in nell_conceptid2nodeid.keys()] == nell_nodeid2conceptid)

    data_dict = {
        ('wn_concept_id', 'synset_', 'token_id'): (
            torch.tensor(RelationalGraphBuilder.wn_dst, dtype=torch.long),
            torch.tensor(RelationalGraphBuilder.wn_src, dtype=torch.long)),
        ('nell_concept_id', 'belong_', 'token_id'): (
            torch.tensor(nell_dst, dtype=torch.long), torch.tensor(nell_src, dtype=torch.long)),
    }

    num_nodes_dict = {
        "token_id": np.sum(f.attention_mask),
        "wn_concept_id": len(RelationalGraphBuilder.wn_nodeid2conceptid),
        "nell_concept_id": len(nell_nodeid2conceptid),
    }

    id_type_list = []

    for i, relation_type in enumerate(relation_list):
        id_type = "wn{}_id".format(relation_type)
        id_type_list.append(id_type)

        data_dict[("wn_concept_id", relation_type, id_type)] = \
            (torch.tensor(builder_list[i].wn_relation_src, dtype=torch.long),
             torch.tensor(builder_list[i].wn_relation_dst, dtype=torch.long))

        num_nodes_dict[id_type] = len(builder_list[i].wn_relation_nodeid2conceptid)
    # dgl.graph()
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    # assign concept id
    g.nodes['wn_concept_id'].data["conceptid"] = torch.tensor(RelationalGraphBuilder.wn_nodeid2conceptid,
                                                              dtype=torch.long)
    g.nodes['nell_concept_id'].data["conceptid"] = torch.tensor(nell_nodeid2conceptid, dtype=torch.long)

    for i, id_type in enumerate(id_type_list):
        g.nodes[id_type].data["conceptid"] = torch.tensor(builder_list[i].wn_relation_nodeid2conceptid,
                                                          dtype=torch.long)

    # assign bert tokenize id
    wn_defid_list = get_definition_ids(RelationalGraphBuilder.wn_nodeid2conceptid,
                                       RelationalGraphBuilder.wn_conceptids2synset, tokenizer, encoder,
                                       defid2def, conceptid2defid)
    g.nodes['wn_concept_id'].data["definition_id"] = torch.tensor(wn_defid_list)

    for i, id_type in enumerate(id_type_list):
        wn_relation_nodeid2conceptid = builder_list[i].wn_relation_nodeid2conceptid
        if not wn_relation_nodeid2conceptid:
            g.nodes[id_type].data["definition_embedding"] = torch.tensor([])
            g.nodes[id_type].data["definition_id"] = torch.tensor([])
            # logger.info("{} is empty".format(id_type))
        else:
            relation_defid_list = get_definition_ids(wn_relation_nodeid2conceptid,
                                                     RelationalGraphBuilder.wn_conceptids2synset, tokenizer,
                                                     encoder, defid2def, conceptid2defid)
            g.nodes[id_type].data["definition_id"] = torch.tensor(relation_defid_list)
    return g


def get_definition_ids(nodeid2conceptid, conceptids2synset, tokenizer, encoder, defid2def, conceptid2defid):
    defid_list = []
    # embedding_list = []
    # kg_definition_list = []
    for concept_id in nodeid2conceptid:
        try:
            if concept_id in conceptid2defid:
                defid_list.append(conceptid2defid[concept_id])
                continue
            defid_list.append(len(conceptid2defid))
            conceptid2defid[concept_id] = len(conceptid2defid)
            def_sentence = wn.synset(conceptids2synset[concept_id]).definition()
            defid2def.append(def_sentence)
        except:
            logger.warning("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! wronng concept synset")
            exit()

    return defid_list

def get_tokenized_definition(nodeid2conceptid, conceptids2synset, tokenizer):
    kg_definition_list = []
    for concept_id in nodeid2conceptid:
        try:
            kg_definition_list.append(wn.synset(conceptids2synset[concept_id]).definition())
        except:
            logger.warning("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! wronng concept synset")
            kg_definition_list.append("[UNK]")

    tokenized_definition = tokenizer(kg_definition_list, padding=True)

    return tokenized_definition


def create_input(args, batch, global_step, batch_synset_graphs=None, wn_synset_graphs=None, evaluate=False):
    if evaluate:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "global_step": global_step,
            "batch_synset_graphs": batch_synset_graphs,
            "wn_synset_graphs": wn_synset_graphs,
        }

        for i, kg in enumerate(args.use_kgs):
            name = kg + "_concept_ids"
            inputs[name] = batch[4 + i]

    else:
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[4],
            "end_positions": batch[5],
            "global_step": global_step,
            "batch_synset_graphs": batch_synset_graphs,
            "wn_synset_graphs": wn_synset_graphs,
        }

        for i, kg in enumerate(args.use_kgs):
            name = kg + "_concept_ids"
            inputs[name] = batch[6 + i]

    if args.model_type == "roberta" or args.model_type == "roberta_base":
        del inputs["token_type_ids"]
        del inputs["global_step"]
        del inputs["batch_synset_graphs"]
        del inputs["wn_synset_graphs"]
        for i, kg in enumerate(args.use_kgs):
            del inputs[kg + "_concept_ids"]

    if args.text_embed_model == "roberta" and args.model_type != "roberta":
        del inputs["token_type_ids"]

    if args.text_embed_model == "roberta_base" and args.model_type != "roberta_base":
        del inputs["token_type_ids"]

    if args.model_type == "bert":
        del inputs["global_step"]
        del inputs["batch_synset_graphs"]
        del inputs["wn_synset_graphs"]
        for i, kg in enumerate(args.use_kgs):
            del inputs[kg + "_concept_ids"]

    return inputs


def get_final_text(pred_text, orig_text, tokenizer, verbose=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    if isinstance(tokenizer, BertTokenizer):
        tok_text = " ".join(tokenizer.basic_tokenizer.tokenize(orig_text))
    else:
        tok_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose:
            logger.info("Couldn't map start position")
        return orig_text
        # return pred_text.strip()

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def find_nbest_start_end_logit(result, n_best_size, max_answer_length, feature, feature_index, ):
    start_indexes = _get_best_indexes(result.start_logits, n_best_size)
    end_indexes = _get_best_indexes(result.end_logits, n_best_size)
    prelim_predictions = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if end_index < start_index:
                continue
            if start_index >= min(len(feature.tokens),
                                  feature.tokid_span_to_orig_map[0] + feature.tokid_span_to_orig_map[2]):
                continue
            if end_index >= min(len(feature.tokens),
                                feature.tokid_span_to_orig_map[0] + feature.tokid_span_to_orig_map[2]):
                continue
            if start_index < feature.tokid_span_to_orig_map[0]:
                continue

            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
    return prelim_predictions


def find_nbest_prediction_for_example(prelim_predictions, n_best_size, example, features, tokenizer,
                                      verbose_logging=False):
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        pred_start_index = -1
        pred_end_index = -1
        if pred.start_index > -1:  # this is a non-null prediction
            # ori_tok_start = feature.tokid_span_to_orig_map[pred.start_index]
            # ori_tok_end = feature.tokid_span_to_orig_map[pred.end_index]

            pred_start_index = pred.start_index - feature.tokid_span_to_orig_map[0] + feature.tokid_span_to_orig_map[1]
            pred_end_index = pred.end_index - feature.tokid_span_to_orig_map[0] + feature.tokid_span_to_orig_map[1]

            orig_doc_start = example.doc_tok_to_ori_map[pred_start_index]
            orig_doc_end = example.doc_tok_to_ori_map[pred_end_index]
            orig_doc_start, orig_doc_end = refine_char_start_end(orig_doc_start, orig_doc_end,
                                                                 example.doc_ori_to_tok_map)
            # try:
            orig_text = example.doc_text[orig_doc_start: (orig_doc_end + 1)].strip()
            # except:
            #     orig_text = example.doc_text[orig_doc_start: orig_doc_end]
            tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
            if isinstance(tokenizer, RobertaTokenizer):
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens).strip()
            elif isinstance(tokenizer, BertTokenizer):
                # tok_text = " ".join(tok_tokens)
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")
                # tok_text = tok_text.strip()
                # tok_text = " ".join(tok_text.split())

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())


            final_text = get_final_text(tok_text, orig_text, tokenizer, verbose_logging)
            if final_text in seen_predictions:
                continue
        else:
            # orig_text = ""
            final_text = ""
            seen_predictions[""] = True

        nbest.append(_NbestPrediction(
            text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=pred_start_index,
            end_index=pred_end_index))

    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=-1, end_index=-1))
    return nbest


def convert_example_to_features_init(tokenizer_for_convert, ):
    global tokenizer

    tokenizer = tokenizer_for_convert


def refine_char_start_end(orig_doc_start, orig_doc_end, doc_ori_to_tok_map):
    tok_start = doc_ori_to_tok_map[orig_doc_start]
    i = 1
    while orig_doc_start - i >= 0 and tok_start == doc_ori_to_tok_map[orig_doc_start - i]:
        i += 1
    new_start = orig_doc_start - i + 1

    tok_end = doc_ori_to_tok_map[orig_doc_end]
    i = 1
    while orig_doc_end + i < len(doc_ori_to_tok_map) and tok_end == doc_ori_to_tok_map[orig_doc_end + i]:
        i += 1

    new_end = orig_doc_end + i - 1
    return new_start, new_end


def relocate_entities(passage_entities, new_text):
    entity_strings = set()
    for entity in passage_entities:
        entity_strings.add(entity['orig_text'])
    new_entities = []
    for e_str in entity_strings:
        e_starts, e_ends = find_all(e_str, new_text)
        for start, end in zip(e_starts, e_ends):
            new_entities.append({'orig_text': e_str, 'start_position': start, 'end_position': end})

    return new_entities


def find_all(aim_str, text):
    start = 0
    end = len(text)
    aim_len = len(aim_str)
    start_positions = []
    end_positions = []
    while start < end:
        i = text.find(aim_str, start, end)
        if i == -1:
            break
        start = i + 1
        start_positions.append(i)
        end_positions.append(i + aim_len - 1)
    return start_positions, end_positions


def roberta_from_string_to_subtokens(subtokens, tokenizer):
    ori_to_sub_map = []
    sub_to_ori_map = []
    cur_len = 0
    text_from_sub = ""
    to_check = False
    for i, token in enumerate(subtokens):
        if token == "âĢ":
            to_check = True
            sub_to_ori_map.append(cur_len)
            continue
        if to_check and token == 'Ļ':
            tmp = "'"
            string_from_token = tmp
        else:
            string_from_token = tokenizer.convert_tokens_to_string(token)

        text_from_sub += string_from_token
        cur_tok_len = len(string_from_token)
        ori_to_sub_map.extend([i] * cur_tok_len)
        sub_to_ori_map.append(cur_len)
        cur_len += cur_tok_len
        to_check = False
    return text_from_sub, ori_to_sub_map, sub_to_ori_map

def check_concept_list(w):
    for i in range(len(w)):
        if not isinstance(w[i], list):
            print("wrong_type{}".format(i))
            print(w[i])
        for j in range(len(w[i])):
            if not isinstance(w[i][j], int):
                print("wrong_type{}   {}".format(i, j))
                print(w[i][j])


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 0x200e:
        return True
    cat = unicodedata.category(c)
    if cat == "Zs":
        return True
    return False

def remove_special_punc(c):
    # global remove_special_punc_count
    if c == "‘":
        # logger.warning("remove_special_punc is triggered")
        # remove_special_punc_count += 1
        return "\'"
    if c == "’":
        # logger.warning("remove_special_punc is triggered")
        # remove_special_punc_count += 1
        return "\'"
    if c == "“":
        # logger.warning("remove_special_punc is triggered")
        # remove_special_punc_count += 1
        return "\""
    if c == "”":
        # logger.warning("remove_special_punc is triggered")
        # remove_special_punc_count += 1
        return "\""
    return c

def renew_offset(og_start, og_end, og_text, all_text):
    for new_start in range(max(0, og_start-5), min(og_start+3, len(all_text))):
        for new_end in range(max(0, og_end - 5), min(og_end + 3, len(all_text))):
            if new_start > new_end:
                continue
            entity_text = all_text[new_start: new_end + 1]
            if og_text == entity_text:
                return new_start, new_end, entity_text
    return None, None, og_text