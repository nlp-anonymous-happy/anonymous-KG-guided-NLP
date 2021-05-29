# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright 2019 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script kgs_retrieve related NELL entities and their concepts for each named-entity in ReCoRD
# 1. transform ReCoRD entity from word sequences into strings (use _ to replace whitespace and eliminate punc)
# 2. preprocess NELL entity name (remove front 'n' for NELL entities when digit is in the beginning and additional _)
# 3. for ReCoRD entities with more than one token, use exact match
# 4. for one-word ReCoRD entities, do wordnet lemmatization before matching (and matching by both raw and morphed forms)
# 5. in a passage, if entity A is a suffix of entity B, use B's categories instead

import logging
import string
import os
from collections import namedtuple
from nltk.corpus import wordnet as wn

from kgs_retrieve.baseretriever import KGRetriever, read_concept_embedding
import networkx as nx


_TempRectuple = namedtuple('entity_record', [
            'entity_string', 'start', 'end', 'retrieved_concepts', 'retrieved_entities'])

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

puncs = set(string.punctuation)

class NellRetriever(KGRetriever):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.score_threshold = 0.9
        self.name = "nell"
        self.max_concept_length = 0

        # load kg embddings.
        logger.info("Loading KG embeddings for nell.")
        concept_embedding_path = os.path.join(filepath, "nell_concept2vec.txt")
        self.id2concept, self.concept2id, self.concept_embedding_mat = read_concept_embedding(
            concept_embedding_path)

        # load concept  sets with embeddings.
        self.concept_set = set()
        with open(filepath + 'nell_concept_list.txt') as fin:
            for line in fin:
                concept_name = line.strip()
                self.concept_set.add(concept_name)

        # load nell csv file and build NELL entity to category dict
        logger.info('Begin to load NELL csv...')
        fin = open(filepath + 'NELL.08m.1115.esv.csv')
        self.nell_ent_to_cpt = {}
        self.nell_ent_to_fullname = {}

        header = True
        for line in fin:
            if header:
                header = False
                continue
            line = line.strip()
            items = line.split('\t')
            if items[1] == 'generalizations' and float(items[4]) >= self.score_threshold:
                nell_ent_name = preprocess_nell_ent_name(items[0])
                category = items[2]
                if nell_ent_name not in self.nell_ent_to_cpt:
                    self.nell_ent_to_cpt[nell_ent_name] = set()
                    self.nell_ent_to_fullname[nell_ent_name] = set()
                self.nell_ent_to_cpt[nell_ent_name].add(category)
                self.nell_ent_to_fullname[nell_ent_name].add(items[0])
        logger.info('Finished reading NELL csv.')

    def find_entity_info(self, entities):
        info_dict = dict()
        for entity in entities:
            ent_name = entity[0].lower()
            if ent_name in info_dict:
                continue
            else:
                nell_name = "_".join(ent_name.split(" "))
                cpt, nell_ent = set(), set()
                if nell_name in self.nell_ent_to_cpt:
                    cpt.update(self.nell_ent_to_cpt[nell_name])
                    nell_ent.update(self.nell_ent_to_fullname[nell_name])
                if '_' not in nell_name:
                    for pos_tag in ['n', 'v', 'a', 'r']:
                        morph = wn.morphy(nell_name, pos_tag)
                        if morph is not None and morph in self.nell_ent_to_cpt:
                            cpt.update(self.nell_ent_to_cpt[morph])
                            nell_ent.update(self.nell_ent_to_fullname[morph])

                info_dict[ent_name] = {"nell_concept": cpt, "nell_entities": nell_ent}
        return info_dict


    def update_entity_info(self, old_info_dict):
        new_info_dict = dict()
        for trt, vals in old_info_dict.items():
            new_nell_cpt_set, new_nell_ent_set = set(), set()
            for other_trt, other_vals in old_info_dict.items():
                if other_trt != trt and other_trt.endswith(trt):
                    new_nell_cpt_set.update(other_vals['nell_concept'])
                    new_nell_ent_set.update(other_vals['nell_entities'])
            # no need to replace
            if len(new_nell_cpt_set) == 0:
                new_nell_cpt_set = vals['nell_concept']
                new_nell_ent_set = vals['nell_entities']
            new_nell_cpt_set = new_nell_cpt_set & self.concept_set # filter concepts with pretrained embedding
            if len(new_nell_cpt_set) > 0:
                new_info_dict[trt] = {'nell_concept': new_nell_cpt_set, 'nell_entities': new_nell_ent_set}
        return new_info_dict

    def lookup_concept_ids_single(self, tokens, entities, entity_info_dict):
        concept_ids = [[] for _ in range(len(tokens))]

        for entity in entities:
            ent_name = entity[0].lower()
            if ent_name not in entity_info_dict:
                continue
            concept_info = entity_info_dict[ent_name]
            start_token = entity[1]
            end_token = entity[2]
            for pos in range(start_token, end_token + 1):
                concept_ids[pos] += [self.concept2id[category_name] for category_name in concept_info['nell_concept']]
        return concept_ids


    def lookup_concept_ids(self, tokenization_info, **kwargs):
        token_entities_dict = self.find_entity_info(tokenization_info.query_entities + tokenization_info.doc_entities)
        new_token_entities_dict = self.update_entity_info(token_entities_dict)

        query_concept_ids = self.lookup_concept_ids_single(tokenization_info.query_tokens,
                                                           tokenization_info.query_entities,
                                                           new_token_entities_dict)
        doc_concept_ids = self.lookup_concept_ids_single(tokenization_info.doc_tokens,
                                                           tokenization_info.doc_entities,
                                                           new_token_entities_dict)

        query_max_concept_length = max([len(concepts) for concepts in query_concept_ids])
        doc_max_concept_length = max([len(concpets) for concpets in doc_concept_ids])
        max_concept_length = max(query_max_concept_length, doc_max_concept_length)
        return query_concept_ids, doc_concept_ids, max_concept_length

# remove category part of NELL entities, digit prefix 'n' and additional '_'
def preprocess_nell_ent_name(raw_name):
    ent_name = raw_name.split(':')[-1]
    digits = set(string.digits)
    if ent_name.startswith('n') and all([char in digits for char in ent_name.split('_')[0][1:]]):
        ent_name = ent_name[1:]
    ent_name = "_".join(filter(lambda x: len(x) > 0, ent_name.split('_')))
    return ent_name


def preprocess_record_ent_name(raw_str):
    raw_str.replace(" ", "_")
    raw_str = "".join(c for c in raw_str if c == "_" or c not in puncs)
    return raw_str
