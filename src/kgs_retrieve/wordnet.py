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

import string
import logging
import nltk
import os
from nltk.corpus import wordnet as wn
from kgs_retrieve.baseretriever import KGRetriever, read_concept_embedding, run_strip_accents
import pickle
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
POS_LIST = ['n', 'v', 'a', 'r']

class WordnetRetriever(KGRetriever):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.name = "wordnet"
        self.max_concept_length = 0

        concept_embedding_path = os.path.join(filepath, "wn_concept2vec.txt")
        self.id2concept, self.concept2id, self.concept_embedding_mat = read_concept_embedding(
            concept_embedding_path)

        self.offset_to_wn18name_dict = {}
        fin = open(os.path.join(filepath, 'wordnet-mlj12-definitions.txt'))
        for line in fin:
            info = line.strip().split('\t')
            offset_str, synset_name = info[0], info[1]
            self.offset_to_wn18name_dict[offset_str] = synset_name
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        logger.info('Finish loading wn18 definition file.')

        self.pos = POS_LIST

        self.wn18_dir = os.path.join(self.filepath, "wn18/text")
        self.wn18_path = os.path.join(self.wn18_dir, "full.txt")

        self.synset_name_set_path = os.path.join(self.wn18_dir, "synset_name.txt")
        if not os.path.exists(self.synset_name_set_path):
            self.synset_name_set = self.create_synset_name_set()
        else:
            with open(self.synset_name_set_path, "rb") as fp:
                self.synset_name_set = set(pickle.load(fp))

        repeated_id_path = os.path.join(self.filepath, "repeated_id.npy")

        self.repeated_id = np.load(repeated_id_path, allow_pickle='TRUE').item()

        self.conceptids2synset = {}

    def create_entire_wn18_graph(self):
        wn18_full = open(self.wn18_path, 'a')

        wn18_train = open(os.path.join(self.wn18_dir, "train.txt"), 'r')
        for line in wn18_train.readlines():
            wn18_full.writelines(line, )
        wn18_train.close()

        wn18_valid = open(os.path.join(self.wn18_dir, "valid.txt"), 'r')
        for line in wn18_valid.readlines():
            wn18_full.writelines(line, )
        wn18_valid.close()

        wn18_test = open(os.path.join(self.wn18_dir, "test.txt"), 'r')
        for line in wn18_test.readlines():
            wn18_full.writelines(line, )

        wn18_test.close()
        wn18_full.close()

    def create_synset_name_set(self):
        synset_name_set = set()
        if not os.path.exists(self.wn18_path):
            self.create_entire_wn18_graph()
        wn18_full = open(os.path.join(self.wn18_dir, "full.txt"), 'r')

        for line in wn18_full.readlines():
            src, relation, dst = line.strip().split("\t")
            if src not in synset_name_set:
                synset_name_set.add(src)
            if dst not in synset_name_set:
                synset_name_set.add(dst)
        wn18_full.close()

        synset_name_list = list(synset_name_set)
        with open(self.synset_name_set_path, 'wb') as fp:
            pickle.dump(synset_name_list, fp)
        return synset_name_set

    def lookup_concept_ids_single(self, text, ori_to_tok_map, tok_num, tolower, no_stopwords, ignore_length, tokenizer, is_filter,
                                  is_lemma, is_clean, is_morphy, query=True, query_size=0):
        concept_ids = [[] for _ in range(tok_num)]
        words = text.split(" ")
        word_to_ori_map = []
        is_begin = True
        conceptids2synset = {}

        for i, c in enumerate(text):
            if is_begin:
                word_to_ori_map.append([i])
            if c == " ":
                is_begin = True
            else:
                if is_begin:
                    is_begin = False
                else:
                    word_to_ori_map[-1].append(i)
        # logger.info("text: {}".format(words))
        for i, word in enumerate(words):
            retrieve_token = run_strip_accents(word.lower()) if tolower else word
            if retrieve_token in set(string.punctuation):
                logger.debug('{} is punctuation, skipped!'.format(retrieve_token))
                continue
            if no_stopwords and retrieve_token in self.stopwords:
                logger.debug('{} is stopword, skipped!'.format(retrieve_token))
                continue
            if ignore_length > 0 and len(retrieve_token) <= ignore_length:
                logger.debug('{} is too short, skipped!'.format(retrieve_token))
                continue

            try:
                synsets = wn.synsets(retrieve_token)
            except:
                logger.warning("{} can't work in nltk".format(retrieve_token))
                synsets = []
            wn18synset_names = []
            if is_morphy:
                # logger.info("morphy match")
                morphy_set = self.get_morphy(retrieve_token)
                if retrieve_token not in morphy_set:
                    # logger.info("{} not in morphy_set{}".format(retrieve_token, morphy_set))
                    morphy_set.add(retrieve_token)
            else:
                # logger.info("exact match")
                morphy_set = None


            for synset in synsets:
                if is_filter and not self.is_in_full_wn18(synset):
                    continue

                if not is_lemma and not self.is_center_entity(synset, retrieve_token, morphy_set, is_morphy):
                    continue

                offset_str = str(synset.offset()).zfill(8)
                if offset_str in self.offset_to_wn18name_dict:
                    full_synset_name = self.offset_to_wn18name_dict[offset_str]

                    if is_clean and self.is_repeated(self.concept2id[full_synset_name]):
                        continue
                    if self.concept2id[full_synset_name] in conceptids2synset and conceptids2synset[self.concept2id[full_synset_name]] != synset:
                        logger.warning("different wn object {} {} map to the same id {}".format
                                       (conceptids2synset[self.concept2id[full_synset_name]], synset, self.concept2id[full_synset_name]))
                        if self.concept2id[full_synset_name] not in self.repeated_id:
                            self.repeated_id[self.concept2id[full_synset_name]] = [str(conceptids2synset[self.concept2id[full_synset_name]]), str(synset)]

                    wn18synset_names.append(full_synset_name)
                    conceptids2synset[self.concept2id[full_synset_name]] = synset

            if len(wn18synset_names) > 0:
                ori_index = word_to_ori_map[i]
                toks_id = []
                for ori_id in ori_index:
                    toks_id.append(ori_to_tok_map[ori_id])
                toks_id = list(set(toks_id))
                for tok_id in toks_id:
                    for synset_name in wn18synset_names:
                        concept_ids[tok_id].extend([self.concept2id[synset_name]])

        return concept_ids, conceptids2synset

    def lookup_concept_ids(self, example_tokenized, tokenizer, **kwargs):
        """
            :param tokenization_info:
            :param tokenizer_type:
            :return:

            find the concepts in wordnet, and add the ids to the corresponding tokens.
            """
        do_lower_case = kwargs.pop("do_lower_case", False)
        no_stopwords = kwargs.pop("no_stopwords", False)
        ignore_length = kwargs.pop("ignore_length", 0)
        is_filter = kwargs.pop("is_filter")
        is_lemma = kwargs.pop("is_lemma")
        is_clean = kwargs.pop("is_clean")
        is_morphy = kwargs.pop("is_morphy")

        # tolower = not do_lower_case
        tolower = True

        query_text = example_tokenized.query_text
        doc_text = example_tokenized.doc_text

        query_ori_to_tok_map, doc_ori_to_tok_map = example_tokenized.query_ori_to_tok_map, example_tokenized.doc_ori_to_tok_map
        query_concept_ids, query_conceptids2synset = self.lookup_concept_ids_single(
            query_text, query_ori_to_tok_map, len(example_tokenized.query_tokens), tolower, no_stopwords, ignore_length,
            tokenizer, is_filter=is_filter, is_lemma=is_lemma, is_clean=is_clean, is_morphy=is_morphy)
        doc_concept_ids, doc_conceptids2synset = \
            self.lookup_concept_ids_single(doc_text, doc_ori_to_tok_map, len(example_tokenized.doc_tokens), tolower,
                                           no_stopwords, ignore_length, tokenizer, is_filter=is_filter, is_lemma=is_lemma,
                                           is_clean=is_clean, is_morphy=is_morphy, query=False,
                                           query_size=len(example_tokenized.query_tok_to_ori_map))

        query_max_concept_length = max([len(concepts) for concepts in query_concept_ids])
        doc_max_concept_length = max([len(concpets) for concpets in doc_concept_ids])
        max_concept_length = max(query_max_concept_length, doc_max_concept_length)

        return query_concept_ids, doc_concept_ids, max_concept_length, query_conceptids2synset, doc_conceptids2synset

    def is_center_entity(self, entity, word, morphy_set, is_morphy):
        if len(str(entity).split("'")) == 3:
            tmp = str(entity).split("'")[1]
        else:
            tmp = str(entity).replace("')", "('").split("('")[1]

        # if is_filter and not self.is_in_full_wn18(tmp):
        #     return False

        tmp = tmp.split(".")
        if len(tmp) == 3:
            if is_morphy:
                return tmp[0] in morphy_set
            else:
                return tmp[0] == word
        else:
            tmp2 = ""
            for i, substring in enumerate(tmp):
                if i >= len(tmp)-2:
                    break
                tmp2 += substring
            if is_morphy:
                return tmp2 in morphy_set
            else:
                return tmp2 == word

    def is_in_full_wn18(self, synset_name):
        if len(str(synset_name).split("'")) == 3:
            tmp = str(synset_name).split("'")[1]
        else:
            tmp = str(synset_name).replace("')", "('").split("('")[1]

        return tmp in self.synset_name_set

    def get_morphy(self, lemma, check_exceptions=True):
        morphy_list = [form
                        for p in self.pos
                        for form in wn._morphy(lemma, p, check_exceptions)]
        return set(morphy_list)

    def is_repeated(self, id):
        return id in self.repeated_id


