import os
from nltk.corpus import wordnet as wn
import logging

logger = logging.getLogger(__name__)
class RelationalGraphBuilder(object):
    wn_src = []
    wn_dst = []

    wn_conceptid2nodeid = {}
    wn_nodeid2conceptid = []

    wn_conceptids2synset = None

    def __init__(self, f, relation_type, wn18_dir, offset_to_wn18name_dict, concept2id, single_relation_dict, retrievers, is_filter=False):
        RelationalGraphBuilder.wn_src = []
        RelationalGraphBuilder.wn_dst = []
        RelationalGraphBuilder.wn_conceptid2nodeid = {}
        RelationalGraphBuilder.wn_nodeid2conceptid = []
        RelationalGraphBuilder.wn_conceptids2synset = f.kgs_conceptids2synset["wordnet"]
        RelationalGraphBuilder.retrievers = retrievers

        self.wn_relation_conceptid2nodeid = {}
        self.wn_relation_nodeid2conceptid = []
        self.wn_relation_src = []
        self.wn_relation_dst = []
        self.relation_type = relation_type
        self.wn18_dir = wn18_dir
        self.offset_to_wn18name_dict = offset_to_wn18name_dict
        self.concept2id = concept2id
        self.entity = None
        self.single_relation_dict = single_relation_dict
        self.is_filter = is_filter

    @classmethod
    def update_wn_src(cls, token_id):
        cls.wn_src.append(token_id)

    @classmethod
    def update_wn_dst(cls, concept_id):
        cls.wn_dst.append(cls.wn_conceptid2nodeid[concept_id])

    @classmethod
    def update_wn_conceptid2nodeid(cls, concept_id):
        cls.wn_conceptid2nodeid[concept_id] = len(cls.wn_conceptid2nodeid)

    @classmethod
    def update_wn_nodeid2conceptid(cls, concept_id):
        cls.wn_nodeid2conceptid.append(concept_id)

    @classmethod
    def is_update_relation_graph(cls, concept_id):
        return concept_id not in cls.wn_conceptid2nodeid

    def get_entity(self, concept_id):
        self.entity = RelationalGraphBuilder.wn_conceptids2synset[concept_id]

    def update_dst_id_list(self,concept_id):
        single_wn_relation_dst = []
        self.get_entity(concept_id)
        single_dst_list = self.retrive_dst_item_lists()
        single_dst_concept_id_list = self.look_up_concept_id(single_dst_list)

        for i in single_dst_concept_id_list:
            if i not in self.wn_relation_conceptid2nodeid:
                self.wn_relation_conceptid2nodeid[i] = len(self.wn_relation_conceptid2nodeid)
                self.wn_relation_nodeid2conceptid.append(i)

            single_wn_relation_dst.append(self.wn_relation_conceptid2nodeid[i])

        self.wn_relation_src.extend([RelationalGraphBuilder.wn_conceptid2nodeid[concept_id] for _, _ in enumerate(single_wn_relation_dst)])
        self.wn_relation_dst.extend(single_wn_relation_dst)

    def retrive_dst_item_lists(self):
        dst_list = []
        if self.entity in self.single_relation_dict:
            # logger.info("current entity: {}".format(self.entity))
            dst_list = self.single_relation_dict[self.entity]
        return dst_list

    def look_up_concept_id(self, entity_list):
        concept_id_list = []
        for dst_entity in entity_list:
            if len(str(dst_entity).split("'")) == 3:
                str_dst_entity = str(dst_entity).split("'")[1]
            else:
                str_dst_entity = str(dst_entity).replace("')", "('").split("('")[1]
            if self.is_filter and not RelationalGraphBuilder.retrievers.is_in_full_wn18(str_dst_entity):
                continue

            offset_str = str(dst_entity.offset()).zfill(8)
            if offset_str in self.offset_to_wn18name_dict:
                synset_name = self.offset_to_wn18name_dict[offset_str]
                concept_id = self.concept2id[synset_name]
                concept_id_list.append(concept_id)

                if concept_id in RelationalGraphBuilder.wn_conceptids2synset:
                    continue

                RelationalGraphBuilder.wn_conceptids2synset[concept_id] = str_dst_entity

                try:
                    wn.synset(RelationalGraphBuilder.wn_conceptids2synset[concept_id])
                except:
                    logger.warning("error!!!!!!!!!!!! wrong synset:{}".format(RelationalGraphBuilder.wn_conceptids2synset[concept_id]))
                    exit()
        return concept_id_list