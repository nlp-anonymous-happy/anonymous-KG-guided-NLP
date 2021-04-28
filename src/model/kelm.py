import os
import copy
import logging
from typing import Any, Dict

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
from transformers import (
    PretrainedConfig,
    BertModel, BertTokenizer, BertConfig,
    RobertaModel, RobertaTokenizer, RobertaConfig,
    AutoTokenizer,
    AutoConfig,
    AutoModel
)

from kgs_retrieve.kg_utils import initialize_kg_retriever
from model import multi_relation_net
import dgl
from time import time

WEIGHTS_NAME = "pytorch_model.bin"
MODEL_NAME = "model.pkl"
logger = logging.getLogger(__name__)

BertLayerNorm = nn.LayerNorm


class KELMConfig(PretrainedConfig):
    def __init__(
            self,
            pad_token_id=0,
            kgretrievers=None,
            **kwargs,
    ):
        # the base text embedding model configuratoin
        super().__init__(pad_token_id=pad_token_id)
        self.speed_up_version = kwargs.get("speed_up_version")
        self.wn_def_embed_mat_dir = kwargs.get("wn_def_embed_mat_dir")
        self.text_embed_model = kwargs.get("text_embed_model", None)
        self.use_context_graph = kwargs.get("use_context_graph")

        self.config_name = kwargs.get("config_name", "")
        # logger.info("KELM config name: {}".format(self.config_name))
        self.model_name_or_path = kwargs.get("model_name_or_path", "")
        self.cache_dir = kwargs.get("cache_dir", "")

        # if self.text_embed_model == "bert":
        #     self.base_config = BertConfig.from_pretrained(self.config_name if self.config_name else self.model_name_or_path,cache_dir=self.cache_dir)
        # elif self.text_embed_model == "roberta":
        #     self.base_config = RobertaConfig.from_pretrained(self.config_name, cache_dir= self.cache_dir)
        self.base_config = AutoConfig.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path,
            # cache_dir=self.cache_dir if self.cache_dir else None,
        )
        # self.config_name = "_".join(self.config_name.split("-"))
        init_dir = kwargs.get("init_dir", "")
        # self.pretrained_path = init_dir if len(init_dir) > 1 else None

        # the kg-related configurations
        self.use_kgs = kwargs.get("use_kgs", dict())
        self.text_embed_size = self.base_config.hidden_size
        self.concept_embed_sizes = dict()
        if kgretrievers is not None:
            self.add_kgretrievers(kgretrievers)

        self.mem_method = kwargs.get("mem_method", "raw")
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.output_attentions = False
        self.output_hidden_states = False

        # upper layesr configurations
        self.cat_mul = kwargs.get("cat_mul", True)
        self.cat_sub = kwargs.get("cat_sub", True)
        self.cat_twotime = kwargs.get("cat_twotime", True)
        self.cat_twotime_mul = kwargs.get("cat_twotime_mul", True)
        self.cat_twotime_sub = kwargs.get("cat_twotime_sub", False)
        self.set_sizes()

        self.freeze = kwargs.get("freeze")

        self.relation_list = kwargs.get("relation_list")
        self.relation_agg = kwargs.get("relation_agg")


    def set_sizes(self):
        self.memory_output_size = self.text_embed_size + sum(list(self.concept_embed_sizes.values()))

        self.output_length = self.memory_output_size * 2
        if self.cat_mul:
            self.output_length += self.memory_output_size
        if self.cat_sub:
            self.output_length += self.memory_output_size
        if self.cat_twotime:
            self.output_length += self.memory_output_size
        if self.cat_twotime_mul:
            self.output_length += self.memory_output_size
        if self.cat_twotime_sub:
            self.output_length += self.memory_output_size

    @classmethod
    def init_from_args(
            cls,
            args,
            pad_token_id=0,
            kgretrievers=None
    ):
        # the base text embedding model configuratoin
        kwargs = dict()
        kwargs['freeze'] = args.freeze

        kwargs['text_embed_model'] = args.text_embed_model
        kwargs['config_name'] = args.config_name
        kwargs['model_name_or_path'] = args.model_name_or_path
        kwargs['cache_dir'] = args.cache_dir
        kwargs['init_dir'] = args.init_dir
        kwargs['use_kgs'] = args.use_kgs
        kwargs['mem_method'] = args.mem_method
        kwargs['cat_mul'] = args.cat_mul
        kwargs['cat_sub'] = args.cat_sub
        kwargs['cat_twotime'] = args.cat_twotime
        kwargs['cat_twotime_mul'] = args.cat_twotime_mul
        kwargs['cat_twotime_sub'] = args.cat_twotime_sub

        kwargs["relation_list"] = args.relation_list
        kwargs["relation_agg"] = args.relation_agg

        kwargs["speed_up_version"] = args.speed_up_version
        kwargs["wn_def_embed_mat_dir"] = args.wn_def_embed_mat_dir
        kwargs["use_context_graph"] = args.use_context_graph

        return KELMConfig(kgretrievers=kgretrievers, **kwargs)

    def add_kgretrievers(self, retrievers):
        """
        This method could be only run once, if no retriever is given when initiating and before the KELMModel creation.
        :param retrievers:
        :return:
        """
        self.retrievers = retrievers
        for kg in self.use_kgs:
            self.concept_embed_sizes[kg] = self.retrievers[kg].get_concept_embed_size()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        base_config = output.get("base_config", None)
        if base_config is not None:
            output['base_config'] = base_config.to_dict()
            output['base_config']['architectures'] = base_config.architectures
        for kg, retriever in output["retrievers"].items():
            output['retrievers'][kg] = retriever.to_dict()
        return output

    # @classmethod
    # def from_pretrained(cls, cache_dir):
    #     return torch.load(cache_dir)
    #
    # def save_pretrained(self, save_directory: str):
    #     torch.save(self, save_directory)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        retrievers = dict()
        for key, kg_args in config_dict['retrievers'].items():
            file_path = kg_args['file_path']
            # file_path = "."+kg_args['file_path'] # when running test_metrics.py
            retrievers[key] = initialize_kg_retriever(key, file_path)
            max_length = kg_args['max_concept_length']
            retrievers[key].update_max_concept_length(max_length)

        config = cls(**config_dict)
        if len(retrievers.items()) > 0:
            config.add_kgretrievers(retrievers)
        config.set_sizes()

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

class SelfMatching(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0, very_large_value=1e6,
                 cat_mul=True, cat_sub=True, cat_twotime=True,
                 cat_twotime_mul=False, cat_twotime_sub=True):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.cat_mul = cat_mul
        self.cat_sub = cat_sub
        self.cat_twotime = cat_twotime
        self.cat_twotime_mul = cat_twotime_mul
        self.cat_twotime_sub = cat_twotime_sub
        self.very_large_value = very_large_value

        w4C = torch.empty(input_size, 1)
        w4Q = torch.empty(input_size, 1)
        w4mlu = torch.empty(1, 1, input_size)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def trilinear_for_attention(self, C):  # C [batch_size, seq_size, input_size]
        batch_size, seq_size, input_size = C.shape
        C = F.dropout(C, p=self.dropout_rate)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, seq_size])  # [batch_size, seq_size,seq_size]
        subres1 = torch.matmul(C, self.w4Q).transpose(1, 2).expand([-1, seq_size, -1])
        subres2 = torch.matmul(C * self.w4mlu, C.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res = res + self.bias
        return res

    def forward(self, input, attention_mask):
        """
        :param input: shape [batch_size, seq_size, input_size]
        :param attention_mask: shape [batch_size, seq_size, 1]
        :return:
        """
        assert len(input.shape) == 3 and len(attention_mask.shape) == 3 \
               and attention_mask.shape[-1] == 1
        assert input.shape[:2] == attention_mask.shape[:2]

        sim_score = self.trilinear_for_attention(input)

        softmax_mask = (1 - attention_mask) * self.very_large_value  # [batch_size, seq_size, 1]
        # softmax_mask = softmax_mask.float().transpose(1,2).expand((*softmax_mask.shape[:2],softmax_mask.shape[1]))
        softmax_mask = softmax_mask.float()
        softmax_mask = -1 * torch.matmul(softmax_mask, softmax_mask.transpose(1, 2))
        sim_score = sim_score + softmax_mask

        attn_prob = F.softmax(sim_score, dim=2)  # [batch_size, seq_size, seq_size]
        weighted_sum = torch.matmul(attn_prob, input)  # [batch_size, seq_size, input_size]

        if any([self.cat_twotime_mul, self.cat_twotime, self.cat_twotime_sub]):
            twotime_att_prob = torch.matmul(attn_prob, attn_prob)
            twotime_weighted_sum = torch.matmul(twotime_att_prob, input)

        out_tensors = torch.cat((input, weighted_sum), dim=2)
        if self.cat_mul:
            out_tensors = torch.cat((out_tensors, torch.mul(input, weighted_sum)), dim=2)
        if self.cat_sub:
            out_tensors = torch.cat((out_tensors, input - weighted_sum), dim=2)
        if self.cat_twotime:
            out_tensors = torch.cat((out_tensors, twotime_weighted_sum), dim=2)
        if self.cat_twotime_mul:
            out_tensors = torch.cat((out_tensors, torch.mul(input, twotime_weighted_sum)), dim=2)
        if self.cat_twotime_sub:
            out_tensors = torch.cat((out_tensors, input - twotime_weighted_sum), dim=2)

        return out_tensors


class KELM(nn.Module):
    config_class = KELMConfig
    base_model_prefix = "KELM"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_context_graph = self.config.use_context_graph
        self.use_kgs = self.config.use_kgs
        self.text_embed_model = None
        logger.info("***** Initializing KELM model *****")
        logger.info(
            "Building the first layer Text embedding layer with type {} and size {}".format(config.text_embed_model,
                                                                                            config.text_embed_size))

        self.text_embed_model = AutoModel.from_pretrained(config.model_name_or_path,
                                                          config=config.base_config,
                                                          # cache_dir=config.pretrained_path
                                                          )
        # freeze bert
        if config.freeze:
            for param in self.text_embed_model.parameters():
                param.requires_grad = False

        for kg, retriever in config.retrievers.items():
            # if kg == "nell":
            #     self.nell_embed_mat = retriever.get_embedding_mat()
            if kg == "wordnet":
                wn_embed_mat = retriever.get_embedding_mat()

                concept_embed_size = retriever.get_concept_embed_size()
                concepts_embed_number = retriever.get_concept_vocab_size()

                self.wn_embedding = nn.Embedding(concepts_embed_number,
                                              concept_embed_size)  # embed the concept_ids to concept_embeddings
                self.wn_embedding.weight.data.copy_(torch.from_numpy(wn_embed_mat))
                self.wn_embedding.weight.requires_grad = False
            if kg == "nell":
                nell_embed_mat = retriever.get_embedding_mat()

                concept_embed_size = retriever.get_concept_embed_size()
                concepts_embed_number = retriever.get_concept_vocab_size()

                self.nell_embedding = nn.Embedding(concepts_embed_number,
                                              concept_embed_size)  # embed the concept_ids to concept_embeddings
                self.nell_embedding.weight.data.copy_(torch.from_numpy(nell_embed_mat))
                self.nell_embedding.weight.requires_grad = False
        if self.use_context_graph:
            wn_def_embed_mat = torch.load(config.wn_def_embed_mat_dir)["definition_embedding_mat"]
            logger.info("wn_def_embed_mat shape: {}".format(wn_def_embed_mat.shape))
            # self.wn_def_embedding = nn.Embedding(31863,
            #                                  1024)

            self.wn_def_embedding = nn.Embedding(wn_def_embed_mat.shape[0],
                                                 wn_def_embed_mat.shape[1])
            self.wn_def_embedding.weight.data.copy_(wn_def_embed_mat)

            self.wn_def_embedding.weight.requires_grad = False

        if config.text_embed_model == "bert" or config.text_embed_model == "roberta":
            self.rgcn_output_size = 1024
            self.concept_embed_size = 100
            self.bert_embedding_size = 1024
            self.hidden_size = 1024
        elif config.text_embed_model == "roberta_base":
            self.rgcn_output_size = 768
            self.concept_embed_size = 100
            self.bert_embedding_size = 768
            self.hidden_size = 768
        else:
            logger.warning("not recongize embed model, program exit")
            exit()

        num_part = self.use_context_graph + len(self.use_kgs)
        self.memory_output_size = self.bert_embedding_size + self.concept_embed_size * num_part
        self.output_length = 6 * self.memory_output_size


        logger.info("Building the third layer: Self-Matching Layer with size {}".format(config.memory_output_size))
        self.self_matching = SelfMatching(self.memory_output_size, cat_mul=config.cat_mul,
                                          cat_sub=config.cat_sub, cat_twotime=config.cat_twotime,
                                          cat_twotime_mul=config.cat_twotime_mul,
                                          cat_twotime_sub=config.cat_twotime_sub)

        logger.info("Building the rgcn layer")

        gnn_relation_list = ['self_connection']
        logger.warning("gnn_relation_list: {}".format(gnn_relation_list))
        for i in self.config.relation_list:
            gnn_relation_list.append(i+'_')
        logger.warning("gnn_relation_list: {}".format(gnn_relation_list))

        if self.use_context_graph:
            self.rgcn_context = multi_relation_net.RGAT(self.bert_embedding_size,
                                                    self.bert_embedding_size,
                                                    self.bert_embedding_size,
                                                    rel_names=gnn_relation_list,
                                                    relation_agg=self.config.relation_agg,
                                                    att_out_feats=self.bert_embedding_size,
                                                        )

            self.gat_context = multi_relation_net.GAT(num_layers=1,
                                                      in_dim=self.bert_embedding_size,
                                                      num_hidden=self.concept_embed_size,
                                                      heads=[1],
                                                      feat_drop=0,
                                                      attn_drop=0,
                                                      negative_slope=0.2,
                                                      activation=None,
                                                      residual=False)

        for kg in self.use_kgs:
            if kg == "wordnet":
                self.rgcn_wn = multi_relation_net.RGAT(self.concept_embed_size,
                                                        self.concept_embed_size,
                                                        self.concept_embed_size,
                                                        rel_names=gnn_relation_list,
                                                        relation_agg=self.config.relation_agg,
                                                        att_out_feats=self.concept_embed_size,
                                                       )

                self.gat_wn = multi_relation_net.GAT(num_layers=1,
                                                     in_dim=self.concept_embed_size,
                                                     num_hidden=self.concept_embed_size,
                                                     heads=[1],
                                                     feat_drop=0,
                                                     attn_drop=0,
                                                     negative_slope=0.2,
                                                     activation=None,
                                                     residual=False)
            if kg=="nell":
                self.gat_nell = multi_relation_net.GAT(num_layers=1,
                                                       in_dim=self.concept_embed_size,
                                                       num_hidden=self.concept_embed_size,
                                                       heads=[1],
                                                       feat_drop=0,
                                                       attn_drop=0,
                                                       negative_slope=0.2,
                                                       activation=None,
                                                       residual=False)

        logger.info("Building the fourth layer: Output Layer")
        # self.graph_qa_kt_outputs = nn.Linear(2*(self.bert_embedding_size+self.concept_embed_size), 2)
        self.qa_kt_outputs = nn.Linear(self.output_length, 2)
        self.qa_kt_outputs.weight.data.normal_(0, config.initializer_range)

        logger.info("Building projector layer")

        self.projected_token_text = nn.Linear(self.bert_embedding_size, self.concept_embed_size,
                                                  bias=False)  # map the bert encoding to the pre-defined dimension.
        self.projected_token_text.weight.data.normal_(0, config.initializer_range)

        self.projected_token_text_nell = nn.Linear(self.bert_embedding_size, self.concept_embed_size,
                                                  bias=False)  # map the bert encoding to the pre-defined dimension.
        self.projected_token_text_nell.weight.data.normal_(0, config.initializer_range)
        # self.init_weights()
        logger.info("***** The parameters information in KELM *****")
        logger.info(" The shape of text embed : {}".format(config.text_embed_size))
        logger.info(" The shape of memory output : {}".format(config.memory_output_size))
        logger.info(" The shape of last layer: {}".format(config.output_length))

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    def update_definition_embedding_mat(self, definition_embedding_mat):
        # wn_def_embed_mat = definition_embedding_mat

        logger.info("definition_embedding_mat {}".format(definition_embedding_mat))
        logger.info("og weight {}".format(self.wn_def_embedding.weight.data))

        self.wn_def_embedding = nn.Embedding(definition_embedding_mat.shape[0],
                                             definition_embedding_mat.shape[1])

        self.wn_def_embedding.weight.data.copy_(definition_embedding_mat)
        self.wn_def_embedding.weight.requires_grad = False
        logger.info("current weight {}".format(self.wn_def_embedding.weight.data))

        # self.wn_def_embedding.to(self.text_embed_model.device)

    def forward(self, **kwargs):
        """
        :param input_ids: shape: [batch_size, max_seq_length (,1)]. e.g. [101 16068 1551 131 11253 10785 7637 3348 113 1286 114 1105 19734 1123 1493 113 1268 114 1112 1131 4927 1123 1159 1113 1103 2037 1437 1114 1123 3235 137 1282 14507 2636 102 1650 3696 9255 153 2591 13360 6258 3048 10069 131 5187 131 3927 142 9272 117 1367 1347 1381 197 19753 11392 12880 2137 131 1367 131 1512 142 9272 117 1367 1347 1381 11253 10785 7637 1144 3090 1131 1110 7805 1123 1148 2027 1114 20497 1389 27891 1667 11247 119 1109 3081 118 1214 118 1385 2851 117 1150 1640 1144 1300 1482 1121 2166 6085 117 1163 1107 1126 3669 1113 1109 4258 157 18963 7317 2737 3237 1115 1131 1110 17278 1106 1129 20028 1330 1901 1106 1123 9304 13465 119 1153 1163 131 112 1284 787 1396 1198 1276 1149 1195 787 1231 1515 170 2963 118 146 787 182 1210 1808 6391 119 146 1138 2094 1105 170 2963 1107 1139 7413 117 1103 1436 2053 1107 1103 1362 117 170 1632 2261 1105 146 787 182 170 1304 6918 1873 119 146 787 182 1304 9473 119 112 137 13426 11253 117 3081 117 1110 1210 1808 6391 1114 1123 3049 2963 137 13426 18662 18284 5208 2483 1163 1131 5115 1176 112 170 1304 6918 1873 112 137 13426 11253 1105 1393 4896 1591 1667 1508 1147 4655 1113 2080 1165 1131 1108 3332 19004 1111 170 1248 1159 1171 1107 1351 102]
        :param attention_mask: [batch_size, max_seq_length(, 1)]. e.g. [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        :param kwargs (optional input):
            start_positions: [batch_size(,1)]
            end_positions: [batch_size (,1)]
            token_type_ids: [batch_size, max_seq_length(, 1)]. e.g. [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
            wordnet_concept_ids: [batch_size, max_seq_length, max_wn_length]. e.g. [[0,0,0,0,0],[0,1,0,0,0],[92,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
            nell_concept_ids: [batch_size, max_seq_length, max_nell_length]. e.g. 0:[] 1:[] 2:[] 3:[] 4:[19, 92, 255] 5:[19, 92, 255] 6:[19, 92, 255] 7:[] 8:[] 9:[] 10:[] 11:[] 12:[] 13:[] 14:[] 15:[] 16:[] 17:[] 18:[] 19:[] 20:[] 21:[] 22:[] 23:[] 24:[] 25:[] 26:[] 27:[] 28:[] 29:[] 30:[] 31:[] 32:[] 33:[] 34:[] 35:[] 36:[] 37:[] 38:[] 39:[] 40:[] 41:[] 42:[] 43:[] 44:[] 45:[] 46:[] 47:[] 48:[] 49:[] 50:[] 51:[] 52:[] 53:[] 54:[] 55:[] 56:[] 57:[] 58:[] 59:[] 60:[] 61:[] 62:[] 63:[] 64:[] 65:[] 66:[] 67:[] 68:[] 69:[19, 92, 255] 70:[19, 92, 255] 71:[19, 92, 255] 72:[] 73:[] 74:[] 75:[] 76:[] 77:[] 78:[] 79:[] 80:[] 81:[] 82:[] 83:[] 84:[] 85:[] 86:[] 87:[] 88:[] 89:[] 90:[] 91:[] 92:[] 93:[] 94:[] 95:[] 96:[] 97:[] 98:[] 99:[] 100:[] 101:[] 102:[] 103:[] 104:[] 105:[] 106:[] 107:[] 108:[] 109:[] 110:[] 111:[] 112:[] 113:[] 114:[] 115:[] 116:[] 117:[] 118:[] 119:[] 120:[] 121:[] 122:[] 123:[] 124:[] 125:[] 126:[] 127:[] 128:[] 129:[] 130:[] 131:[] 132:[] 133:[] 134:[] 135:[] 136:[] 137:[] 138:[] 139:[] 140:[] 141:[] 142:[] 143:[] 144:[] 145:[] 146:[] 147:[] 148:[] 149:[] 150:[] 151:[] 152:[] 153:[] 154:[] 155:[] 156:[] 157:[] 158:[] 159:[] 160:[] 161:[] 162:[] 163:[] 164:[] 165:[] 166:[] 167:[] 168:[] 169:[] 170:[] 171:[] 172:[] 173:[] 174:[] 175:[] 176:[] 177:[] 178:[] 179:[] 180:[] 181:[] 182:[] 183:[] 184:[] 185:[] 186:[] 187:[] 188:[] 189:[] 190:[] 191:[] 192:[50, 239] 193:[] 194:[] 195:[] 196:[] 197:[] 198:[] 199:[] 200:[] 201:[] 202:[] 203:[] 204:[] 205:[] 206:[] 207:[] 208:[] 209:[] 210:[] 211:[] 212:[] 213:[] 214:[] 215:[] 216:[] 217:[] 218:[] 219:[] 220:[] 221:[] 222:[50, 239] 223:[] 224:[] 225:[] 226:[] 227:[138, 91] 228:[] 229:[] 230:[] 231:[] 232:[] 233:[] 234:[] 235:[] 236:[] 237:[] 238:[] 239:[] 240:[] 241:[] 242:[] 243:[] 244:[] 245:[]
        :return:
        """
        # start_forward_time = time()
        input_ids = kwargs.get("input_ids")
        # logger.info("rank:{}".format(input_ids.device))
        attention_mask = kwargs.get("attention_mask")
        loss_wn = 0.0
        loss_cls = 0.0
        batch_synset_graphs_id = kwargs.get("batch_synset_graphs")
        wn_synset_graphs = kwargs.get("wn_synset_graphs")
        batch_synset_graphs = [wn_synset_graphs[i] for i in batch_synset_graphs_id]
        batch_context_graphs_list = []
        batch_wn_graphs_list = []
        batch_entity2token_graphs_wn_list = []
        batch_entity2token_graphs_nell_list = []

        token_length_list = []

        if self.config.text_embed_model == "bert":
            text_output = self.text_embed_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=kwargs.get("token_type_ids"),
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states
            )[0]
        elif self.config.text_embed_model == "roberta" or self.config.text_embed_model == "roberta_base":
            text_output = self.text_embed_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]
        relation_list = self.config.relation_list

        inverse_relation_list = []
        # node_type in origin graph
        id_type_list = []
        context_type_list = []
        for i, relation_type in enumerate(relation_list):
            inverse_relation_list.append("{}_".format(relation_type))

            id_type = "wn{}_id".format(relation_type)
            id_type_list.append(id_type)

            context_type = "wn{}_context".format(relation_type)
            context_type_list.append(context_type)

        # start_time = time()
        for i, g in enumerate(batch_synset_graphs):
            assert (len(g.nodes("token_id")) == torch.sum(attention_mask[i, :]))
            token_length_list.append(len(g.nodes("token_id")))

            # reconstruct context graph
            context_g, wn_g = self.reconstruct_dgl_graph(g, relation_list, inverse_relation_list,
                                                         id_type_list, context_type_list,
                                                         text_output[i, :, :], input_ids.device)

            entity2token_graph = self.construct_entity2token_graph(i, g, text_output, input_ids.device)

            if "wordnet" in self.use_kgs or self.use_context_graph:
                batch_entity2token_graphs_wn_list.append(entity2token_graph["wordnet"])
            if "nell" in self.use_kgs:
                batch_entity2token_graphs_nell_list.append(entity2token_graph["nell"])
            batch_context_graphs_list.append(context_g)
            batch_wn_graphs_list.append(wn_g)

        memory_output_new = text_output

        if self.use_context_graph:
            batch_context_graphs_dgl = dgl.batch(batch_context_graphs_list)
            graph_context_embedding = self.rgcn_context(batch_context_graphs_dgl, batch_context_graphs_dgl.ndata['feature'])
            batch_context_graphs_dgl.nodes["wn_concept_context"].data["feature"] = graph_context_embedding[
                "wn_concept_context"]
            # batch_context_graphs_dgl.nodes["wn_concept_context"].data["feature_project"] = self.bert_projected_token_ids(
            #     graph_context_embedding["wn_concept_context"])
            batch_context_graphs_list = dgl.unbatch(batch_context_graphs_dgl)

            context_embed_new = torch.zeros(
                (memory_output_new.shape[0], memory_output_new.shape[1], self.concept_embed_size),
                dtype=torch.float32, device=input_ids.device)

        for kg in self.use_kgs:
            if kg=="wordnet":
                batch_wn_graphs_dgl = dgl.batch(batch_wn_graphs_list)
                graph_wn_embedding = self.rgcn_wn(batch_wn_graphs_dgl, batch_wn_graphs_dgl.ndata['feature'])
                batch_wn_graphs_dgl.nodes["wn_concept_id"].data["feature"] = graph_wn_embedding["wn_concept_id"]
                batch_wn_graphs_list = dgl.unbatch(batch_wn_graphs_dgl)

                concept_embed_new = torch.zeros(
                    (memory_output_new.shape[0], memory_output_new.shape[1], self.concept_embed_size),
                    dtype=torch.float32, device=input_ids.device)

            if kg=="nell":
                nell_embed_new = torch.zeros(
                    (memory_output_new.shape[0], memory_output_new.shape[1], self.concept_embed_size),
                    dtype=torch.float32, device=input_ids.device)

        # start_time = time()
        for idx, _ in enumerate(batch_synset_graphs):
            if "wordnet" in self.use_kgs or self.use_context_graph:
                g_e2t = batch_entity2token_graphs_wn_list[idx]
            if self.use_context_graph:
                g_e2t.nodes["wn_concept_id"].data["context_feature"] = batch_context_graphs_list[idx].nodes["wn_concept_context"].data["feature"]
                g_e2t.nodes["sentinel_id"].data["context_feature"] = torch.zeros_like(g_e2t.nodes["token_id"].data["context_feature"], device=input_ids.device)
            if "wordnet" in self.use_kgs:
                g_e2t.nodes["wn_concept_id"].data["id_feature"] = batch_wn_graphs_list[idx].nodes["wn_concept_id"].data["feature"]
                g_e2t.nodes["token_id"].data["id_feature"] = self.projected_token_text(g_e2t.nodes["token_id"].data["context_feature"])
                g_e2t.nodes["sentinel_id"].data["id_feature"] = torch.zeros_like(g_e2t.nodes["token_id"].data["id_feature"], device=input_ids.device)

            if self.use_context_graph and "wordnet" in self.use_kgs:
                g_e2t_homo = dgl.to_homogeneous(g_e2t, ndata=['id_feature', 'context_feature'])
                g_e2t_homo.ndata['context_feature'] = self.gat_context(g_e2t_homo,
                                 g_e2t_homo.ndata['context_feature'])
                g_e2t_homo.ndata['id_feature'] = self.gat_wn(g_e2t_homo, g_e2t_homo.ndata['id_feature'])
                tmp_graph = dgl.to_heterogeneous(g_e2t_homo, g_e2t.ntypes, g_e2t.etypes)

                concept_embed_new[idx, :tmp_graph.num_nodes("token_id"), :] = tmp_graph.nodes["token_id"].data["id_feature"]
                context_embed_new[idx, :tmp_graph.num_nodes("token_id"), :] = tmp_graph.nodes["token_id"].data["context_feature"]
            elif self.use_context_graph:
                g_e2t_homo = dgl.to_homogeneous(g_e2t, ndata=['context_feature'])
                g_e2t_homo.ndata['context_feature'] = self.gat_context(g_e2t_homo,
                                 g_e2t_homo.ndata['context_feature'])
                tmp_graph = dgl.to_heterogeneous(g_e2t_homo, g_e2t.ntypes, g_e2t.etypes)
                context_embed_new[idx, :tmp_graph.num_nodes("token_id"), :] = tmp_graph.nodes["token_id"].data["context_feature"]
            elif "wordnet" in self.use_kgs:
                g_e2t_homo = dgl.to_homogeneous(g_e2t, ndata=['id_feature'])
                g_e2t_homo.ndata['id_feature'] = self.gat_wn(g_e2t_homo, g_e2t_homo.ndata['id_feature'])
                tmp_graph = dgl.to_heterogeneous(g_e2t_homo, g_e2t.ntypes, g_e2t.etypes)
                concept_embed_new[idx, :tmp_graph.num_nodes("token_id"), :] = tmp_graph.nodes["token_id"].data["id_feature"]

            # test_id_embed(tmp_graph)
            # batch_entity2token_graphs_list_homo_s.append(tmp_graph)
            if "nell" in self.use_kgs:
                g_e2t_nell = batch_entity2token_graphs_nell_list[idx]
                g_e2t_nell_homo = dgl.to_homogeneous(g_e2t_nell, ndata=['id_feature'])
                g_e2t_nell_homo.ndata['id_feature'] = self.gat_nell(g_e2t_nell_homo, g_e2t_nell_homo.ndata['id_feature'])
                tmp_graph_nell = dgl.to_heterogeneous(g_e2t_nell_homo, g_e2t_nell.ntypes, g_e2t_nell.etypes)
                nell_embed_new[idx, :tmp_graph_nell.num_nodes("token_id"), :] = tmp_graph_nell.nodes["token_id"].data["id_feature"]
            # test_nell_id_embed(tmp_graph_nell)
        # # logger.info("time for one by one: {}".format(time() - start_time))
        if "nell" in self.use_kgs:
            memory_output_new = torch.cat((memory_output_new, nell_embed_new), 2)

        if self.use_context_graph and "wordnet" in self.use_kgs:
            k_memory = torch.cat((concept_embed_new, context_embed_new), 2)
        elif self.use_context_graph:
            k_memory = context_embed_new
        elif "wordnet" in self.use_kgs:
            k_memory = concept_embed_new
        else:
            k_memory = torch.tensor([], device=input_ids.device)
        memory_output_new = torch.cat((memory_output_new, k_memory), 2)

        # 3rd: self-matching layer
        att_output = self.self_matching(memory_output_new,
                                        attention_mask.unsqueeze(2))  # [batch_size, max_seq_length, memory_output_size]
        # 4th layer: output layer
        logits = self.qa_kt_outputs(att_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        start_positions = kwargs.get("start_positions", None)
        end_positions = kwargs.get("end_positions", None)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            loss_cls = total_loss.detach().clone()

            loss_wn = torch.tensor(loss_wn, device=loss_cls.device)
        loss_dic = {
            'loss_wn': loss_wn,
            'loss_cls': loss_cls,
        }
        output = (start_logits, end_logits)

        # logger.info("time for forward: {}".format(time()-start_forward_time))
        return ((total_loss,) + output + (loss_dic,)) if total_loss is not None else output

    def construct_entity2token_graph(self, i, g, text_output, device):
        entity2token_graph = {}

        if "wordnet" in self.use_kgs or self.use_context_graph:
            # build wn attention graph
            data_dict = {
                ("wn_concept_id", "synset_", "token_id"): g.edges(etype="synset_"),
                ("sentinel_id", "sentinel", "token_id"): (g.nodes("token_id"), g.nodes("token_id")),
            }
            num_nodes_dict = {"token_id": g.num_nodes("token_id"), "sentinel_id": g.num_nodes("token_id"),
                              "wn_concept_id": g.num_nodes("wn_concept_id"),}
            entity2token_graph_wn = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict).to(device)
            entity2token_graph_wn.nodes["token_id"].data["context_feature"] = text_output[i,
                                                                           :entity2token_graph_wn.num_nodes("token_id"), :]

            entity2token_graph["wordnet"] = entity2token_graph_wn
        if "nell" in self.use_kgs:
            # build nell attention graph
            data_dict_nell = {
                ("nell_concept_id", "belong_", "token_id"): g.edges(etype="belong_"),
                ("sentinel_id", "sentinel", "token_id"): (g.nodes("token_id"), g.nodes("token_id")),
            }
            num_nodes_dict_nell = {"token_id": g.num_nodes("token_id"), "sentinel_id": g.num_nodes("token_id"),
                              "nell_concept_id": g.num_nodes("nell_concept_id"),}
            entity2token_graph_nell = dgl.heterograph(data_dict_nell, num_nodes_dict=num_nodes_dict_nell).to(device)
            entity2token_graph_nell.nodes["token_id"].data["id_feature"] = self.projected_token_text_nell(
                text_output[i, :entity2token_graph_nell.num_nodes("token_id"), :])
            entity2token_graph_nell.nodes['nell_concept_id'].data["id_feature"] = self.nell_embedding(
                g.nodes['nell_concept_id'].data["conceptid"].to(device))

            entity2token_graph_nell.nodes['sentinel_id'].data["id_feature"] = torch.zeros_like(
                entity2token_graph_nell.nodes['token_id'].data["id_feature"], device=device)

            entity2token_graph["nell"] = entity2token_graph_nell

        return entity2token_graph

    def reconstruct_dgl_graph(self, g, relation_list, inverse_relation_list, id_type_list, context_type_list,
                              text_output, device):
        # reconstruct context graph
        if self.use_context_graph:
            context_data_dict = {
                ("wn_concept_context", "self_connection", "wn_concept_context"): (
                    g.nodes("wn_concept_id"), g.nodes("wn_concept_id")),
            }
            context_num_nodes_dict = {"wn_concept_context": g.num_nodes("wn_concept_id"), }

            for index, relation_type in enumerate(relation_list):
                context_data_dict[(context_type_list[index], inverse_relation_list[index], "wn_concept_context")] = \
                    g.edges(etype=relation_type)[1], g.edges(etype=relation_type)[0]

                context_num_nodes_dict[context_type_list[index]] = g.num_nodes(id_type_list[index])

            context_g = dgl.heterograph(context_data_dict, num_nodes_dict=context_num_nodes_dict).to(device)


            context_g.nodes['wn_concept_context'].data["feature"] = self.wn_def_embedding(g.nodes['wn_concept_id'].data["conceptid"].to(device))

            for index, context_type in enumerate(context_type_list):
                if not context_g.num_nodes(context_type):
                    continue
                # context_g.nodes[context_type].data["feature"] = self.wn_def_embed_mat[g.nodes[id_type_list[index]].data["definition_id"].numpy()].to(device)
                context_g.nodes[context_type].data["feature"] = self.wn_def_embedding(g.nodes[id_type_list[index]].data["conceptid"].to(device))

        else:
            context_g = None

        if "wordnet" in self.use_kgs:
            # reconstruct kge graph
            wn_data_dict = {
                ('wn_concept_id', 'self_connection', 'wn_concept_id'): (g.nodes("wn_concept_id"), g.nodes("wn_concept_id")),
            }
            num_nodes_dict = {"wn_concept_id": g.num_nodes("wn_concept_id"), }

            for i, relation_type in enumerate(relation_list):
                wn_data_dict[(id_type_list[i], inverse_relation_list[i], "wn_concept_id")] = \
                    g.edges(etype=relation_type)[1], g.edges(etype=relation_type)[0]

                num_nodes_dict[id_type_list[i]] = g.num_nodes(id_type_list[i])

            wn_g = dgl.heterograph(wn_data_dict, num_nodes_dict=num_nodes_dict).to(device)


            wn_g.nodes['wn_concept_id'].data["feature"] = self.wn_embedding(g.nodes['wn_concept_id'].data["conceptid"].to(device))

            for i, id_type in enumerate(id_type_list):
                if not wn_g.num_nodes(id_type):
                    continue

                wn_g.nodes[id_type].data["feature"] = self.wn_embedding(g.nodes[id_type].data["conceptid"].to(device))
        else:
            wn_g = None

        return context_g, wn_g

    def save_pretrained(self, save_directory):
        output_model_file = os.path.join(save_directory, MODEL_NAME)
        torch.save(self.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def from_pretrained(self, init_dir):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % torch.distributed.get_rank()}
        logger.info("map_location: {}".format(map_location))
        self.load_state_dict(torch.load(os.path.join(init_dir, MODEL_NAME), map_location=map_location))


def configure_tokenizer_model_kelm(args, logger, kgretrievers):
    logger.info("***** Loading {} tokenizer for KELM *****".format(args.text_embed_model))

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.tokenizer_path,
            do_lower_case=args.do_lower_case,
            # cache_dir=args.cache_dir if args.cache_dir else None,
        )

    except:
        raise NotImplementedError("Model {}'s tokenizer has not been implemented!".format(args.text_embed_model))

    logger.info("***** init config and model *****")
    config = KELMConfig.init_from_args(args, kgretrievers=kgretrievers)
    logger.info("use both context and kge graph")
    model = KELM(config)

    try:
        logger.info("***** Loading KELM from checkpoint {} *****".format(args.init_dir))
        model.from_pretrained(args.init_dir)
        logger.info("***** Load is done *****")

    except:
        logger.info("Configuration has not been saved and create a new one arguments.")

    return tokenizer, model

    # return tokenizer, None


def check_params(target, model):
    a = True
    for target_param, param in zip(target.parameters(), model.parameters()):
        if not a:
            break
        logger.info("self.definition_embed_model.parameters()", target_param)
        logger.info("self.text_embed_model.parameters()", param)

def test_id_embed(graph, ):
    token_ids_with_concept = graph.edges(etype=('wn_concept_id', 'synset_', 'token_id'))[1].unique()
    combined = torch.cat((token_ids_with_concept, graph.nodes("token_id")))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]

    logger.info("token_id context_feature token_ids_with_concept")
    logger.info(torch.sum(graph.nodes["token_id"].data["context_feature"][token_ids_with_concept, :], dim=1))
    logger.info("token_id context_feature difference")
    logger.info(torch.sum(graph.nodes["token_id"].data["context_feature"][difference, :], dim=1))

    logger.info("token_id id_feature token_ids_with_concept")
    logger.info(torch.sum(graph.nodes["token_id"].data["id_feature"][token_ids_with_concept, :], dim=1))
    logger.info("token_id id_feature difference")
    logger.info(torch.sum(graph.nodes["token_id"].data["id_feature"][difference, :], dim=1))

def test_nell_id_embed(graph, ):
    try:
        token_ids_with_concept = graph.edges(etype=('nell_concept_id', 'belong_', 'token_id'))[1].unique()
        combined = torch.cat((token_ids_with_concept, graph.nodes("token_id")))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]

        # logger.info("token_id context_feature token_ids_with_concept")
        # logger.info(torch.sum(graph.nodes["token_id"].data["context_feature"][token_ids_with_concept, :], dim=1))
        # logger.info("token_id context_feature difference")
        # logger.info(torch.sum(graph.nodes["token_id"].data["context_feature"][difference, :], dim=1))

        logger.info("token_id id_feature token_ids_with_concept")
        logger.info(torch.sum(graph.nodes["token_id"].data["id_feature"][token_ids_with_concept, :], dim=1))
        logger.info("token_id id_feature difference")
        logger.info(torch.sum(graph.nodes["token_id"].data["id_feature"][difference, :], dim=1))
    except:
        logger.info("no concept")
        logger.info(torch.sum(graph.nodes["token_id"].data["id_feature"], dim=1))
