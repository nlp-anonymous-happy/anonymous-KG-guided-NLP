import torch
from kgs_retrieve.wordnet import WordnetRetriever
from kgs_retrieve.nell import NellRetriever
import os
import logging
logger = logging.getLogger(__name__)

def initialize_kg_retriever(kg_name, file_path, data_path, model_tpye="",load=True):
    if load:
        if os.path.exists(file_path):
            logger.info("reuse kg_retriever")
            retriever = torch.load(file_path + kg_name + model_tpye)
            return retriever

    if kg_name == "wordnet":
        logger.info("init new wordnet kg_retriever")
        return WordnetRetriever(data_path)
    elif kg_name == "nell":
        logger.info("init new nell kg_retriever")
        return NellRetriever(data_path)
    else:
        raise ValueError
