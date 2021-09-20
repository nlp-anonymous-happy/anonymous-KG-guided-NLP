from transformers import BertTokenizer, BertConfig, BertForQuestionAnswering, AutoModel
import argparse
import logging

def configure_tokenizer_model_bert(args, logger, is_preprocess=False):
    logger.info("***** Loading tokenizer *****")
    tokenizer = BertTokenizer.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                  do_lower_case=args.do_lower_case)

    # logger.info("Loading configuration from {}".format(args.cache_dir))
    logger.info("***** Loading configuration from {} ******".format(args.init_dir))
    config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.init_dir)
    config.vocab_size = len(tokenizer.vocab)

    logger.info("***** Loading pretrained model from {} *****".format(args.init_dir))
    if is_preprocess:
        model = AutoModel.from_pretrained(args.model_name_or_path,
                                  config=config,
                                  cache_dir=args.init_dir)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.init_dir,
                                                         config=config,
                                                         cache_dir=args.init_dir)

    return tokenizer, model