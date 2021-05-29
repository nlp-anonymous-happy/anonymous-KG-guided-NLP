from transformers import RobertaTokenizer, RobertaConfig, RobertaForQuestionAnswering, AutoModel

def configure_tokenizer_model_roberta(args, logger, is_preprocess=False):
    logger.info("***** Loading tokenizer *****")
    tokenizer = RobertaTokenizer.from_pretrained(args.config_name,
                                                     do_lower_case=args.do_lower_case,
                                                 cache_dir=args.init_dir,)

    logger.info("***** Loading configuration *****")
    config = RobertaConfig.from_pretrained(args.config_name, cache_dir=args.init_dir)

    logger.info("Loading pretrained model from {}".format(args.init_dir))

    if is_preprocess:
        model = AutoModel.from_pretrained(args.model_name_or_path,
                                  config=config,
                                  cache_dir=args.init_dir)
    else:
        model = RobertaForQuestionAnswering.from_pretrained(args.init_dir, config=config,
                                                            cache_dir=args.init_dir)


    return tokenizer, model