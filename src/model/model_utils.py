from model.kelm import KELM, configure_tokenizer_model_kelm
from model.bert import configure_tokenizer_model_bert
from transformers import BertForQuestionAnswering, RobertaForQuestionAnswering, BertTokenizer, RobertaTokenizer

def configure_tokenizer_model(args, logger, kgretrievers=None, is_preprocess=False):
    if is_preprocess:
        if args.text_embed_model == "bert":
            tokenizer, model = configure_tokenizer_model_bert(args, logger, is_preprocess)
        elif args.text_embed_model == "roberta":
            tokenizer, model = configure_tokenizer_model_roberta(args, logger, is_preprocess)
        elif args.text_embed_model == "roberta_base":
            tokenizer, model = configure_tokenizer_model_roberta_base(args, logger, is_preprocess)
    else:
        if args.model_type == "kelm":
            tokenizer, model = configure_tokenizer_model_kelm(args, logger, kgretrievers)
        elif args.model_type == "bert":
            tokenizer, model = configure_tokenizer_model_bert(args, logger)
        elif args.model_type == "roberta":
            tokenizer, model = configure_tokenizer_model_roberta(args, logger)
        elif args.text_embed_model == "roberta_base":
            tokenizer, model = configure_tokenizer_model_roberta_base(args, logger)
        else:
            raise NotImplementedError("The {} has not been implemented!".format(args.model_type))
    return tokenizer, model

def load_model_from_checkpoint(args, checkpoint):
    if args.model_type == "bert":
        model = BertForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
        tokenizer = BertTokenizer.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                  do_lower_case=args.do_lower_case, )

    elif args.model_type == "kelm":
        model = KELM.from_pretrained(checkpoint)
        if args.text_embed_model == "bert":
            tokenizer = BertTokenizer.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                      do_lower_case=args.do_lower_case, )
        elif args.text_embed_model == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large",
                                                     do_lower_case=args.do_lower_case, )

    elif args.model_type == "roberta":
        model = RobertaForQuestionAnswering.from_pretrained(checkpoint)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large",
                                                     do_lower_case=args.do_lower_case, )
    else:
        raise ValueError("{} Model has not been saved.".format(args.model_type))
    return model, tokenizer
