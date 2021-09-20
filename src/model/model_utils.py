from model.bert import configure_tokenizer_model_bert
from model.roberta import configure_tokenizer_model_roberta
from transformers import BertTokenizer, RobertaTokenizer

def configure_tokenizer_model(args, logger, kgretrievers=None, is_preprocess=False):
    if is_preprocess:
        if args.text_embed_model == "bert":
            tokenizer, model = configure_tokenizer_model_bert(args, logger, is_preprocess)
        elif args.text_embed_model == "roberta":
            tokenizer, model = configure_tokenizer_model_roberta(args, logger, is_preprocess)
    else:
        if args.model_type == "kelm":
            if args.dataset == "record" or "squad":
                from model.kelm_record import configure_tokenizer_model_kelm
            elif args.dataset == "multirc":
                from model.kelm_multirc import configure_tokenizer_model_kelm
            elif  args.dataset == "copa":
                from model.kelm_copa import configure_tokenizer_model_kelm
            else:
                logger.warning("not recongize dataset, program exit")
                exit()

            tokenizer, model = configure_tokenizer_model_kelm(args, logger, kgretrievers)
        elif args.model_type == "roberta":
            tokenizer, model = configure_tokenizer_model_roberta(args, logger)
        else:
            raise NotImplementedError("The {} has not been implemented!".format(args.model_type))
    return tokenizer, model

