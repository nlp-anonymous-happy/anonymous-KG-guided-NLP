import collections
import logging
import math
import six

from utils.record_official_evaluate import *
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    # TODO: figureout what is start_index vs start_logit
)

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"]
)


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

# squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
#  squad_evaluate(examples, predictions)
#  predictions = compute_predictions_logits(...)
# def write_predictions(all_examples, all_features, all_results, n_best_size,
#                       max_answer_length, do_lower_case, output_prediction_file,
#                       output_nbest_file, output_null_log_odds_file,
#                       version_2_with_negative, null_score_diff_threshold,
#                       verbose, predict_file, evaluation_result_file):
#  write_predictions(processor.predict_examples, features, all_results,
#                       args.n_best_size, args.max_answer_length,
#                       args.do_lower_case, output_prediction_file,
#                       output_nbest_file, output_null_log_odds_file,
#                       args.version_2_with_negative,
#                       args.null_score_diff_threshold, args.verbose, args.predict_file, output_evaluation_result_file)
# all_results.append(
#                     RawResult(
#                         unique_id=unique_id,
#                         start_logits=start_logits,
#                         end_logits=end_logits))
# def evaluate(dataset, predictions): dataset<->predict_json<->examples.
def record_evaluate(examples, predictions):
    f1 = exact_match = total = 0
    total = len(examples)
    correct_ids = []
    for example in examples:
        if example.id not in predictions:
            logger.warning('Unanswered question {} will receive score 0.'.format(example.id))
            continue

        ground_truths = list(map(lambda x: x[0], example.answer_entities))
        prediction = predictions[example.id]

        _exact_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        if int(_exact_match) == 1:
            correct_ids.append(example.id)
        exact_match += _exact_match

        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    logger.info('* Exact_match: {}\n* F1: {}'.format(exact_match, f1))

    return {'exact_match': exact_match, 'f1': f1}#, correct_ids

def get_wrong_predictions(examples, predictions):
    wrong_records = []
    for example in examples:
        if example.qas_id not in predictions:
            logger.warning('Unanswered question {} will receive score 0.'.format(example['qas_id']))
            continue

        ground_truths = list(map(lambda x: x['text'], example.answers))
        prediction = predictions[example.qas_id]

        _exact_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        if int(_exact_match) != 1:
            wrong_record = dict()
            wrong_record['qas_id'] = example.qas_id
            wrong_record['doc_text'] = " ".join(example.doc_tokens)
            wrong_record['question'] = example.question_text
            wrong_record['ground_truths'] = ground_truths
            wrong_record['prediction'] = prediction
            wrong_records.append(wrong_record)

    return wrong_records


def get_final_text(pred_text, orig_text, verbose, tokenizer):
    #TODO: check this function.
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the ReCoRD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

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

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    # tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

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

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
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

def find_start_end_logit_for_feature(feature, start_indexes, end_indexes, max_answer_length, feature_index, result):
    cur_predictions = []
    valid_token_range = len(feature.tokens)

    for start_index in start_indexes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= valid_token_range: # drop the predicted index pointing to the padding tokens.
            # e.g. max_seq_length = 384. len(tokens) == 200. We don't consider tokens[200:] as the prediction.
            continue
        if start_index not in feature.tokid_span_to_orig_map: # drop the predicted index in the question tokens.
            # TODO: can I remove this part into the _get_best_indexes? At this step to deal with exceptional cases.
            continue

        for end_index in end_indexes:
            if end_index >= valid_token_range:
                continue
            if end_index not in feature.tokid_span_to_orig_map:
                continue
            # if not feature.token_is_max_context.get(start_index, False):
            #     continue # TODO: deal with this token_is_max_context attribute.
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            cur_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index],
                )
            )
    return cur_predictions

def find_nbest_prediction_for_example(prelim_predictions, n_best_size, example, features, tokenizer, verbose_logging=False):#TODO: control the parameter verbose_logging
    # TODO: check this function
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            orig_doc_start = example.doc_tok_to_ori_map[feature.tokid_span_to_orig_map[pred.start_index]]
            orig_doc_end = example.doc_tok_to_ori_map[feature.tokid_span_to_orig_map[pred.end_index]]
            orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, verbose_logging, tokenizer) # TODO: why need tokenizer?
            if final_text in seen_predictions:
                continue
        else:
            final_text = ""
        seen_predictions[""] = True

        nbest.append(_NbestPrediction(
            text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
    return nbest

def evaluate_start_end(start_true, end_true, start_pred, end_pred):
    ignored_index = start_true.size(1)
    start_pred.clamp_(0, ignored_index)
    end_pred.clamp_(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_true, start_pred)
    end_loss = loss_fct(end_true, end_pred)
    total_loss = (start_loss + end_loss) / 2
    return total_loss


def compute_predictions_logits(
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
    with open(predict_file, "r", encoding='utf-8') as reader:
        predict_json = json.load(reader)["data"]
        all_candidates = {}
        for passage in predict_json:
            passage_text = passage['passage']['text']
            candidates = []
            for entity_info in passage['passage']['entities']:
                start_offset = entity_info['start']
                end_offset = entity_info['end']
                candidates.append(passage_text[start_offset: end_offset + 1])
            for qa in passage['qas']:
                all_candidates[qa['id']] = candidates

    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            cur_predictions = find_start_end_logit_for_feature(feature, start_indexes, end_indexes, max_answer_length, feature_index, result)
            prelim_predictions.extend(cur_predictions)

        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        nbest = find_nbest_prediction_for_example(prelim_predictions, n_best_size, example, features, tokenizer, verbose_logging=verbose_logging)
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
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        picked_index = 0
        for pred_index in range(len(nbest_json)):
            if any([f1_score(nbest_json[pred_index]['text'], candidate) > 0. for candidate in
                    all_candidates[example.id]]):
                picked_index = pred_index
                break
        all_predictions[example.id] = nbest_json[picked_index]["text"]
        all_nbest_json[example.id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


