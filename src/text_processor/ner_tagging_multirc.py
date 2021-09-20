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

# This script perform NER tagging for raw SQuAD datasets
# All the named entites found in question and context are recorded with their offsets in the output file
# CoreNLP is used for NER tagging

import os
import json
import argparse
import logging
import urllib
import sys
from tqdm import tqdm
from pycorenlp import StanfordCoreNLP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory to store tagging results.")
    parser.add_argument("--train_file", default='../../data/SQuAD/train-v1.1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='../../data/SQuAD/dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    return parser.parse_args()


# transform corenlp tagging output into entity list
# some questions begins with whitespaces and they are striped by corenlp, thus begin offset should be added.
def parse_output(text, tagging_output, begin_offset=0):
    entities = []
    select_states = ['ORGANIZATION', 'PERSON', 'MISC', 'LOCATION']
    for sent in tagging_output['sentences']:
        state = 'O'
        start_pos, end_pos = -1, -1
        for token in sent['tokens']:
            tag = token['ner']
            if tag == 'O' and state != 'O':
                if state in select_states:
                    entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos],
                                     'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
                state = 'O'
            elif tag != 'O':
                if state == tag:
                    end_pos = token['characterOffsetEnd']
                else:
                    if state in select_states:
                        entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos],
                                         'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
                    state = tag
                    start_pos = token['characterOffsetBegin']
                    end_pos = token['characterOffsetEnd']
        if state in select_states:
            entities.append(
                {'text': text[begin_offset + start_pos: begin_offset + end_pos], 'start': begin_offset + start_pos,
                 'end': begin_offset + end_pos - 1})
    return entities


def tagging(dataset, nlp, output_path):
    all_data = []
    skip_context_cnt, skip_question_cnt, skip_answer_cnt = 0, 0, 0
    for line in dataset:
        paragraph = line["passage"]
        context = paragraph['text']
        context_tagging_output = nlp.annotate(urllib.parse.quote(context),
                                              properties={'annotators': 'ner', 'outputFormat': 'json'})
        # assert the context length is not changed
        if len(context.strip()) == context_tagging_output['sentences'][-1]['tokens'][-1]['characterOffsetEnd']:
            context_entities = parse_output(context, context_tagging_output, len(context) - len(context.lstrip()))
        else:
            context_entities = []
            skip_context_cnt += 1
            logger.info('Skipped context due to offset mismatch:')
            logger.info(context)
        test_right_ner(context_entities, context)

        paragraph['context_entities'] = context_entities

        for qa in paragraph['questions']:
            question = qa['question']
            question_tagging_output = nlp.annotate(urllib.parse.quote(question),
                                                   properties={'annotators': 'ner', 'outputFormat': 'json'})
            if len(question.strip()) == question_tagging_output['sentences'][-1]['tokens'][-1][
                'characterOffsetEnd']:
                question_entities = parse_output(question, question_tagging_output,
                                                 len(question) - len(question.lstrip()))
            else:
                question_entities = []
                skip_question_cnt += 1
                logger.info('Skipped question due to offset mismatch:')
                logger.info(question)
            test_right_ner(question_entities, question)

            qa['question_entities'] = question_entities

            for answer_dict in qa["answers"]:
                answer = answer_dict["text"]
                answer_tagging_output = nlp.annotate(urllib.parse.quote(answer),
                                                       properties={'annotators': 'ner', 'outputFormat': 'json'})

                if not answer_tagging_output['sentences']:
                    logger.info("answer text is empty")
                    answer_entities = []
                else:
                    if len(answer.strip()) == answer_tagging_output['sentences'][-1]['tokens'][-1][
                        'characterOffsetEnd']:
                        answer_entities = parse_output(answer, answer_tagging_output,
                                                         len(answer) - len(answer.lstrip()))
                    else:
                        answer_entities = []
                        skip_answer_cnt += 1
                        logger.info('Skipped answer due to offset mismatch:')
                        logger.info(answer)
                test_right_ner(answer_entities, answer)

                answer_dict['answer_entities'] = answer_entities
        all_data.append(json.dumps(line))

    output_data_lines = "\n".join(all_data)
    with open(output_path, "w", encoding='utf-8') as writer:
        writer.write(output_data_lines)
        writer.write("\n")

    logger.info('In total, {} contexts and {} questions and {} answers are skipped...'.format(skip_context_cnt, skip_question_cnt, skip_answer_cnt))


def read_json_lines(path, mode="r", encoding="utf-8", **kwargs):
    with open(path, mode=mode, encoding=encoding, **kwargs) as f:
        for line in f.readlines():
            yield json.loads(line)

def test_right_ner(entity_list, og_text):
    for tt in entity_list:
        if og_text[tt["start"]:tt["end"] + 1] != tt["text"]:
            logger.warning("bad")
            logger.warning(tt["text"])
            logger.warning(og_text[tt["start"]:tt["end"]])
            logger.warning(tt)

if __name__ == '__main__':
    args = parse_args()

    args.output_dir = "../../data/multirc/"
    args.train_file = "../../data/multirc/train.jsonl"
    args.predict_file = "../../data/multirc/val.jsonl"
    args.test_file = "../../data/multirc/test.jsonl"

    # make output directory if not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # register corenlp server
    nlp = StanfordCoreNLP('http://localhost:9753')

    # load train and dev datasets
    trainset = read_json_lines(args.train_file)
    devset = read_json_lines(args.predict_file)
    # ftrain = open(args.train_file, 'r', encoding='utf-8')
    # trainset = json.load(ftrain)
    # fdev = open(args.predict_file, 'r', encoding='utf-8')
    # devset = json.load(fdev)

    # for dataset, path, name in zip((trainset, devset), (args.train_file, args.predict_file), ('train', 'dev')):
    #     output_path = os.path.join(args.output_dir, "{}.tagged.jsonl".format(os.path.basename(path)[:-6]))
    #     tagging(dataset, nlp, output_path)
    #     # output_path = os.path.join(args.output_dir, "{}.tagged.jsonl".format(os.path.basename(path)[:-6]))
    #     # json.dumps(dataset, open(output_path, 'w', encoding='utf-8'))
    #     logger.info('Finished tagging {} set'.format(name))

    dataset = read_json_lines(args.test_file)
    path = args.test_file
    name = "test"

    output_path = os.path.join(args.output_dir, "{}.tagged.jsonl".format(os.path.basename(path)[:-6]))
    tagging(dataset, nlp, output_path)
    # output_path = os.path.join(args.output_dir, "{}.tagged.jsonl".format(os.path.basename(path)[:-6]))
    # json.dumps(dataset, open(output_path, 'w', encoding='utf-8'))
    logger.info('Finished tagging {} set'.format(name))