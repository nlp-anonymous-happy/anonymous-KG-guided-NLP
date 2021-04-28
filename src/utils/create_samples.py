import json
import random


def create_samples(mode, sample_numbers):
    input_file = "../../data/record/{}.json".format(mode)
    output_file = "../../data/record/{}_{}.json".format(mode, "0831")
    with open(input_file, "r") as reader:
        all_items = json.load(reader)
        all_items = all_items["data"]

    all_samples = random.sample(all_items, sample_numbers)
    all_samples_json = {"data": all_samples}
    with open(output_file, "w") as writer:
        json.dump(all_samples_json, writer, indent=4, separators=(',', ':'))

create_samples("train", 100)
create_samples("dev", 30)