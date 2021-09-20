import sys
import subprocess


def main():
    mark = "multirc_kelm_seed12_test_best_total"
    output_dir = "../outputs/{}_kelm_both_test/".format(mark)
    deepspeed_run = [
        # new second
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        "1",
        "--master_port",
        "2256",
        "./run_multirc_test.py",
        "--mark",
        mark,
        "--init_dir",
        # "../cache/bert-large-cased/",
        "../checkpoint/best_seed12",
        "--data_dir",
        "../data",
        "--cache_dir",
        "../cache/",
        "--cache_store_dir",
        "../cache/",
        "--config_name",
        "../cache/bert-large-cased/",
        "--model_name_or_path",
        "../cache/bert-large-cased/",
        "--tokenizer_path",
        "../cache/bert-large-cased/",
        "--wn18_dir",
        "../data/kgs/wn18/text/",
        "--output_dir",
        output_dir,
        "--cache_file_suffix",
        "multirc_kelm_three_relation_filter_and_clean",
        "--max_seq_length",
        "256",
        "--relation_agg",
        "sum",
        "--per_gpu_train_batch_size",
        "128",
        "--per_gpu_eval_batch_size",
        "128",
        "--gradient_accumulation_steps",
        "1",
        "--num_train_epochs",
        "3",
        "--evaluate_epoch",
        "0.5",
        "--do_eval",
        "false",
        "--evaluate_during_training",
        "true",
        "--model_type",
        "kelm",
        "--text_embed_model",
        "bert",
        "--seed",
        "12",
        "--freeze",
        "true",
        "--save_model",
        "true",
        "--data_preprocess",
        "true",
        "--data_preprocess_evaluate",
        "true",
        "--is_clean",
        "True",
        "--is_filter",
        "True",
        "--is_morphy",
        "False",
        "--chunksize",
        "40",
        "--threads",
        "40",
        "--is_all_relation",
        "false",
        "--learning_rate",
        "1e-5",
        "--train_file",
        "multirc/train.tagged.jsonl",
        "--predict_file",
        "multirc/test.tagged.jsonl",
        "--full_table",
        "False",
        "--use_context_graph",
        "false",
        "--use_wn",
        "true",
        "--use_nell",
        "true",
        "--test",
        "true",
        "--write_preds",
        "true",
    ]
    cmd = deepspeed_run
    result = subprocess.Popen(cmd)
    result.wait()


if __name__ == '__main__':
    main()