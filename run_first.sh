if [ ! -d log ]; then
mkdir log
fi

DATA_DIR=./data/
INIT_DIR=./cache/bert-large-cased/
CACHE_DIR=./cache/
CACHE_STORE_DIR=./cache/
Model_name_or_path=./cache/bert-large-cased/
Config_name=./cache/bert-large-cased/
TOKENIZER_PATH=./cache/bert-large-cased/
WN18_DIR=./data/kgs/wn18/text/
TRAIN_FILE=record/train.json
PRED_FILE=record/dev.json

mark=new_exp_first
mark_r=${mark}_train_first

OUTPUT_DIR=./outputs/${mark}
Tensorboard_dir=./runs
PWD_DIR=`pwd`

CUDA_VISIBLE_DEVICES=1,2,3,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=83498 src/run_record_qa.py \
  --cache_file_suffix full_test_2 \
  --relation_agg sum \
  --per_gpu_train_batch_size 12 \
  --per_gpu_eval_batch_size 12 \
  --warmup_steps 0 \
  --max_steps 21000 \
  --mark  $mark_r \
  --tensorboard_dir $Tensorboard_dir \
  --model_type kelm \
  --text_embed_model bert \
  --save_steps 2000 \
  --evaluate_steps 2000 \
  --learning_rate 1e-3 \
  --num_train_epochs 10 \
  --threads 50 \
  --is_all_relation false \
  --freeze true \
  --init_dir $INIT_DIR \
  --cache_dir $CACHE_DIR \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --model_name_or_path $Model_name_or_path\
  --config_name $Config_name\
  --wn18_dir $WN18_DIR \
  --tokenizer_path $TOKENIZER_PATH \
  --cache_store_dir $CACHE_STORE_DIR \
  --train_file $TRAIN_FILE \
  --predict_file $PRED_FILE \
  --evaluate_during_training true \
  --data_preprocess false \
  --data_preprocess_evaluate false \
  --do_train true \
  --do_eval false \
  --do_lower_case false \
  --full_table false \
  --use_context_graph false \
  --use_wn true \
  --use_nell true \
  --seed 45 1>$PWD_DIR/log/kelm.${mark_r} 2>&1
