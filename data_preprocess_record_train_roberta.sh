if [ ! -d log ]; then
mkdir log
fi

DATA_DIR=./data/
INIT_DIR=./cache/roberta-large/
CACHE_DIR=./cache/
CACHE_STORE_DIR=./cache/
Model_name_or_path=./cache/roberta-large/
Config_name=./cache/roberta-large/
TOKENIZER_PATH=./cache/roberta-large/
WN18_DIR=./data/kgs/wn18/text/
TRAIN_FILE=record/train.json
PRED_FILE=record/dev.json

mark=preprocess_record_roberta
mark_r=${mark}_train_roberta

OUTPUT_DIR=./outputs/${mark}
Tensorboard_dir=./runs
PWD_DIR=`pwd`

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12498 src/run_record_qa.py \
  --cache_file_suffix record_roberta_cache \
  --relation_agg sum \
  --per_gpu_train_batch_size 12 \
  --per_gpu_eval_batch_size 12 \
  --warmup_steps 0 \
  --max_steps 21000 \
  --mark  $mark_r \
  --tensorboard_dir $Tensorboard_dir \
  --model_type kelm \
  --text_embed_model roberta \
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
  --data_preprocess true \
  --data_preprocess_evaluate false \
  --do_train true \
  --do_eval false \
  --is_clean true \
  --is_filter true \
  --is_morphy false \
  --full_table false \
  --do_lower_case false \
  --seed 45 1>$PWD_DIR/log/kelm.${mark_r} 2>&1
