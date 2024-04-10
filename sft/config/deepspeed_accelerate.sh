#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="data-mining-project"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true
#export HF_TOKEN="" put your HF_TOKEN HERE
DIR=`pwd`

MODEL="vilm/vinallama-7b-chat"
#DATA="/home/nguyen/code/llm-tune/data/combine_multi_view/train/combine_pedestrian_vid_name.csv"
DATA=${DIR}/data/translated_train_cleaned.csv
EVAL_DATA=${DIR}/data/translated_train_cleaned.csv
ACCELERATE_PATH=${DIR}/sft/config/ds_accelerate_config.yaml
HUB_MODEL_ID="nguyen-brat/vinallama-7b-bf16_v1"
WANDBRUNNAME="vinallama-tune"
DEEPSPEED_PATH=${DIR}/sft/config/zero3.json
LOCAL_MODEL_SAVE_DIR=${DIR}/model/vinallama_v1
merge_adapters=false

TRAINING_ARGS=(
    --model_name_or_path $MODEL
    --tokenizer_name $MODEL
    --bf16 true
    --tf32 true
    --model_max_length 2048
    --learning_rate 5e-5
    --weight_decay 0.0
    --adam_beta2 0.95
    --warmup_ratio 0.01
    --lr_scheduler_type "cosine"
    --use_flash_attention_2 true
    --gradient_checkpointing true
    --num_train_epochs 5
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 4
    --train_type "unsupervise-tune"
    --low_cpu_mem_usage False
    --use_peft true
    --target_modules "all-linear"
    --patient 2
    #--deepspeed $DEEPSPEED_PATH
    --save_safetensors true
    --quantization 0
    --overwrite_output_dir true
)
DATA_ARGS=(
    --do_train true
    --do_eval true
    --train_file $DATA
    --validation_file $EVAL_DATA
    --response_template "<|im_start|>assistant"
    --instruction_text_column "question"
    --response_text_column "answer"
)
EVAL_AND_LOGGING_ARGS=(
    --push_to_hub true
    --hub_private_repo true
    --hub_model_id $HUB_MODEL_ID
    --run_name $WANDBRUNNAME
    --output_dir $LOCAL_MODEL_SAVE_DIR
    --evaluation_strategy "epoch"
    --eval_accumulation_steps 10
    #--eval_steps 0.1
    --save_strategy "epoch"
    #--save_steps 0.1
    --save_total_limit 1
    --logging_steps 1
    --load_best_model_at_end true
    --metric_for_best_model "loss"
    --report_to "wandb"
)

accelerate launch --config_file $ACCELERATE_PATH --deepspeed_config_file $DEEPSPEED_PATH ${DIR}/sft/fine_tune.py \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

bf16_value=false
for arg in "${TRAINING_ARGS[@]}"; do
    # Check if the argument starts with "--bf16"
    if [[ $arg == "--bf16"* ]]; then
        # Extract the value of --bf16 by removing "--bf16 " from the argument
        bf16_value="${arg#--bf16}"
        break  # Stop iterating after finding the --bf16 argument
    fi
done

if [ "$merge_adapters" = true ]; then
    python "${DIR}/sft/merge_adapter.py" --output_dir "$LOCAL_MODEL_SAVE_DIR" \
        --bf16 "$bf16_value"
fi