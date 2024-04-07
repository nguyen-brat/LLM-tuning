export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="train-combine-model"
#export HF_TOKEN="" put your HF_TOKEN HERE
DIR=`pwd`

MODEL="Qwen/Qwen1.5-7B-Chat"
#DATA="/home/nguyen/code/llm-tune/data/combine_multi_view/train/combine_pedestrian_vid_name.csv"
DATA=${DIR}/data/mix_insft_2k.json
EVAL_DATA=${DIR}/data/mix_insft_2k.json
ACCELERATE_PATH=${DIR}/sft/config/accelerate_config.yaml
HUB_MODEL_ID="nguyen-brat/combine-qwen1.5-7b-bf16-train_v2"
WANDBRUNNAME="offical-combine-model-training-qwen"
DEEPSPEED_PATH=${DIR}/sft/config/zero3.json
LOCAL_MODEL_SAVE_DIR=${DIR}/model/combine_model_v3

TRAINING_ARGS=(
    --model_name_or_path $MODEL
    --tokenizer_name $MODEL
    --bf16 True
    --tf32 True
    --torch_dtype bfloat16
    --model_max_length 2048
    --learning_rate 8e-5
    --weight_decay 0.0
    --adam_beta2 0.95
    --warmup_ratio 0.01
    --lr_scheduler_type "cosine"
    --use_flash_attention_2 False
    --gradient_checkpointing True
    --num_train_epochs 1
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 8
    --train_type "unsupervise-tune"
    --use_peft True
    --patient 2
    --deepspeed $DEEPSPEED_PATH
    --save_safetensors True
    --quantization 0
)
DATA_ARGS=(
    --do_train True
    --do_eval True
    --train_file $DATA
    --validation_file $EVAL_DATA
    --response_template "<|im_start|>assistant"
    --instruction_text_column "instruction"
    --response_text_column "output"
)
EVAL_AND_LOGGING_ARGS=(
    --push_to_hub True
    --hub_private_repo True
    --hub_model_id $HUB_MODEL_ID
    --run_name $WANDBRUNNAME
    --output_dir $LOCAL_MODEL_SAVE_DIR
    --evaluation_strategy "steps"
    --eval_accumulation_steps 10
    --eval_steps 5
    --save_strategy "steps"
    --save_steps 10
    --save_total_limit 1
    --logging_steps 1
    --load_best_model_at_end True
    --metric_for_best_model "loss"
    --report_to "wandb"
)
   
accelerate launch --config_file $ACCELERATE_PATH ${DIR}/sft/fine_tune.py \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}