export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="train-combine-model"
export HF_TOKEN="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"
DIR=`pwd`

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="Qwen/Qwen1.5-7B-Chat"
DATA="/home/nguyen/code/llm-tune/data/combine_multi_view/train/combine_pedestrian_vid_name.csv"
EVAL_DATA="/home/nguyen/code/llm-tune/data/combine_multi_view/val/combine_pedestrian_vid_name.csv"
DS_PATH="llm-tune/custom_train/src/config/zero3.json"

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $GPUS_PER_NODE
	--pipeline-model-parallel-size $(($GPUS_PER_NODE*$NNODES))
)
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
    --num_train_epochs 20
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 8
    --gradient_checkpointing True
)
DATA_ARGS=(
    --do_train True
    --do_eval True
    --train_file $DATA
    --validation_file $EVAL_DATA
)
EVAL_AND_LOGGING_ARGS=(
    --push_to_hub True
    --hub_private_repo True
    --hub_model_id "nguyen-brat/combine-qwen1.5-7b-bf16-train_v2"
    --run_name="offical-combine-model-training-qwen"\
    --output_dir /home/nguyen/code/llm-tune/model/combine_model_v3
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
    --use_peft True
)
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS /home/nguyen/code/llm-tune/custom_train/dev/fine_tune_dev.py \
    --deepspeed $DS_PATH \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}