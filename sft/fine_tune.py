import logging
import os
import sys
import wandb

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPTQConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.integrations import deepspeed
from trl import SFTTrainer
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

from callback import *
from dataloader import *
from param import *
from log import *
from custom_save import *

logger = logging.getLogger(__name__)
local_rank = None

def main():
    global local_rank
    parser = HfArgumentParser((LoraArguments, ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        lora_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        lora_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'quantization', False):
    #     training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    # local_rank = training_args.local_rank
    # device_map = None
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if lora_args.quantization:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    #     if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
    #         logging.warning(
    #             "FSDP or ZeRO3 are not incompatible with QLoRA."
    #         )
    # Detecting last checkpoint.
    last_checkpoint = train_log(logger, model_args, data_args, training_args)
    set_seed(training_args.seed)

################################################################ load model config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
################################################################## load tokenizer config
    tokenizer_kwargs = {
        "pretrained_model_name_or_path":model_args.model_name_or_path,
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "padding_side":"right",
        "model_max_length":model_args.model_max_length,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model_args.model_max_length:
        tokenizer.model_max_length = config.max_position_embeddings - 100

    data_files = {"train": data_args.train_file}
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    data_loader, data_collator = get_dataloader_collator(
        tokenizer_kwargs=tokenizer_kwargs,
        file_paths = data_files,
        max_length = tokenizer.model_max_length,
        response_template=data_args.response_template,
        instruction_template=data_args.instruction_template,
        instruction_column=data_args.instruction_text_column,
        response_column=data_args.response_text_column,
        tune_type=data_args.train_type,
    )
################################################################ load model
    if model_args.model_name_or_path:
        # torch_dtype = (
        #     model_args.torch_dtype
        #     if model_args.torch_dtype in ["auto", None]
        #     else getattr(torch, model_args.torch_dtype)
        # )
        quantization_config=None
        if lora_args.quantization == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype
            )
        elif lora_args.quantization == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype
            )
        else:
            if lora_args.quantization:
                raise ValueError(f"Not support lora quantization in {lora_args.quantization} bit")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            # device_map=device_map,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else "eager",
        )
    else:
        raise ValueError("You must specify model_name_or_path")
    if lora_args.use_peft:
        logger.info("preparing peft model...")
        if lora_args.quantization:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                target_modules=lora_args.target_modules.split(',') if len(lora_args.target_modules.split(','))>1 else lora_args.target_modules,
            )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
##################################################################################### Trainer
    # Initialize our Trainer
    if training_args.do_eval:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.patient)]
    else:
        callbacks = []
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=tokenizer.model_max_length,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if training_args.do_eval else None,
        peft_config=lora_config,
        packing=True,
        **data_loader,
        #dataset_text_field="text",
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        trainer.save_model()
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
        wandb.finish()

if __name__ == "__main__":
    main()