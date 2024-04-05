import logging
import os
import sys

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPTQConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
import deepspeed
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

def main():
    parser = HfArgumentParser((LoraArguments, ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        lora_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        lora_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'quantization', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank
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
        "padding_side":"right"
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
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
        max_length = model_args.model_max_length,
        response_template=None,
        instruction_template=None,
        tune_type='intruction-sft'
    )
################################################################ load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            quantization_config=GPTQConfig(
                bits=lora_args.quantization, disable_exllama=True
            ) if lora_args.quantization else None,
            use_flash_attention_2=model_args.use_flash_attention_2,
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
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_args.target_modules,
            )
        model = get_peft_model(model, lora_config)
    
    # if training_args.gradient_checkpointing:
    #     model.enable_input_require_grads()

    if model_args.use_flash_attention_2:
        model.config.use_cache = False
    model.config.use_cache = False

##################################################################################### Trainer
    # Initialize our Trainer
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.05)]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=tokenizer.model_max_length,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        peft_config=lora_config,
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

    # if training_args.push_to_hub:
    #     if PartialState().process_index == 0:
    #         # remove data folder\
    #         logger.info("Pushing model to hub...")
    #         api = HfApi(token=os.environ.get("HF_TOKEN"))
    #         api.create_repo(training_args.hub_model_id, repo_type="model", private=True, exist_ok=True)
    #         api.upload_folder(
    #             folder_path="/home/nguyen/code/llm-tune/model/combine_model",
    #             repo_id=training_args.hub_model_id,
    #             repo_type="model",
    #             token=os.environ.get("HF_TOKEN")
    #         )

if __name__ == "__main__":
    # main()
    pass