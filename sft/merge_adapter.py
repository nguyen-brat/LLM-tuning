from peft import AutoPeftModelForCausalLM
import torch
import argparse

# load PEFT model in fp16
def main(args):
    if args.bf16:
        torch_type = getattr(torch, "bfloat16")
    else:
        torch_type = getattr(torch, "float16")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch_type,
    )  
    # Merge LoRA and base model and save
    model = model.merge_and_unload()        
    model.save_pretrained(
        args.output_dir, safe_serialization=True, max_shard_size="4GB"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge lora adapter args')
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--bf16", default=True, type=bool)
    args = parser.parse_args()

    main(args)