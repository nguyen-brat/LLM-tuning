# Create environment
```
conda create -n "llm-tune" python==3.10
conda activate llm-tune
```
# Installation
Install cuda 12.1 or 12.2 and nvcc [Here](https://www.cherryservers.com/blog/install-cuda-ubuntu) if you has the different version please uninstall by `sudo apt-get purge *nvidia*` then reinstall it again in the link above
```
git clone https://github.com/nguyen-brat/LLM-tuning.git
cd LLM-tuning
conda install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers && \
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt
pip cache purge
```
# Fine-tune
Using deepspeed (Faster, offload less to cpu, probaly same ram saved with fsdp)
```
bash sft/config/deepspeed_accelerate.sh
```
Using fsdp (slower, offload more ram on cpu, probaly save more vram ???)
```
bash sft/config/fsdp_accelerate.sh
```
The script support all training argument of huggingface trainer and some custom argument. Please read code in file `sft/param.py` too see all option
# Help
This source code is suitable for training a large LLM on limitation GPU resource. It support the following feature:
- Lora / Qlora (specify `--use-peft` option to use. Default True. [Paper](https://arxiv.org/pdf/2106.09685.pdf))
- DeepSpeed (Specify `--deepspeed` option to use, pass the deepspeed config. [Paper](https://arxiv.org/pdf/2207.00032.pdf))
- gradient cache (Specify `--gradient-checkpoint`. Default True. [Paper](https://arxiv.org/pdf/1808.00079.pdf))
- flash attention 2 (Specify `use_flash_attention_2`. Default True. [Paper](https://arxiv.org/pdf/2307.08691.pdf))
- sparse attention (modify the deepspeed config to use this feature. Read the document [here](https://www.deepspeed.ai/tutorials/sparse-attention/))

You should read the parameter in the sft/param.py file to see all option you can tune with.\
If you want to train a really large model to yout gpu you should you DeepSpeed zero-3 otherwise using zero-2 is faster.\
This source code support 2 type of training include. Change the `--train_type` in the script to choose the type you want. There are two option :
- LLM instruction tuning (only support 1 dialog now)
- LLM unspervise tuning

# Data file format
## CSV
```
Column1         Column2         ...
data            data            ...
```
## Json
```
[
    {"Column 1": value11, "Column 2": value12, ...},
    {"Column 1": value21, "Column 2": value22, ...},
    .
    .
    {"Column 1": valuenn1, "Column 2": valuen2, ...}
]
```