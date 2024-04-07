# Create environment
```
conda create -n "llm-tune" python==3.10
conda activate llm-tune
```
# Installation  
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && \
conda clean -ya && \
conda install -c "nvidia/label/cuda-12.1.1" cuda-nvcc && conda clean -ya
pip install -r requirements.txt
python -m nltk.downloader
pip install flash-attn
pip install deepspeed
pip cache purge
pip install wandb
```
# Fine-tune
```
bash llm-tune/custom_train/src/sft/config/accelerate_train_scripts.sh
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