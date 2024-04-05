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
bash custom_train/dev/script/train_script_zero3.sh
```