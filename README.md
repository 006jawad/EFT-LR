# EFT-LR
EFT-LR is a benchmarking tool for evaluating **learning rate (LR) policies** during LLM fine-tuning. It helps analyze the impact of different LR policies on model performance, providing insights for optimization.  

## Installation  
To use EFT-LR, clone the repository and install the required dependencies:  

```bash
git clone https://github.com/006jawad/EFT-LR
cd EFT-LR
pip install -r requirements.txt
```

## Usage
The training and evaluation workflow is provided in the ```code.ipynb``` file. Open the notebook to experiment with different LR policies and analyze their effects on fine-tuning performance.

Below is a quick reference:
### Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 finetune.py \
  --base_model 'meta-llama/Llama-2-7b-hf' \
  --data_path 'tatsu-lab/alpaca' \
  --output_dir 'path/to/output' \
  --batch_size 128 \
  --num_epochs 5 \
  --learning_rate 1e-3 \
  --learning_rate_policy 'stepLR'
```
### Evaluation
```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained='meta-llama/Llama-2-7b-hf',peft='path/to/checkpoint' \
  --tasks "piqa,openbookqa,social_iqa,mnli,truthfulqa,hellaswag,ai2_arc" \
  --output_path 'path/to/output.json' \
  --log_samples
```
