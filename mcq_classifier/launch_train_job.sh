#!/bin/bash

# VENV_PATH="/home/nchandak/miniforge3/envs/trainingtt/"

# source "$VENV_PATH/bin/activate"

if [ $# -eq 0 ]; then
    echo "Please provide a Python file name as argument"
    exit 1
fi

# Detect number of GPUs (requires nvidia-smi)
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $N_GPUS GPUs"

export WANDB_API_KEY=""

# source /home/nchandak/miniforge3/bin/activate minir1
source /home/nchandak/qaevals/how-to-qa/qa/bin/activate
module load cuda/12.1

cd /home/nchandak/qaevals/how-to-qa/mcq_classifier
# accelerate launch mmlu_pro_4way.py
# accelerate launch mengye.py
# accelerate launch mmlu_try.py
# accelerate launch mmlu_pro_try.py
# accelerate launch manifold.py
# accelerate launch super_gqpa.py
# accelerate launch mmlu_multi_evals.py
# accelerate launch multi_evals.py
# accelerate launch super_gqpa_4way.py
# bash scripts/check_halawi_train.sh $1 $2

# accelerate launch "$1"
accelerate launch --config_file zero3_config.yaml --num_processes $N_GPUS "$1"
conda deactivate
