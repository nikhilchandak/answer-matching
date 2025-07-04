#!/bin/bash
cd /home/nchandak/qaevals/how-to-qa/
source qa/bin/activate
module load cuda/12.1

# export LOGLEVEL=DEBUG
# 'chat_template_kwargs': {'enable_thinking': false}}
lm_eval --model vllm --model_args $1 --apply_chat_template --tasks $2 --batch_size auto --log_samples --output_path $3 ${@:5}
# lm_eval --model hf --model_args $1 --tasks $2 --batch_size auto --log_samples --output_path $3 ${@:5}
deactivate