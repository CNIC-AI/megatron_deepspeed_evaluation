#!/bin/bash
clear

source env.sh

echo $PWD

# model conversion (deepspeed => megatron)
python \
    $MEGATRON_DIR/tools/convert_checkpoint/deepspeed_to_megatron.py \
    --target_tp 1 \
    --target_pp 1 \
    --input_folder $CHECKPOINT_PATH \
    --output_folder $MEGATRON_MODEL_PATH

# model conversion (single megatron file => huggingface)

# https://github.com/alibaba/Megatron-LLaMA/blob/25306de84d300b47a3973cd798463ae7d09019bd/tools/checkpoint_conversion/llama_checkpoint_conversion.py

python llama_checkpoint_conversion.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --load_path $MEGATRON_MODEL_PATH \
    --save_path $HF_MODEL_PATH \
    --target_params_dtype "fp16" \
    --make_vocab_size_divisible_by 1 \
    --print-checkpoint-structure \
    --megatron-path $MEGATRON_DIR >logs/log

# submit inference task
mkdir -p logs

# sbatch run.sh
