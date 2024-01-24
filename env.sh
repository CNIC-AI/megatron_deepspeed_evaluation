module list

PWD=$(cd "$(dirname "$0")" && pwd)

source ~/anaconda3/etc/profile.d/conda.sh
# conda activate torch113_eval
conda activate torch113_fix_transformers

# path to scripts or resources
export MEGATRON_DIR=/public/home/liufang395/eval/Megatron-DeepSpeed-nvidia

CHECKPOINT_PATH=$MEGATRON_DIR/workspace/llama/checkpoints/global_step1

MEGATRON_MODEL_PATH=./model0 # conversion output
HF_MODEL_PATH=./model
