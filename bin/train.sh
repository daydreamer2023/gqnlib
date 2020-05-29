
# Run training

# Kwargs
DATASET=${1:-shepard_metzler_5_parts}
SEED=${2:-0}

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${DATASET}

# Dataset path
export DATASET_DIR=./data/
export DATASET_NAME=${DATASET}_torch

# Config for training
export CONFIG_PATH=./examples/config.json

python3 ./examples/train.py --seed ${SEED} --steps 2000000 \
    --log-save-interval 20
