
# Run training

# Settings
MODEL=gqn
DATASET=shepard_metzler_5_parts
CUDA=0,1
STEPS=2000000

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${DATASET}

# Dataset path
export DATASET_DIR=./data/
export DATASET_NAME=${DATASET}_torch

# Config for training
export CONFIG_PATH=./examples/config.json

python3 ./examples/train.py --cuda ${CUDA} --model ${MODEL} --steps ${STEPS} \
    --log-save-interval 1
