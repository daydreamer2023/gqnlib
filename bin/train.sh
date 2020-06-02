
# Run training

# Settings
MODEL=gqn
CUDA=0,1
SEED=0
STEPS=2000000
TEST_INTERVAL=200000
DATASET=shepard_metzler_5_parts

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${MODEL}

# Dataset path
export DATASET_DIR=./data/
export DATASET_NAME=${DATASET}_torch

# Config for training
export CONFIG_PATH=./examples/config.json

python3 ./examples/train.py --cuda ${CUDA} --model ${MODEL} --seed ${SEED} \
    --steps ${STEPS} --test-interval ${TEST_INTERVAL}
