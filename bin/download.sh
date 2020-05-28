
# Download utility
# bash bin/downloasd.sh <dataset-name> <batch-size>

# Dataset name mast be one of the following
# * jaco
# * mazes
# * rooms_free_camera_with_object_rotations
# * rooms_ring_camera
# * rooms_free_camera_no_object_rotations
# * shepard_metzler_5_parts
# * shepard_metzler_7_parts

# Kwargs
export DATASET_NAME=${1:-shepard_metzler_5_parts}
export BATCH_SIZE=${2:-64}

# Path
export DATA_DIR=./data/
export DATASET_DIR=${DATA_DIR}/${DATASET_NAME}/

# Check gsutil command
if type gsutil > /dev/null/ 2>&1; then
    echo "gsutil command does exist"
else
    echo "gsutil command does not exist"
    exit 1
fi

# Check data dir
if [[ ! -d ${DATA_DIR} ]]; then
    echo "Make data dir"
    mkdir ${DATA_DIR}
fi

# Download dataset
if [[ ! -d ${DATASET_DIR} ]]; then
    echo "Download dataset"
    gsutil -m cp -r gs://gqn-dataset/${DATASET_NAME}/ ./${DATA_DIR}/
else
    echo "Specified dataset already exists"
fi

# Convert tfrecords to gzip files
python3 ./examples/convert_tfrecord_torch.py --dataset ${DATASET_NAME} \
    --mode train --batch-size ${BATCH_SIZE}

python3 ./examples/convert_tfrecord_torch.py --dataset ${DATASET_NAME} \
    --mode test --batch-size ${BATCH_SIZE}
