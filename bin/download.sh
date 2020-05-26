
# Download utility

# Dataset name mast be one of the following
# * jaco
# * mazes
# * rooms_free_camera_with_object_rotations
# * rooms_ring_camera
# * rooms_free_camera_no_object_rotations
# * shepard_metzler_5_parts
# * shepard_metzler_7_parts

export DATA_DIR=./data/
export DATASET_NAME=${1:-shepard_metzler_5_parts}

# Check data dir
if [ ! -d ${DATA_DIR} ]; then
    echo "Make data dir"
    mkdir ${DATA_DIR}
fi

cd ${DATA_DIR}

# Download dataset
if [ ! -d ${DATASET_NAME} ]; then
    echo "Download dataset"
    gsutil cp -r gs://gqn-dataset/${DATASET_NAME}/ .
else
    echo "Specified dataset already exists"
fi
