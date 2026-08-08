#!/bin/bash

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE

PYTHON_SCRIPT="Pro_HLS_GPT_application_v4_7.py"

TILE_ID="10TDM"
START_DATE="2023001"
END_DATE="2024001"

HLS_DATA_DIR="/mmfs1/scratch/jacks.local/junjie.li/Foundation model/data/demo/HLS_128_patches"
OUTPUT_DIR="/mmfs1/scratch/jacks.local/junjie.li/Foundation model/data/demo/output"
DEM_DIR="/mmfs1/scratch/jacks.local/junjie.li/Foundation model/data/demo/DEM_128_patches"

RECONSTRUCTED_DATES="2023201,2023227,2023235"

CHUNK_SIZE=128
IMG_WIDTH=128
IMG_HEIGHT=128
BATCH_SIZE=512
IS_EVALUATION=1

python "${PYTHON_SCRIPT}" \
    --tile_id "${TILE_ID}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --hls_data_dir "${HLS_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --chunk_size "${CHUNK_SIZE}" \
    --img_width "${IMG_WIDTH}" \
    --img_height "${IMG_HEIGHT}" \
    --batch_size "${BATCH_SIZE}" \
    --dem_dir "${DEM_DIR}" \
    --reconstructed_dates "${RECONSTRUCTED_DATES}" \
    --is_evaluation "${IS_EVALUATION}"
