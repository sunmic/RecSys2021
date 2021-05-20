#!/bin/bash
#
# Sequentially uncompressed LZO parts and merge them into single GIANT csv file
# Supports parallelization with max number of processes specified using `num_processes`
#

if [ $# -lt 3 ]; then
    echo "Usage: $0 parts output num_processes"
    exit 1
fi

PARTS_DIR="$1"
OUTPUT_DIR="$2"
NUM_PROCESSES="$3"

set -e

ITER=0
for PART_FILE in "${PARTS_DIR}"/*.lzo ; do
    LOCAL_PART_FILE="${OUTPUT_DIR}/$(basename ${PART_FILE})"
    PREFIX=$(basename "${LOCAL_PART_FILE}" .lzo)

    ((ITER==0)) && echo "Waiting..." && wait && rm -rf "${OUTPUT_DIR}/"*.lzo
    echo "Processing ${PREFIX}..."
    cp "${PART_FILE}" "${LOCAL_PART_FILE}" && lzop -dc "${LOCAL_PART_FILE}" >>"${OUTPUT_DIR}/full-part-${ITER}.csv" &
    ITER=$((ITER + 1))
    ITER=$((ITER % $NUM_PROCESSES))
done

echo "Joining files..."
JOINED_FILE="${OUTPUT_DIR}/part.csv"
touch ${JOINED_FILE}
for N in $(seq 0 $((NUM_PROCESSES-1))); do
    echo "Joining file $N..."
    cat "${OUTPUT_DIR}/full-part-${N}.csv" >>${JOINED_FILE}
    rm "${OUTPUT_DIR}/full-part-${N}.csv"
done
