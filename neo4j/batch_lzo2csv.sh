#!/bin/bash
#
# Sequentially decompress all *.lzo at <partsdir> files and convert them into multiple GIANT 
# csv files (for each node/relationship type) ready to be used by neo4j-admin import tool.
#

if [ $# -lt 2 ]; then
    echo "Usage: $0 <partsdir> <importdir> [<temp>]"
    exit 1
fi

PARTS_DIR="$1"
IMPORT_DIR="$2"

if [ $# -lt 3 ]; then
    TEMP_DIR=$(mktemp -d)
else
    TEMP_DIR="$3"
fi

LATEST_PREFIX=$(cat .progress)
LATEST_PREFIX_NUM=${LATEST_PREFIX:${#LATEST_PREFIX}-5:5}

set -e

LOCAL_USER_IDS=./user.ids
LOCAL_TWEET_IDS=./tweet.ids
SHARED_MEM_USER_IDS=/dev/shm/user.ids
SHARED_MEM_TWEET_IDS=/dev/shm/tweet.ids
if [ -e $LOCAL_USER_IDS ]; then
    echo "=====> Copying user ids to shared memory..."
    cp $LOCAL_USER_IDS $SHARED_MEM_USER_IDS
else
    touch $SHARED_MEM_USER_IDS
fi

if [ -e $LOCAL_TWEET_IDS ]; then
    echo "=====> Copying tweet ids to shared memory..."
    cp $LOCAL_TWEET_IDS $SHARED_MEM_TWEET_IDS
else
    touch $SHARED_MEM_TWEET_IDS
fi

ITER=0
for PART_FILE in "${PARTS_DIR}"/*.lzo ; do 
    LOCAL_PART_FILE="${TEMP_DIR}/$(basename ${PART_FILE})"
    PREFIX=$(basename "${LOCAL_PART_FILE}" .lzo)

    if [ $LATEST_PREFIX ]; then
        PREFIX_NUM=${PREFIX:${#PREFIX}-5:5}
        if [ $PREFIX_NUM -le $LATEST_PREFIX_NUM ]; then
            echo "Looks like $PREFIX was already read. Skipping..."
            continue
        fi
    fi

    cp "${PART_FILE}" "${LOCAL_PART_FILE}"
    echo "=====> Running csv conversion for $PREFIX"
    python3 lzo2csv_admin_import.py "${LOCAL_PART_FILE}" "${IMPORT_DIR}/part-" "${SHARED_MEM_USER_IDS}" "${SHARED_MEM_TWEET_IDS}" "a"
    echo "=====> Cleanup"
    rm "${LOCAL_PART_FILE}"
    echo "${PREFIX}" >./.progress

    ITER=$((ITER+1))
    if [ $((ITER%5)) -eq 0 ]; then
        echo "=====> Saving ids"
        cp ${SHARED_MEM_USER_IDS} ${LOCAL_USER_IDS}
        cp ${SHARED_MEM_TWEET_IDS} ${LOCAL_TWEET_IDS}
        echo "${PREFIX}" >./.progress-ids
    fi
done