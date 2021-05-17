#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage: $0 <partsdir> <importdir> <username> <password>"
    exit 1
fi

PARTS_DIR="$1"
IMPORT_DIR="$2"
DB_USERNAME="$3"
DB_PASSWORD="$4"

if [ $# -lt 5 ]; then
    TEMP_DIR=$(mktemp -d)
else
    TEMP_DIR="$5"
fi

LATEST_PREFIX=$(cat .progress)
LATEST_PREFIX_NUM=${LATEST_PREFIX:${#LATEST_PREFIX}-5:5}

set -e

for PART_FILE in "${PARTS_DIR}"/*.lzo ; do 
    LOCAL_PART_FILE="${TEMP_DIR}/$(basename ${PART_FILE})"
    PREFIX=$(basename "${PART_FILE}" .lzo)

    PREFIX_NUM=${PREFIX:${#PREFIX}-5:5}
    if [ $PREFIX_NUM -le $LATEST_PREFIX_NUM ]; then
        echo "Looks like $PREFIX was already read. Skipping..."
        continue
    fi

    cp "${PART_FILE}" "${LOCAL_PART_FILE}"
    echo "====> Running csv conversion"
    python3 lzo2csv_import.py "${LOCAL_PART_FILE}" "${IMPORT_DIR}/${PREFIX}-" ./user.ids ./tweet.ids
    echo "====> Adding data to neo4j"
    set +e
    while true; do
        ./import-db.sh "${PREFIX}-" "${IMPORT_DIR}"
        if [ $? -ne 0 ]; then
            echo "import-db non zero status. Retrying in 5 sec..."
            sleep 5
        else
	        break
        fi
    done
    set -e
    echo "====> Cleanup"
    rm "${LOCAL_PART_FILE}"
    sudo rm "${IMPORT_DIR}/${PREFIX}-follow.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-like.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-author.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-reply.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-retweet.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-rtcomment.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-tweet.csv"
    sudo rm "${IMPORT_DIR}/${PREFIX}-user.csv"
    echo "${PREFIX}" >./.progress
done