#!/bin/bash
# 
# Sequentially convert lzo files from <partsdir> into csv files in <importdir>.
# Then load them into database using load_partial_data.cypher Cypher script.
# Neo4j database is required to be run on host machine.
#

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
    python3 lzo2csv.py "${LOCAL_PART_FILE}" "${TEMP_DIR}/${PREFIX}-" ./user.ids ./tweet.ids
    # sudo is needed - import dir will be owned by neo4j user
    sudo mv "${TEMP_DIR}/${PREFIX}-"*.csv "${IMPORT_DIR}"/
    echo "====> Adding data to neo4j"
    # Sometimes db restarts for unknown reason, or there're connection problems
    set +e
    while true; do
        cypher-shell --username ${DB_USERNAME} --password ${DB_PASSWORD} \
                     --param "fileprefix => \"${PREFIX}\"" \
                     --file cypher/load_partial_data.cypher
        if [ $? -ne 0 ]; then
            echo "Cypher shell non zero status. Retrying in 5 sec..."
            sleep 5
        else
	        break
        fi
    done
    set -e
    echo "====> Cleanup"
    rm "${LOCAL_PART_FILE}"
    # sudo is needed - import dir will be owned by neo4j user
    sudo find "${IMPORT_DIR}" -name "${PREFIX}-*.csv" | sudo xargs -I{} rm {}
    echo "${PREFIX}" >./.progress
done