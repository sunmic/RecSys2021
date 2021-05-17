
#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 file_prefix dir"
    exit 1 
fi 

set -e

PREFIX_DIR="$2/$1"
NODE_FILES="${PREFIX_DIR}-user.csv,${PREFIX_DIR}-tweet.csv"
RELATIONSHIP_FILES="${PREFIX_DIR}-author.csv,${PREFIX_DIR}-follow.csv,${PREFIX_DIR}-like.csv,${PREFIX_DIR}-reply.csv,${PREFIX_DIR}-retweet.csv,${PREFIX_DIR}-rtcomment.csv"

neo4j-admin import --skip-duplicate-nodes=true \
                   --trim-strings=true \
                   --high-io=true \
                   --ignore-empty-strings=true \
                   --id-type=STRING \
                   --nodes $NODE_FILES \
                   --relationships $RELATIONSHIP_FILES