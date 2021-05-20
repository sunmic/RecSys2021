
#!/bin/bash
#
# Create database using neo4j-admin import. It's preferred way of data import, 
# as it's much faster than load_partial_data.cypher Cypher script.
# Neo4j database should not be running on host machine.
# Neo4j data dir ($NEO4J_HOME/data) should be empty (script will fail if there is existing database)
#

if [ $# -lt 2 ]; then
    echo "Usage: $0 file_prefix dir"
    exit 1 
fi 

set -e

PREFIX_DIR="$2/$1"
NODE1_FILES="${PREFIX_DIR}-user.csv"
NODE2_FILES="${PREFIX_DIR}-tweet.csv"
RELATIONSHIP1_FILES="${PREFIX_DIR}-author.csv"
RELATIONSHIP2_FILES="${PREFIX_DIR}-follow.csv"
RELATIONSHIP3_FILES="${PREFIX_DIR}-like.csv"
RELATIONSHIP4_FILES="${PREFIX_DIR}-reply.csv"
RELATIONSHIP5_FILES="${PREFIX_DIR}-retweet.csv"
RELATIONSHIP6_FILES="${PREFIX_DIR}-rtcomment.csv"
RELATIONSHIP7_FILES="${PREFIX_DIR}-seen.csv"

neo4j-admin import --skip-bad-relationships=true \
                   --trim-strings=true \
                   --high-io=true \
                   --ignore-empty-strings=true \
                   --id-type=STRING \
                   --verbose \
                   --nodes $NODE1_FILES \
                   --nodes $NODE2_FILES \
                   --relationships $RELATIONSHIP1_FILES \
                   --relationships $RELATIONSHIP2_FILES \
                   --relationships $RELATIONSHIP3_FILES \
                   --relationships $RELATIONSHIP4_FILES \
                   --relationships $RELATIONSHIP5_FILES \
                   --relationships $RELATIONSHIP6_FILES \
                   --relationships $RELATIONSHIP7_FILES
