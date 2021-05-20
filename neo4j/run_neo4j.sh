#!/bin/bash
# 
# Run neo4j using docker
#

if [ $# -lt 1 ]; then
	echo "Usage: $0 neo4j_homedir"
	exit 1
fi 

NEO4J_HOME="$1"

sudo docker pull neo4j:4.2.6

sudo docker run \
		--publish=7474:7474 \
		--publish=7687:7687 \
		--volume="$NEO4J_HOME/data:/data" \
		--volume="$NEO4J_HOME/plugins:/var/lib/neo4j/plugins" \
		--volume="$NEO4J_HOME/conf:/var/lib/neo4j/conf" \
		--volume="$NEO4J_HOME/import:/var/lib/neo4j/import" \
		--env NEO4J_dbms_memory_pagecache_size=8G \
		--restart always \
		--detach \
		neo4j:4.2.6
