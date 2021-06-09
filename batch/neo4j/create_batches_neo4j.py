from os import write
from proto.neighbourhood_pb2 import Neighbourhood, Batch

import neo4j
from tqdm import tqdm

import os
import argparse
import hashlib
import math
import gzip


BASE_QUERY = """
    MATCH (u: {label}) WITH u SKIP {offset} LIMIT {limit}
    CALL gds.alpha.randomWalk.stream('{graph_name}', {{ start: id(u), walks: {num_walks}, steps: {num_steps} }})
    YIELD nodeIds
    RETURN id(u) as start, collect(nodeIds) as nodeIds
    """
    

def followers_query(offset, limit, num_walks, num_steps):
    return BASE_QUERY.format(
        label='User', offset=offset, limit=limit, graph_name='users', num_walks=num_walks, num_steps=num_steps
    )


def tweets_query(offset, limit, num_walks, num_steps):
    return BASE_QUERY.format(
        label='Tweet', offset=offset, limit=limit, graph_name='users_tweets', num_walks=num_walks, num_steps=num_steps
    )


def user_query(offset, limit, num_walks, num_steps):
    return BASE_QUERY.format(
        label='Tweet', offset=offset, limit=limit, graph_name='users_tweets', num_walks=num_walks, num_steps=num_steps
    )


def calculate_max_iterations(session, label, offset, limit):
    count_query = f"MATCH (n: {label}) RETURN count(*) as count"
    results = session.run(count_query)
    count = results.value("count")
    return math.ceil((count - offset) / limit)


def write_batch_to_file(batch, offset, prefix, dest_dir):
    filename = f"{prefix}_batch_{offset}.pb"
    filename_hash = hashlib.md5()
    filename_hash.update(filename.encode())
    filename_hash.update(bytes(len(batch.elements)))
    dir_name = filename_hash.hexdigest()[:3]

    os.makedirs(os.path.join(dest_dir, dir_name), exist_ok=True)
    output_fp = os.path.join(dest_dir, dir_name, filename)
    output_f = gzip.open(output_fp, "wb", compresslevel=5)
    output_f.write(batch.SerializeToString())
    output_f.close()


def main(args):
    query_func = None
    label = None
    if args.mode == "followers":
        query_func = followers_query
        label = "User"
    elif args.mode == "tweets":
        query_func = tweets_query
        label = "Tweet"
    elif args.mode == "users":
        query_func = user_query
        label = "User"
    else:
        raise ValueError("Invalid mode!")
    
    uri = f"bolt://{args.neo4j_host}:7687"
    driver = neo4j.GraphDatabase.driver(uri, auth=(args.neo4j_username, args.neo4j_password))
    session = driver.session()

    if args.max_it is None:
        print("Fetching node count...")
        max_it = calculate_max_iterations(session, label, args.offset, args.fetch_size)
        print(f"Remaining number of iterations: {max_it}")
    else:
        max_it = args.max_it
    
    batch = Batch()
    for it in tqdm(range(max_it)):
        offset = args.offset + it * args.fetch_size
        limit = args.fetch_size
        results = session.run(query_func(offset, limit, args.walks, args.steps))

        for record in results:
            source = record["start"]
            source_indices = []
            target_indices = []
            nodes = set()
            for path in record["nodeIds"]:
                for i in range(len(path) - 1):
                    start, end = path[i], path[i + 1]
                    source_indices.append(start)
                    target_indices.append(end)
                    nodes.add(start)
                    nodes.add(end)
            
            neighbourhood = batch.elements.add()
            neighbourhood.start = source
            neighbourhood.nodes.extend(list(nodes))
            neighbourhood.edge_index_source.extend(source_indices)
            neighbourhood.edge_index_target.extend(target_indices)

            if len(batch.elements) >= args.batch_size:
                batch_index = math.ceil(args.offset / args.fetch_size) + it
                print(f"Saving batch with index: {batch_index}, count: {len(batch.elements)}")
                write_batch_to_file(batch, batch_index, args.mode, args.destdir)
                batch.Clear()
    if len(batch.elements) > 0:
        batch_index = math.ceil(args.offset / args.fetch_size) + max_it
        print(f"Saving batch with index: {batch_index}, count: {len(batch.elements)}")
        write_batch_to_file(batch, batch_index, args.mode, args.destdir)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description="Create neighbourhood batch files using neighbourhood sampling with neo4j GDS")
    parser.add_argument("--fetch-size", type=int, default=100, help="NEO4j node fetch size")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of neighbourhoods in single batch file")
    parser.add_argument("--walks", type=int, default=50, help="Number of random walks per node")
    parser.add_argument("--steps", type=int, default=3, help="Max number of steps for each walk")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start from")
    parser.add_argument("--max-it", type=int, default=None, help="Max iterations")
    parser.add_argument("--neo4j-host", default="localhost", help="Neo4j server host")
    parser.add_argument("--neo4j-username", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="", help="Neo4j password")
    parser.add_argument("mode", choices=["followers", "tweets", "users"], help="Neighbourhood type mode")
    parser.add_argument("destdir", help="Destination directory")

    args = parser.parse_args()

    return args

if __name__=='__main__':
    main(parse_args())
