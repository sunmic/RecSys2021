#!/usr/bin/env python3
#
# Convert LZO file to set of csv, ready for neo4j import
#

import subprocess
import csv
import sys
import os

from tqdm import tqdm

tweet_features = [
    "text_tokens", "hashtags", 
    "tweet_id", "present_media", 
    "present_links", "present_domains", 
    "tweet_type", "language", 
    "tweet_timestamp",
]
tweet_author_features = tweet_features + ["author_id"]
user_features = [
    "user_id", "follower_count", "following_count", "is_verified", "account_creation"
] # X 2 for engaged with and engaging
relationships_features = [
    "engagee_follows_engager", "reply_timestamp", 
    "retweet_timestamp", "retweet_with_comment_timestamp", 
    "like_timestamp"
]
user_tweet_features = [
    "user_id", "tweet_id", "timestamp"
]
user_user_features = [
    "user1_id", "user2_id"
]

if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} input.lzo output user_ids tweet_ids")
    exit(1)

input_lzo = sys.argv[1]
output_fp = sys.argv[2]
user_ids_fp = sys.argv[3]
tweet_ids_fp = sys.argv[4]
tweet_data_fp = output_fp + "tweet.csv"
user_data_fp = output_fp + "user.csv"
follow_fp = output_fp + "follow.csv"
reply_fp = output_fp + "reply.csv"
retweet_fp = output_fp + "retweet.csv"
like_fp = output_fp + "like.csv"
comment_fp = output_fp + "rtcomment.csv"

user_ids = set()
tweet_ids = set()
if os.path.exists(user_ids_fp):
    print("Reading existing user ids...")
    user_ids_f = open(user_ids_fp, "r")
    for line in user_ids_f.readlines():
        user_ids.add(line.strip())
    user_ids_f.close()
    print(f"Existing ids: {len(user_ids)} user")
if os.path.exists(tweet_ids_fp):
    print("Reading existing tweet ids...")
    tweet_ids_f = open(tweet_ids_fp, "r")
    for line in tweet_ids_f.readlines():
        tweet_ids.add(line.strip())
    tweet_ids_f.close()
    print(f"Existing ids: {len(tweet_ids)} tweet")
new_user_ids = set()
new_tweet_ids = set()

with open(tweet_data_fp, "w") as tweet_f, \
     open(user_data_fp, "w") as user_f, \
     open(follow_fp, "w") as follow_f, \
     open(reply_fp, "w") as reply_f, \
     open(retweet_fp, "w") as retweet_f, \
     open(like_fp, "w") as like_f, \
     open(comment_fp, "w") as comment_f:
    # csv writer objects
    tweet_writer = csv.writer(tweet_f, delimiter=";")
    user_writer = csv.writer(user_f, delimiter=";")
    follows_writer = csv.writer(follow_f, delimiter=";")
    replies_writer = csv.writer(reply_f, delimiter=";")
    retweet_writer = csv.writer(retweet_f, delimiter=";")
    like_writer = csv.writer(like_f, delimiter=";")
    comment_writer = csv.writer(comment_f, delimiter=";")
    # csv headers 
    tweet_writer.writerow(tweet_author_features)
    user_writer.writerow(user_features)
    follows_writer.writerow(user_user_features)
    replies_writer.writerow(user_tweet_features)
    retweet_writer.writerow(user_tweet_features)
    like_writer.writerow(user_tweet_features)
    comment_writer.writerow(user_tweet_features)

    print(f"Decompressing {input_lzo}...")
    proc = subprocess.Popen(
        ['lzop', '-dc', input_lzo],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    pbar = tqdm(desc="Reading rows", postfix="rows read/written")
    for line in iter(proc.stdout.readline, b''):
        # decode
        line_utf8 = line.decode('utf-8').strip()
        cols = line_utf8.split("\x01")
        # write to csv files
        pos = 0
        tweet_id = cols[pos + 2]
        tweet_cols = cols[pos:pos+len(tweet_features)]
        pos = pos + len(tweet_features)

        engaged_with_id = cols[pos]
        if engaged_with_id not in user_ids:
            user_writer.writerow(cols[pos:pos+len(user_features)])
            user_ids.add(engaged_with_id)
            new_user_ids.add(engaged_with_id)
        pos = pos + len(user_features)

        engaging_id = cols[pos]
        if engaging_id not in user_ids:
            user_writer.writerow(cols[pos:pos+len(user_features)])
            user_ids.add(engaging_id)
            new_user_ids.add(engaging_id)
        pos = pos + len(user_features)

        if tweet_id not in tweet_ids:
            tweet_writer.writerow(tweet_cols + [engaged_with_id])
            tweet_ids.add(tweet_id)
            new_tweet_ids.add(tweet_id)

        follows, reply, retweet, comment, like = cols[pos:pos+len(relationships_features)]
        if follows == 'true':
            follows_writer.writerow([engaging_id, engaged_with_id])
        if len(reply) != 0:
            replies_writer.writerow([engaging_id, tweet_id, reply])
        if len(retweet) != 0:
            retweet_writer.writerow([engaging_id, tweet_id, retweet])
        if len(comment) != 0:
            comment_writer.writerow([engaging_id, tweet_id, comment])
        if len(like) != 0:
            like_writer.writerow([engaging_id, tweet_id, like])
        # update progress
        pbar.update(1)
    pbar.close()
    proc.kill()
print(f"Writing new ids: {len(new_user_ids)} new user, {len(new_tweet_ids)} new tweet")
with open(user_ids_fp, "a") as user_ids_f, open(tweet_ids_fp, "a") as tweet_ids_f:
    for uid in new_user_ids:
        user_ids_f.write(f"{uid}\n")
    for tid in new_tweet_ids:
        tweet_ids_f.write(f"{tid}\n")
print("Done!")
        