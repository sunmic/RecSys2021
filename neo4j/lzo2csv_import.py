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
user_features = [
    "user_id", "follower_count", "following_count", "is_verified", "account_creation"
] # X 2 for engaged with and engaging
relationships_features = [
    "engagee_follows_engager", "reply_timestamp", 
    "retweet_timestamp", "retweet_with_comment_timestamp", 
    "like_timestamp"
]

user_cols = [
    "id:ID(User-ID)", "is_verified:boolean", "account_creation:int", 
    "follower_count:int", "following_count:int", ":LABEL"
]
tweet_cols = [
    "id:ID(Tweet-ID)", "text_tokens:int[]", "hashtags:string[]",
    "present_media:string[]", "present_links:string[]", "present_domains:string[]",
    "tweet_type:string", "language:string", "timestamp:int" , ":LABEL"
]
author_cols = [
    ":START_ID(User-ID)", ":END_ID(Tweet-ID)", ":TYPE" 
]
follow_cols = [
    ":START_ID(User-ID)", ":END_ID(User-ID)", ":TYPE" 
]
engagement_cols = [
    ":START_ID(User-ID)", "timestamp:int", ":END_ID(Tweet-ID)", ":TYPE"
]

ARRAY_DELIMITER=","

def tweet_to_csv(tweet_values):
    text_tokens = ARRAY_DELIMITER.join(tweet_values[0].split("\t"))
    hashtags = ARRAY_DELIMITER.join(tweet_values[1].split("\t"))
    tweet_id = ARRAY_DELIMITER.join(tweet_values[2].split("\t"))
    present_media = ARRAY_DELIMITER.join(tweet_values[3].split("\t"))
    present_links = ARRAY_DELIMITER.join(tweet_values[4].split("\t"))
    present_domains = ARRAY_DELIMITER.join(tweet_values[5].split("\t"))
    suffix = tweet_values[6:]
    return [tweet_id, text_tokens, hashtags, present_media, present_links, present_domains] + suffix + ["Tweet"]

def user_to_csv(user_values):
    user_id = user_values[0]
    user_suffix = user_values[1:3]
    user_prefix = user_values[3:]
    return [user_id] + user_prefix + user_suffix  + ["User"]

def author_rel_row(user, tweet):
    return [user, tweet, "Author"]
def follow_rel_row(user1, user2):
    return [user1, user2, "Follow"]
def like_rel_row(user, tweet, timestamp):
    return [user, int(timestamp), tweet, "Like"]
def retweet_rel_row(user, tweet, timestamp):
    return [user, int(timestamp), tweet, "Retweet"]
def reply_rel_row(user, tweet, timestamp):
    return [user, int(timestamp), tweet, "Reply"]
def rtcomment_rel_row(user, tweet, timestamp):
    return [user, int(timestamp), tweet, "RetweetComment"]

if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} input.lzo output user_ids tweet_ids [mode]")
    exit(1)

input_lzo = sys.argv[1]
output_fp = sys.argv[2]
user_ids_fp = sys.argv[3]
tweet_ids_fp = sys.argv[4]
if len(sys.argv) > 5:
    mode = sys.argv[5]
    if mode not in ['a', 'w']:
        raise ValueError("Invalid mode. Supported modes: a (append), w (write)")
else:
    mode = 'w'

tweet_data_fp = output_fp + "tweet.csv"
user_data_fp = output_fp + "user.csv"
author_fp = output_fp + "author.csv"
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

with open(tweet_data_fp, mode) as tweet_f, \
     open(user_data_fp, mode) as user_f, \
     open(follow_fp, mode) as follow_f, \
     open(reply_fp, mode) as reply_f, \
     open(retweet_fp, mode) as retweet_f, \
     open(like_fp, mode) as like_f, \
     open(comment_fp, mode) as comment_f, \
     open(author_fp, mode) as author_f:
    
    # csv writer objects
    tweet_writer = csv.writer(tweet_f, delimiter=";")
    user_writer = csv.writer(user_f, delimiter=";")
    author_writer = csv.writer(author_f, delimiter=";")
    follows_writer = csv.writer(follow_f, delimiter=";")
    replies_writer = csv.writer(reply_f, delimiter=";")
    retweet_writer = csv.writer(retweet_f, delimiter=";")
    like_writer = csv.writer(like_f, delimiter=";")
    comment_writer = csv.writer(comment_f, delimiter=";")
    # csv headers 
    if tweet_f.tell() == 0:
        tweet_writer.writerow(tweet_cols)
    if user_f.tell() == 0:
        user_writer.writerow(user_cols)
    if follow_f.tell() == 0:
        follows_writer.writerow(follow_cols)
    if author_f.tell() == 0:
        author_writer.writerow(author_cols)
    if reply_f.tell() == 0:
        replies_writer.writerow(engagement_cols)
    if retweet_f.tell() == 0:
        retweet_writer.writerow(engagement_cols)
    if like_f.tell() == 0:
        like_writer.writerow(engagement_cols)
    if comment_f.tell() == 0:
        comment_writer.writerow(engagement_cols)

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
        tweet_col_values = cols[pos:pos+len(tweet_features)]
        if tweet_id not in tweet_ids:
            tweet_writer.writerow(tweet_to_csv(tweet_col_values))
        pos = pos + len(tweet_features)

        engaged_with_id = cols[pos]
        engaged_with_values = cols[pos:pos+len(user_features)]
        if engaged_with_id not in user_ids:
            user_writer.writerow(user_to_csv(engaged_with_values))
            user_ids.add(engaged_with_id)
            new_user_ids.add(engaged_with_id)
        pos = pos + len(user_features)

        engaging_id = cols[pos]
        engaging_values = cols[pos:pos+len(user_features)]
        if engaging_id not in user_ids:
            user_writer.writerow(user_to_csv(engaging_values))
            user_ids.add(engaging_id)
            new_user_ids.add(engaging_id)
        pos = pos + len(user_features)

        if tweet_id not in tweet_ids:
            author_writer.writerow(author_rel_row(engaged_with_id, tweet_id))
            tweet_ids.add(tweet_id)
            new_tweet_ids.add(tweet_id)

        follows, reply, retweet, comment, like = [c.strip() for c in cols[pos:pos+len(relationships_features)]]
        if follows == 'true':
            follows_writer.writerow(follow_rel_row(engaging_id, engaged_with_id))
        if len(reply) != 0:
            replies_writer.writerow(reply_rel_row(engaging_id, tweet_id, reply))
        if len(retweet) != 0:
            retweet_writer.writerow(retweet_rel_row(engaging_id, tweet_id, retweet))
        if len(comment) != 0:
            comment_writer.writerow(rtcomment_rel_row(engaging_id, tweet_id, comment))
        if len(like) != 0:
            like_writer.writerow(like_rel_row(engaging_id, tweet_id, like))
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
        