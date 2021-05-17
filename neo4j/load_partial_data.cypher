// CSV Data import Cypher script

// user nodes
USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-user.csv' AS row
FIELDTERMINATOR ';'
CREATE 
(author:User{
    id: row.user_id,
    is_verified: toBoolean(row.is_verified),
    account_creation: toInteger(row.account_creation),
    follower_count: CASE trim(row.follower_count) WHEN "" THEN null ELSE toInteger(row.follower_count) END,
    following_count: CASE trim(row.following_count) WHEN "" THEN null ELSE toInteger(row.following_count) END
})
;

// tweeet nodes
USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-tweet.csv' AS row
FIELDTERMINATOR ';'
CREATE
(tweet: Tweet{
    id: row.tweet_id,
    text_tokens: split(row.text_tokens, "\t"),
    hashtags: split(row.hashtags, "\t"),
    present_media: split(row.present_media, "\t"),
    present_links: split(row.present_links, "\t"),
    present_domains: split(row.present_domains, "\t"),
    tweet_type: row.tweet_type,
    language: row.language,
    timestamp: toInteger(row.tweet_timestamp)
})
;

// id indexes, for speed
CREATE CONSTRAINT users_unique IF NOT EXISTS ON (u:User) ASSERT u.id IS UNIQUE;
CREATE CONSTRAINT tweets_unique IF NOT EXISTS ON (t:Tweet) ASSERT t.id IS UNIQUE;

// author relationship
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-tweet.csv' AS row
FIELDTERMINATOR ';'
MATCH (author: User{ id: row.author_id })
MATCH (tweet: Tweet{ id: row.tweet_id })
CREATE (author)-[r:Author]->(tweet)
;

// retweet relationships
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-retweet.csv' AS row
FIELDTERMINATOR ';'
MATCH (user: User{ id: row.user_id })
MATCH (tweet: Tweet{ id: row.tweet_id })
CREATE (user)-[:Retweet{ timestamp: row.timestamp }]->(tweet)
;

// like relationships
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-like.csv' AS row
FIELDTERMINATOR ';'
MATCH (user: User{ id: row.user_id })
MATCH (tweet: Tweet{ id: row.tweet_id })
CREATE (user)-[:Like{ timestamp: row.timestamp }]->(tweet)
;

// reply relationships
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-reply.csv' AS row
FIELDTERMINATOR ';'
MATCH (user: User{ id: row.user_id })
MATCH (tweet: Tweet{ id: row.tweet_id })
CREATE (user)-[:Reply{ timestamp: row.timestamp }]->(tweet)
;

// retweet+comment relationships
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-rtcomment.csv' AS row
FIELDTERMINATOR ';'
MATCH (user: User{ id: row.user_id })
MATCH (tweet: Tweet{ id: row.tweet_id })
CREATE (user)-[:RetweetComment{ timestamp: row.timestamp }]->(tweet)
;

// follow relationships
USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///' + $fileprefix + '-follow.csv' AS row
FIELDTERMINATOR ';'
MATCH (user1 :User{ id: row.user1_id })
MATCH (user2 :User{ id: row.user2_id })
MERGE (user1)-[:Follow]->(user2)
;