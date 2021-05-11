import streamlit as st
from neo4j import GraphDatabase
from transformers import BertTokenizer
import pandas as pd
import sys
import random
import time


if len(sys.argv) < 4:
    st.error(f'Usage: streamlit run {sys.argv[0]} <neo4j_host> <user> <password>')
    exit(-1)

host = sys.argv[1] #'46.101.246.118'
user = sys.argv[2] #'neo4j'
password = sys.argv[3]
uri = f'bolt://{host}:7687'


# @st.cache
def neo4j_driver(uri=uri, user=user, password=password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


driver = neo4j_driver()
db = driver.session()


@st.cache
def get_users():

    users = {}
    result = db.run("MATCH (u: User) RETURN u LIMIT 20")
    for idx, u in enumerate(result):
        node = u.values()[0]
        id = node['id']
        users[id] = node
    return users


# @st.cache
def get_tweets(limit=10, skip=None):
    if skip is None:
        skip = random.randint(0, 1000)
    tweets = []
    # extra = ''
    extra = "AND t.language in ['488B32D24BD4BB44172EB981C1BCA6FA', 'B0FA488F2911701DD8EC5B1EA5E322D8', 'B8B04128918BBF54E2E178BFF1ABA833']" # english, spanish, portuguese
    result = db.run(f"MATCH (t: Tweet) WHERE EXISTS(t.text_tokens) {extra} RETURN t SKIP {skip} LIMIT {limit}")
    for idx, t in enumerate(result):
        tweet = {
                    'hashtags': 0, #[],
                    'present_media': [],
                    'present_domains': [],
                    'present_links': [],
                    'timestamp': 0
                 }

        for k, v in t.values()[0].items():
            tweet[k] = v
            if k in ['hashtags']:
                tweet[k] = len(v)

        text_tokens = tweet['text_tokens']
        # del tweet['text_tokens']
        tweet['text'] = tokenizer.decode(text_tokens).replace('[CLS] ', '').replace('[SEP]', '')
        tweets.append(tweet)
    return tweets


def arr_to_df(arr):
    dic = dict()
    for elem in arr:
        for k, v in elem.items():
            if not k in dic.keys():
                dic[k] = []
            dic[k].append(v)
    df = pd.DataFrame(dic)
    return df


users = get_users()

user_id = st.sidebar.selectbox("Choose user:", list(users.keys()))
mode = st.sidebar.radio("Choose recommendation type:", ['reply', 'retweet', 'quote', 'fav'])

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

st.markdown('# User details:')
user = users[user_id]
for k, v in user.items():
    if k == 'account_creation':
        v = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v))
    st.markdown(f'{k}:\t **{v}**')

st.markdown(f'# Predicted ***{mode}*** for tweets:')

tweets = get_tweets()
df = arr_to_df(tweets)
table = st.table(df[['tweet_type',
                     #'hashtags',
                     # 'language',
                     'text']])

driver.close()

