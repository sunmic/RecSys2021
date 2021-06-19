from GCN.config import POC_SIZE
from torch_geometric.data import InMemoryDataset
from social_neighbourhood_pb2 import SocialNetworkBatch
from torch_geometric.data import Data
import torch
import neo4j
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import torch_delete, reactions, gcn_attributes
import random


class RecSysData(Data):
    def __init__(self):
        super(RecSysData, self).__init__()

    def __inc__(self, key, value):
        if key == 'start_index':
            return self.x_users.size(0)
        if key == 'edge_index':
            return torch.tensor([[self.x_tweets.size(0)], [self.x_users.size(0)]])
        if key.startswith('sn_tweet_index'):
            return self.x_tweets.size(0)
        if key == 'f_edge_index':
            return self.x_users.size(0)
        else:
            return super().__inc__(key, value)


class RecSysBatchDS(InMemoryDataset):
    def __init__(self, root, path, neo4j_pass, transform=None, pre_transform=None, verbose=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.path = path
        self.verbose = verbose
        self.device = device
        self.neo4j_pass = neo4j_pass
        self.poc_size = POC_SIZE  # TODO
        # PyTorch geometric magic
        super(RecSysBatchDS, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        self.batch = SocialNetworkBatch()

        with open(self.path, "rb") as f:
            self.batch.ParseFromString(f.read())

        # Neo4j definitions
        self.user_query_format = """
                UNWIND {user_list} AS u_id
                MATCH (u: User) WHERE id(u) = u_id
                RETURN u.follower_count AS follower_count, u.following_count AS following_count, u.is_verified AS is_verified
                """

        self.tweet_query_format = """
                UNWIND {tweet_list} AS t_id
                MATCH (t: Tweet) WHERE id(t) = t_id
                RETURN t.text_tokens AS text_tokens
                """

        uri = f"bolt://35.204.0.240:7687"
        driver = neo4j.GraphDatabase.driver(uri, auth=("neo4j", self.neo4j_pass))
        self.session = driver.session()

        # Tweet tokenization
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.model.to(self.device)

        # Read data into huge `Data` list.
        data_list = [self.data_item(idx) for idx in range(0, self.poc_size)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process_ut_pairs(self, ut_pairs, proto_edge_types, start_index):

        # reaction_int is needed for stratified split
        # it can also be used to filter 'seen' reactions
        reaction_vector, reaction_int = reactions(proto_edge_types)

        sn_edges, = torch.where(start_index == ut_pairs[1])
        sn_reaction_int = reaction_int[sn_edges]
        sn_reaction_vector = reaction_vector[sn_edges]

        if sn_reaction_vector.size(0) == 0:  # no tweets for the user
            sn_edges_train, sn_edges_test, sn_edges_masked = [], [], []
            sn_reaction_vector_train, sn_reaction_vector_test = torch.empty(0, 4), torch.empty(0, 4)
        elif sn_reaction_vector.size(0) == 1:  # use that tweet for tests
            sn_edges_train, sn_edges_test, sn_edges_masked = [], sn_edges, sn_edges
            sn_reaction_vector_train, sn_reaction_vector_test = torch.empty(0, 4), sn_reaction_vector
        elif sn_reaction_vector.size(0) == 2:  # one for train, one for test
            i1 = random.getrandbits(1)
            i2 = (i1 + 1) % 2
            i1, i2 = torch.tensor([i1]), torch.tensor([i2])
            sn_edges_train, sn_edges_test, sn_edges_masked = sn_edges[i1], sn_edges[i2], sn_edges
            sn_reaction_vector_train, sn_reaction_vector_test = sn_reaction_vector[i1], sn_reaction_vector[i2]
        else:  # we have enough tweets for splits
            try:
                _, sn_edges_masked, _, sn_reaction_int_masked, _, sn_reaction_vector_masked = \
                    train_test_split(sn_edges, sn_reaction_int, sn_reaction_vector, stratify=sn_reaction_int,
                                     test_size=0.4,
                                     random_state=0)
            except ValueError:  # can not stratify because of not enough items - turn off stratification
                _, sn_edges_masked, _, sn_reaction_int_masked, _, sn_reaction_vector_masked = \
                    train_test_split(sn_edges, sn_reaction_int, sn_reaction_vector, test_size=0.4,
                                     random_state=0)

            try:
                sn_edges_train, sn_edges_test, sn_reaction_vector_train, sn_reaction_vector_test = train_test_split(
                    sn_edges_masked, sn_reaction_vector_masked, stratify=sn_reaction_int_masked, test_size=0.5,
                    random_state=0)
            except ValueError:  # can not stratify because of not enough items - turn off stratification
                sn_edges_train, sn_edges_test, sn_reaction_vector_train, sn_reaction_vector_test = train_test_split(
                    sn_edges_masked, sn_reaction_vector_masked, test_size=0.5, random_state=0)

        sn_tweet_index_train = ut_pairs[0, sn_edges_train]
        sn_tweet_index_test = ut_pairs[0, sn_edges_test]

        ut_pairs_gcn = torch.t(torch_delete(torch.t(ut_pairs), sn_edges_masked))
        reaction_vector_gcn = torch_delete(reaction_vector, sn_edges_masked)
        edge_index, edge_type = gcn_attributes(ut_pairs_gcn, reaction_vector_gcn)

        return edge_index, edge_type, \
               sn_reaction_vector_train, sn_reaction_vector_test, sn_tweet_index_train, sn_tweet_index_test

    def data_item(self, index):
        print(f"Processing item {index}")
        nn = self.batch.elements[index]
        snn = nn.social_neighbourhood
        cnn = nn.content_neighbourhood

        if self.verbose:
            print("#nodes in batch: {}".format(len(snn.nodes)))
            print("#tweets in batch: {}".format(len(cnn.nodes)))

        user_nodes = [snn.start] + list(snn.nodes)
        user_result = self.session.run(self.user_query_format.format(user_list=user_nodes))
        users = torch.tensor([list(row.values()) for row in user_result.data()], dtype=torch.float32)
        assert len(user_nodes) == users.size(0), "Protobuf user nodes and neo4j ones differ in size"

        tweet_nodes = list(cnn.nodes)
        tweet_result = self.session.run(self.tweet_query_format.format(tweet_list=tweet_nodes))
        tweets = [row['text_tokens'] for row in tweet_result.data()]
        assert len(tweet_nodes) == len(tweets), "Protobuf tweet nodes and neo4j ones differ in size"

        # recalculate edge index
        ut_sources = list(cnn.edge_index_source)
        ut_targets = list(cnn.edge_index_target)
        f_sources = list(snn.edge_index_source)
        f_targets = list(snn.edge_index_target)

        # very slow please don't use it in final implementation!
        fixed_f_sources = [user_nodes.index(v) for v in f_sources]
        fixed_f_targets = [user_nodes.index(v) for v in f_targets]
        fixed_ut_sources, fixed_ut_targets = [], []
        for s, t in zip(ut_sources, ut_targets):
            fixed_ut_sources.append(user_nodes.index(s))
            fixed_ut_targets.append(tweet_nodes.index(t))

        # Data constructing
        data = RecSysData()
        start_index = user_nodes.index(snn.start)
        data.start_index = start_index
        data.x_users = users

        ut_pairs = torch.tensor((fixed_ut_targets, fixed_ut_sources), dtype=torch.int64)
        edge_index, edge_type, \
        sn_reaction_vector_train, sn_reaction_vector_test, sn_tweet_index_train, sn_tweet_index_test \
            = self.process_ut_pairs(ut_pairs, cnn.edge_types.attributes, start_index)

        data.edge_index = edge_index
        data.edge_type = edge_type
        data.sn_reaction_vector_train = sn_reaction_vector_train
        data.sn_reaction_vector_test = sn_reaction_vector_test
        data.sn_tweet_index_train = sn_tweet_index_train
        data.sn_tweet_index_test = sn_tweet_index_test
        data.size_train = sn_tweet_index_train.size(0)
        data.size_test = sn_tweet_index_test.size(0)

        with torch.no_grad():
            if len(tweets) > 0:
                data.x_tweets = self.embed(tweets, batch=32)
                # data.x_tweets = torch.zeros((len(tweets), 768))
            else:
                print("No tweets!")
                data.x_tweets = torch.zeros((0, 768))

        data.f_edge_index = torch.tensor((fixed_f_sources, fixed_f_targets), dtype=torch.int64)
        return data

    def embed(self, tweets, batch=16):
        inputs = self.tokenizer.batch_encode_plus(
            tweets, padding='max_length',
            truncation=True, return_tensors='pt',
            is_split_into_words=True, max_length=150
        )
        inputs.to(self.model.device)

        result = torch.zeros((len(tweets), 768), dtype=torch.float32)
        for i in tqdm(range(0, len(tweets), batch)):
            output = self.model(
                inputs.input_ids[i:i + batch],
                attention_mask=inputs.attention_mask[i:i + batch],
                output_hidden_states=True
            )
            end = len(tweets) if i + batch > len(tweets) else i + batch
            result[i:end] = output.pooler_output.cpu().detach()
        return result
