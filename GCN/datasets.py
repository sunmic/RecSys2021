from torch_geometric.data import InMemoryDataset, Dataset
from social_neighbourhood_pb2 import SocialNetworkBatch, SocialNetworkNeighbourhood
from torch_geometric.data import Data
import torch
import neo4j
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np


class RecSysData(Data):
    def __init__(self):
        super(RecSysData, self).__init__()

    def __inc__(self, key, value):
        if key == 'start_index':
            return self.x_users.size(0)
        if key == 'x_users':
            return self.x_users.size(0)
        if key == 'x_tweets':
            return self.x_tweets.size(0)
        if key == 'ut_edges':
            return torch.tensor([[self.x_users.size(0)], [self.x_tweets.size(0)]]) 
        if key == 'ut_edge_index_gcn':
            return self.ut_edges.size(1)
        if key == 'ut_edge_index_train':
            return self.ut_edges.size(1)
        if key == 'ut_edge_index_test':
            return self.ut_edges.size(1)
        if key == 'f_edge_index':
            return self.x_users.size(0)
        else:
            return super().__inc__(key, value)

    def __cat_dim__(self, key, value):
        if key == 'ut_edges':
            return 1
        elif 'ut_edge_index' in key:
            return 0
        else:
            return super().__cat_dim__(key, value)


class RecSysBatchDS(InMemoryDataset):
    def __init__(self, root, path, neo4j_pass, transform=None, pre_transform=None, verbose=False, device='cuda'):
        self.path = path
        self.verbose = verbose
        self.device = device
        self.neo4j_pass = neo4j_pass
        self.poc_size = 10  # TODO
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

    # def __len__(self):
    #     return len(self.batch.elements)

    # def __getitem__(self, index) -> T_co:
    def split_ut_edge_index(self, ut_edges, start_index, test_size=0.2, train_size=0.2):
        start_node_edge_index, = np.where(start_index == ut_edges[1, :])

        start_node_edge_index = np.random.permutation(start_node_edge_index)
        test_size = int(len(start_node_edge_index) * test_size)
        train_size = int(len(start_node_edge_index) * train_size)

        ut_edge_index_test = start_node_edge_index[:test_size]
        ut_edge_index_train = start_node_edge_index[test_size:train_size+test_size]
        ut_edge_index_gcn = start_node_edge_index[train_size+test_size:]
        
        return ut_edge_index_gcn, ut_edge_index_train, ut_edge_index_test

    def data_item(self, index):
        nn = self.batch.elements[index]
        snn = nn.social_neighbourhood
        cnn = nn.content_neighbourhood

        if self.verbose:
            print("#nodes in batch: {}".format(len(snn.nodes)))
            print("#tweets in batch: {}".format(len(cnn.nodes)))

        user_nodes = [snn.start] + list(snn.nodes)
        # %time user_result = session.run(user_query_format.format(user_list=user_nodes))
        user_result = self.session.run(self.user_query_format.format(user_list=user_nodes))
        users = torch.tensor([list(row.values()) for row in user_result.data()], dtype=torch.float32)

        tweet_nodes = list(cnn.nodes)
        # %time tweet_result = session.run(tweet_query_format.format(tweet_list=tweet_nodes))
        tweet_result = self.session.run(self.tweet_query_format.format(tweet_list=tweet_nodes))
        tweets = [row['text_tokens'] for row in tweet_result.data()]

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
        with torch.no_grad():
            data.x_tweets = self.embed(tweets, batch=8)
        
        
        # Split ut edge indices
        ut_edges = torch.tensor((fixed_ut_targets, fixed_ut_sources), dtype=torch.int64)
        ut_edge_index_gcn, ut_edge_index_train, ut_edge_index_test = self.split_ut_edge_index(ut_edges, data.start_index, train_size=0.2, test_size=0.2)
        data.ut_edges = ut_edges
        data.ut_edge_index_gcn = ut_edge_index_gcn
        data.ut_edge_index_train = ut_edge_index_train
        data.ut_edge_index_test = ut_edge_index_test

        data.f_edge_index = torch.tensor((fixed_f_sources, fixed_f_targets), dtype=torch.int64)
        
        # data.tweet = ...  # TODO
        data.target = torch.rand(len(data.ut_edge_index_train), 4)  # TODO, nie mamy targetów
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
                inputs.input_ids[i:i+batch],
                attention_mask=inputs.attention_mask[i:i+batch],
                output_hidden_states=True
            )
            end = len(tweets) if i+batch > len(tweets) else i + batch
            result[i:end] = output.pooler_output.cpu().detach()
        return result


