import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class UV_Encoder(nn.Module):

    def __init__(self, db, features, embed_dim, aggregator, cuda="cpu", uv=True,):
        super(UV_Encoder, self).__init__()

        self.db = db
        self.features = features
        self.uv = uv
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

        self.db = db
        if uv:
            # user-items dict
            self.query_template = '''\
                MATCH (u:User)
                WHERE u.id IN {}
                CALL {
                    WITH u
                    MATCH (u)-[:Like|Retweet|Reply|RetweetComment|Seen]->(t: Tweet)
                    RETURN collect(DISTINCT t) as tweets
                }
                RETURN
                    [t in tweets | t.id] as v,
                    [t in tweets | [exists((u)-[:Like]->(t)), exists((u)-[:Retweet]->(t)), exists((u)-[:Reply]->(t)), exists((u)-[:RetweetComment]->(t)) ]] as r'''
        else:
            # item-users dict
            self.query_template = '''\
                MATCH (t:Tweet)
                WHERE t.id IN {}
                CALL {
                    WITH t
                    MATCH (u:User)-[:Like|Retweet|Reply|RetweetComment|Seen]->(t)
                    RETURN collect(t) as tweets, collect(DISTINCT u) as users
                }
                RETURN
                    [u in users | u.id] as u,
                    [u in users | [exists((u)-[:Like]->(t)), exists((u)-[:Retweet]->(t)), exists((u)-[:Reply]->(t)), exists((u)-[:RetweetComment]->(t)) ]] as r'''
               


    def forward(self, nodes):
        tmp_history_uv = []
        tmp_history_r = []
        results = self.db.run(self.query_template.format(nodes))
        for result in results:
            v, r = result.values()
            tmp_history_uv.append(v)
            tmp_history_r.append(r)

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r)  # user-item network
        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
