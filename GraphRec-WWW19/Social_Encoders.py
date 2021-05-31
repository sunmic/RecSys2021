import torch
import torch.nn as nn
import torch.nn.functional as F

class Social_Encoder(nn.Module):

    def __init__(self, db, features, embed_dim, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.db = db
        self.query_template = 'MATCH (u:User)-[:Follow]->(n: User) WHERE id(u) IN {} RETURN id(u), collect(id(n))'
        self.features = features
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):
        to_neighs = []
        results = self.db.run(self.query_template.format(nodes.tolist()))
        social_adj_dict = {k: v for k, v in results.values()}
        for node in nodes.tolist():
            neighs = social_adj_dict[node]
            to_neighs.append(neighs)
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device, dtype=torch.float16).float()
        self_feats = self_feats.t()
        
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
