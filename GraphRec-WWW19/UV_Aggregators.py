import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Attention import Attention

class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim).to(self.device)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.att = Attention(self.embed_dim).to(self.device)

    def r2id(self, ratings):
        ratings = np.array(ratings, dtype='int').astype('str')
        return [int("".join(x), 2) for x in ratings]


    def forward(self, nodes, history_uv, history_r):
        embed_matrix = torch.empty(len(history_uv), self.embed_dim).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = self.r2id(history_r[i])

            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history].to(self.device)
                uv_rep = self.u2e.weight[nodes[i]].to(self.device)
            else:
                # item component
                e_uv = self.u2e.weight[history].to(self.device)
                uv_rep = self.v2e.weight[nodes[i]].to(self.device)

            e_r = self.r2e.weight[tmp_label].to(self.device)
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats