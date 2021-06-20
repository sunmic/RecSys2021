import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Optional

from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv, RGCNConv

from GCN.datasets import RecSysBatchDS
from GCN.config import POC_ROOT
import torchmetrics as metrics

class MLP(torch.nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_nodes, 256)
        self.lin2 = nn.Linear(256, 32)
        self.lin3 = nn.Linear(32, out_nodes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x


class Net(pl.LightningModule):
    def __init__(self, num_tweet_features, num_user_features, path, neo4j_pass, lr=1e-3, batch_size=1):
        super().__init__()

        self.neo4j_pass = neo4j_pass
        self.path = path

        self.lr = 1e-3
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.ones(4) * 10)
        self.batch_size = batch_size

        # self.conv1 = SAGEConv((num_tweet_features, num_user_features), 64)
        self.conv1 = RGCNConv(in_channels=(num_tweet_features, num_user_features), out_channels=64, num_relations=5)
        self.norm1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 64)  # może byc GCNConv lub pochodne, jak chcemy
        self.norm2 = nn.BatchNorm1d(64)
        self.conv3 = SAGEConv(64, 32)
        self.cls = MLP(num_tweet_features + 32, 4)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def bin_tensor2int(self, tensor):
        return int("".join([str(x.int().item()) for x in tensor]), 2)

    def forward(self, x):
        data = x
        start = data.start_index  # index of start node
        x_users = data.x_users  # shape: [N_u, U]
        x_tweets = data.x_tweets  # shape: [N_t, T]
        follow_edge_index = data.f_edge_index  # [2, N_f]
        edge_index = data.edge_index
        edge_type = data.edge_type
        target_size = data.target_size
        target_tweets_index = data.target_tweets_index

        h = self.conv1(x=(x_tweets, x_users), edge_index=edge_index, edge_type=edge_type)
        h = self.norm1(h)
        F.relu(h)

        h = self.conv2(x=h, edge_index=follow_edge_index)
        h = self.norm2(h)
        F.relu(h)

        # ograniczanie follow_edge_index dla tej warstwy? warto sprawdzic, czy jest lepiej????
        h = self.conv3(x=h, edge_index=follow_edge_index)
        F.relu(h)

        # mlp, do whatever u want with it
        h = h[start]
        h = torch.cat([a.repeat(times, 1) for a, times in zip(h, target_size)], 0)
        h = torch.cat((h, x_tweets[target_tweets_index]), -1)
        # F.sigmoid(h) BCEWithLogitsLoss - czy musimy dawać dodatkowo sigmoida ?

        return self.cls(h)

    def step(self, batch, batch_idx, stage: str):
        batch.target = None
        batch.target_tweets_index = None
        if stage == 'train':
            batch.target = batch.sn_reaction_vector_train
            batch.target_tweets_index = batch.sn_tweet_index_train
            batch.target_size = batch.size_train
        elif stage == 'val' or stage == 'test':
            batch.target = batch.sn_reaction_vector_test
            batch.target_tweets_index = batch.sn_tweet_index_test
            batch.target_size = batch.size_test
        else:
            raise ValueError("Invalid stage")
        x, y = batch, batch.target
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        # performance metrics
        acc = ((y_hat > 0) == y).sum() / (y.size(0) * y.size(1))
        engagement_acc = (((y_hat > 0) == y) * y).sum() / y.sum()
        engagement_prec = (((y_hat > 0) == y) * y).sum() / ((y_hat > 0).sum() + 1e-10)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_engag_acc', engagement_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_engag_prec', engagement_prec, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        y_hat = F.sigmoid(y_hat)  # are we sure ? Maybe it is better to do it in forward ?
        y = y.long()
        prec = metrics.functional.precision(y_hat, y, multilabel=True, average='samples')
        f1 = metrics.functional.f1(y_hat, y, multilabel=True, average='samples')
        tm_acc = metrics.functional.accuracy(y_hat, y, average='samples')
        self.log(f'{stage}_torchmetrics_acc', tm_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_torchmetrics_prec', prec, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_torchmetrics_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch, batch_idx, stage='test')

    def prepare_data(self):
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "FAUST")
        # self.pre_transform = T.Compose([T.FaceToEdge(), T.Constant(value=1)])
        # self.train_dataset = FAUST(path, True, T.Cartesian(), self.pre_transform)
        # self.test_dataset = FAUST(path, False, T.Cartesian(), self.pre_transform)

        # root = './root'
        root = POC_ROOT
        # path = '/content/drive/Shareddrives/RecSys21/neighbourhoods/batch_0_1000'
        # path = 'H:/Dyski współdzielone/RecSys21/neighbourhoods/batch_0_1000'
        self.train_dataset = RecSysBatchDS(root, self.path, self.neo4j_pass)
        self.test_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
