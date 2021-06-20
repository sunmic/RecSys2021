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
    def __init__(self, in_nodes, out_nodes, hidden=[256], dropout_rate=0.5):
        super(MLP, self).__init__()
        if len(hidden) == 0:
            raise ValueError("WTF invalid hidden sizes")

        self.lin_in = nn.Linear(in_nodes, hidden[0])
        self.lin_hidden = nn.ModuleList()
        for h_in, h_out in zip(hidden, hidden[1:] + [out_nodes]):
            lin = nn.Linear(h_in, h_out)
            self.lin_hidden.append(lin)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.lin_in(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        for lin in self.lin_hidden:
            x = lin(x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x


class Net(pl.LightningModule):
    def __init__(self, num_tweet_features, num_user_features, root, path, neo4j_pass, clf, num_hidden=64, num_output=32, lr=1e-3, batch_size=32, loss_weights=torch.tensor([10, 10, 10, 10])):
        super().__init__()

        self.neo4j_pass = neo4j_pass
        self.path = path
        self.root = root

        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss(weight=loss_weights)
        self.batch_size = batch_size

        # self.conv1 = SAGEConv((num_tweet_features, num_user_features), 64)
        self.conv1 = RGCNConv(in_channels=(num_tweet_features, num_user_features), out_channels=num_hidden, num_relations=5)
        self.norm1 = nn.BatchNorm1d(num_hidden)
        self.conv2 = SAGEConv(num_hidden, num_hidden)  # może byc GCNConv lub pochodne, jak chcemy
        self.norm2 = nn.BatchNorm1d(num_hidden)
        self.conv3 = SAGEConv(num_hidden, num_output)
        
        self.clf = clf

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

        return self.clf(h)

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
        self.train_dataset = RecSysBatchDS(self.root, self.path, self.neo4j_pass)
        self.test_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
