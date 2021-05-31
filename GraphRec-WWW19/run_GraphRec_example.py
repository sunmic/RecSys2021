import torch
import torch.nn as nn
from clearml import Task
from torch import  stack
import numpy as np
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse
import os
import logging as log
from utils.data.dataset import Neo4jDataset
from utils.data.query import Neo4jQuery
from neo4j import GraphDatabase

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
    
        self.w_uv3 = nn.Linear(16, 4)
        self.criterion = nn.BCELoss()
    
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        
        scores = torch.sigmoid(self.w_uv3(x))
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        
        # batch_nodes_u = Tensor([int(x, 16) for x in batch_nodes_u])
        # batch_nodes_v = Tensor([int(x, 16) for x in batch_nodes_v])
        labels_list = stack(labels_list).t().float()
        
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--host', type=str, default='35.204.0.240', help='database host')
    parser.add_argument('-u', type=str, default='neo4j', help='database user')
    parser.add_argument('-p', type=str, help='database password')
    parser.add_argument('--execute_remotely', type=bool, default=False, help='execute remotely as ClearML Task')

    args = parser.parse_args()

    if args.execute_remotely:
        task = Task.init(project_name='RecSys2021', task_name='run_GraphRec_example')
        task.execute_remotely("default")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    
    host = args.host
    user = args.u
    password = args.p
    driver = GraphDatabase.driver(f'bolt://{host}:7687', auth=(user, password))

    trainset = Neo4jDataset(driver, Neo4jQuery(
        'MATCH (u: User)-[r:Like|Retweet|Reply|RetweetComment|Seen]->(t:Tweet)',
        'RETURN DISTINCT id(u) as u, id(t) as t, \
            [exists((u)-[:Like]->(t)), exists((u)-[:Retweet]->(t)), exists((u)-[:Reply]->(t)), exists((u)-[:RetweetComment]->(t)) ]'
    ), 64, count_all=False)

    # SHOULD BE REPLACED WITH REAL TEST SET
    testset = Neo4jDataset(driver, Neo4jQuery(
        'MATCH (u: User)-[r:Like|Retweet|Reply|RetweetComment|Seen]->(t:Tweet)',
        'RETURN DISTINCT id(u) as u, id(t) as t, \
            [exists((u)-[:Like]->(t)), exists((u)-[:Retweet]->(t)), exists((u)-[:Reply]->(t)), exists((u)-[:RetweetComment]->(t)) ]'
    ), 64, count_all=False)
    
    log.info("Train Loader")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size) # shuffle=True
    log.info("Test Loader")
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size) # shuffle=True

    db = driver.session()
    
    num_users, max_user_id, min_user_id = 45964681, 46017045, 0
    # num_users, max_user_id, min_user_id = [r.values() for r in db.run("MATCH (u:User) RETURN COUNT(u), MAX(id(u)), MIN(id(u))")][0][0]
    log.info("#Users = {}".format(num_users))
    log.warn("Precomputed values")
    
    num_items, max_item_id, min_item_id = 326769765, 372735814, 45882961
    #num_items, max_item_id, min_item_id = [r.values() for r in db.run("MATCH (t:Tweet) RETURN COUNT(t), MAX(id(t)), MIN(id(t))")][0][0]
    log.info("#Items = {}".format(num_items))
    log.warn("Precomputed values")

    num_ratings = 16
    db.close()

    # TODO : Use embeddings more effectively to not waist memory usage
    # u2e = nn.Embedding(max_user_id + 1, embed_dim).to(device)

    # TODO : Use embeddings more effectively to not waist memory usage
    # v2e = nn.Embedding(max_item_id + 1, embed_dim).to(device)

    # workaround: nn.Embedding is unable to use non-default dtype
    num_embeddings, embedding_dim = max_item_id + 1, embed_dim
    embed_weight = torch.zeros(num_embeddings, embedding_dim, dtype=torch.float16)
    torch.nn.init.normal_(embed_weight)
    uv2e = nn.Embedding(num_embeddings, embedding_dim, _weight=embed_weight).to(device)
    log.info("uv2e {}x{} embeddings initialized".format(max_item_id + 1, embed_dim))
    log.warn("Use embeddings more effectively to not waist memory usage")

    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    log.info("r2e {}x{} embeddings initialized".format(num_ratings, embed_dim))

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(uv2e, r2e, uv2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(driver.session(), uv2e, embed_dim, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), uv2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(driver.session(), lambda nodes: enc_u_history(nodes).t(), embed_dim, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(uv2e, r2e, uv2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(driver.session(), uv2e, embed_dim, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(graphrec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()
