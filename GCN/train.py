import argparse
import torch
from GCN.models import Net
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--neo4j_pass', type=str, default=None, metavar='N',
                    help='neo4j password')
parser.add_argument('--path', type=str, default='/content/drive/Shareddrives/RecSys21/neighbourhoods/batch_0_1000', metavar='N',
                    help='path')

args = parser.parse_args()

num_tweet_features = 768
num_user_features = 3
net = Net(num_tweet_features, num_user_features, lr=1e-2, path=args.path, neo4j_pass=args.neo4j_pass, batch_size=32)

trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None, fast_dev_run=False, max_epochs=100)

trainer.fit(net)

print("Done.")