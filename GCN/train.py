import config
import argparse
import torch
from GCN.models import Net, MLP
from GCN.config import POC_ROOT
import pytorch_lightning as pl
from clearml import Task


NUM_TWEET_FEATURES = 768
NUM_USER_FEATURES = 3
NUM_LABELS = 4


def parse_integer_list(hidden_str):
    chunks = hidden_str.split(",")
    return [int(chunk) for chunk in chunks]


def parse_args():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--neo4j_pass', type=str, default=None, metavar='N', help='neo4j password')
    parser.add_argument('--path', type=str, default='/content/drive/Shareddrives/RecSys21/neighbourhoods/new_batch_0_1000', metavar='N', help='neighbourhood package path')
    parser.add_argument('--root', type=str, default=POC_ROOT, help='Dataloader root')
    parser.add_argument('--mlp_hidden', type=str, default='256', help='Comma-separated mlp hidden sizes')
    parser.add_argument('--mlp_dropout', type=float, default=0.5, help='MLP dropout rate')
    parser.add_argument('--gcn_hidden', type=int, default=64, help='GCN hidden size')
    parser.add_argument('--gcn_output', type=int, default=32, help='GCN output size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--loss_weights', type=str, default='10,10,10,10', help='Loss functions label weights (length of 4)')
    parser.add_argument('--execute_remotely', type=bool, default=False, help='execute remotely as ClearML Task')

    args = parser.parse_args()

    return args


def main(args):
    if args.execute_remotely:
        task = Task.init(project_name='RecSys2021', task_name='GCN-weight-10-params')
        task.execute_remotely("default")

    # gcn params
    gcn_hidden = args.gcn_hidden
    gcn_output = args.gcn_output

    # mlp params
    mlp_input = gcn_output + NUM_TWEET_FEATURES
    mlp_output = NUM_LABELS
    mlp_dropout = args.mlp_dropout
    mlp_hidden = None
    try:
        mlp_hidden = parse_integer_list(args.mlp_hidden)
    except ValueError:
        raise ValueError("mlp_hidden must be comma separated list of integers")

    # common params
    lr = args.lr
    batch_size = args.batch
    loss_weights = None
    try:
        loss_weights = parse_integer_list(args.loss_weights)
        assert len(loss_weights) == NUM_LABELS, "Loss weights must match num labels (4)"
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
    except ValueError:
        raise ValueError("loss_weights must be comma separated list of integers")

    mlp = MLP(mlp_input, mlp_output, hidden=mlp_hidden, dropout_rate=mlp_dropout)
    net = Net(
        NUM_TWEET_FEATURES, NUM_USER_FEATURES, 
        num_hidden=gcn_hidden, num_output=gcn_output, 
        clf=mlp, lr=lr, batch_size=batch_size, loss_weights=loss_weights, 
        root=args.root, path=args.path, neo4j_pass=args.neo4j_pass
    )

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None, fast_dev_run=False, max_epochs=100)

    trainer.fit(net)

    trainer.validate(net)

    print("Done.")


if __name__=='__main__':
    args = parse_args()
    main(args)