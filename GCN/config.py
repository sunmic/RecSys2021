import socket
import sys
from clearml import Task
import torch_geometric


hostname = socket.gethostname()

# required by colab worker
# torch-cluster @ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# torch-scatter @ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# torch-sparse @ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# torch-spline-conv @ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
Task.add_requirements('torch-scatter', '@ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html')
Task.add_requirements('torch-sparse', '@ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html')
Task.add_requirements('torch-cluster', '@ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html')
Task.add_requirements('torch-spline-conv', '@ https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html')
Task.add_requirements('neo4j')
Task.add_requirements('transformers')
Task.add_requirements('torchmetrics')

if hostname == 'LAPTOP-LKDD3MT2':
    POC_SIZE = 1000
    POC_ROOT = './root'
else:   # colab
    POC_SIZE = 1000
    POC_ROOT = '/content/root'
    sys.path.append('..')
    sys.path.append('../batch/neo4j/proto')
