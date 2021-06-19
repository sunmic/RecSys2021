import socket
import sys
from clearml import Task
import torch_geometric


hostname = socket.gethostname()

# required by colab worker
# torch-cluster @ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
# torch-scatter @ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
# torch-sparse @ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
# torch-spline-conv @ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
Task.add_requirements('torch-scatter', '@ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl')
Task.add_requirements('torch-sparse', '@ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl')
Task.add_requirements('torch-cluster', '@ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl')
Task.add_requirements('torch-spline-conv', '@ https://pytorch-geometric.com/whl/torch-1.8.0+cu101/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl')
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
