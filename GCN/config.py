import socket
import sys
from clearml import Task
import torch_geometric


hostname = socket.gethostname()

# required by colab worker
if hostname == 'LAPTOP-LKDD3MT2':
    pass
else:   # colab
    # Task.add_requirements()
    sys.path.append('..')
    sys.path.append('../batch/neo4j/proto')
