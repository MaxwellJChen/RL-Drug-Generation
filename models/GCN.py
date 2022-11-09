import torch
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

"""
Base Sequential Graph Generator Model
"""

"""
Generation pipeline:
1. Accept graph
3. Predict termination
4. Find probability and identity of edge existing between current graph node and node bank for all nodes
5. Choose highest probability edge and append to graph
6. Repeat
"""

# Given embedded graph, predicts termination
class termination_predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(27, 60)
        self.conv2 = GCNConv(60, 50)

        self.fc1 = nn.Linear(50, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, G):
        # Global pooling into 1D vector
        batch = G.batch
        edge_index = G.edge_index
        x = G.x

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        x = global_add_pool(x, batch)

        # FCN to final probability output
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# Given embedded graph, "scores" each node on chance of forming an edge
class node_predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(27, 60)
        self.conv2 = GCNConv(60, 50)

        self.bank_fcn = nn.Linear(27, 50)

        self.fc1 = nn.Linear(50, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, G, bank):
        edge_index = G.edge_index
        x = G.x

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        bank = self.bank_fcn(bank)

        x = torch.vstack((x, bank))

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim = 0)

        return x

# Given embedded graph, bank, and node indices, outputs probability distribution of edges
class edge_predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(27, 60)
        self.conv2 = GCNConv(60, 50)

        self.bank_fcn = nn.Linear(27, 50)

        self.fc1 = nn.Linear(100, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, G, bank, node1, node2):
        edge_index = G.edge_index
        x = G.x

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        bank = self.bank_fcn(bank)

        x = torch.vstack((x, bank))

        x = torch.hstack((x[node1], x[node2]))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim = 0)

        return x