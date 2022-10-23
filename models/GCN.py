import torch
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F
import torch.nn as nn

class edge_exists_predictor(torch.nn.Module):
    # Accept graph, node in graph, and predicted node
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 60)
        self.conv2 = GCNConv(60, 50)
        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.output = nn.Linear(10, 1)

    def foward(self, G, n_test, n_pred): # Accepts graph, index of node to test, and predicted node, outputs whether or not they are connected via edge
        batch = G.batch
        edge_index = G.edge_index
        x = G.x

        # Node embedding
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Global pooling into 1D vector
        x = global_add_pool(x, batch)

        #

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output(x)
        return x

# class edge_value_predictor(torch.nn.Module):



class termination_predictor(torch.nn.Module): # Predicts probability of termination
    # Operates on global node vector
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 60)
        self.conv2 = GCNConv(60, 50)
        self.fc1 = nn.Linear(50, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, G):
        batch = G.batch
        edge_index = G.edge_index
        x = G.x

        # Node embedding
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Global pooling into 1D vector
        x = global_add_pool(x, batch)

        # FCN to one-hot encoded node
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


# class termination_model(torch.nn.Module):

class edge_probability_model(torch.nn.Module): # Predicts probability of edge type existing between nodes
    def __init__(self, final_node_embedding):
        self.fc1 = nn.Linear(2*final_node_embedding, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear()


# class edge_type_model(torch.nn.Module):