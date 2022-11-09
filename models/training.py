import pickle
import numpy as np
import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx
import torch
import rdkit.Chem as Chem
from rdkit.Chem.Draw import MolToImage

from models.GCN import termination_predictor, edge_predictor, node_predictor


def nx_draw(graph):
    nx.draw(torch_geometric.utils.to_networkx(graph))
    plt.show()


def print_graph(graph):
    print('x')
    print(graph.x)
    print('edge_index')
    print(graph.edge_index)
    print('edge_attr')
    print(graph.edge_attr)
    print()


def display_smiles(smiles):
    """Display with SMILES RDKit"""
    mol = Chem.MolFromSmiles(smiles)
    img = MolToImage(mol, size=(1000, 1000))
    img.show()


def BFS(G):
    """Graphs are ordered deterministically through a BFS"""
    pairs = [[int(i), int(j)] for i, j in zip(G.edge_index[0], G.edge_index[1])]
    # print(pairs)
    # print(np.unique(pairs))

    nodes = np.unique(pairs)
    d = []
    for x in range(len(nodes)):
        where = list(np.where(pairs == nodes[x]))
        v = [pairs[i][int(not bool(j))] for i, j in zip(where[0], where[1])]
        if x == 0:
            d = [[str(nodes[x]), [str(a) for a in np.unique(v)]]]
        else:
            d += [[str(nodes[x]), [str(a) for a in np.unique(v)]]]
    d = dict(d)

    visited = []
    queue = []
    visited.append(str(nodes[0]))
    queue.append(str(nodes[0]))
    while queue:
        m = queue.pop(0)
        for neighbor in d[m]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    visited = [int(i) for i in visited]
    return visited


def add_node(g_main, g_intermediate, v):
    """
    Update x, edge_index, and edge_attr to hold a new node with the right connections in the BFS ordering
    Changes in function apply directly to original input g_intermediate object

    :param g_main: Original graph
    :param g_intermediate: Graph "recieving" nodes
    :param v: The deterministic BFS ordering of g_main
    :return: A new g_intermediate with the next node appended
    """

    # Update g_intermediate.x
    step = len(g_intermediate.x)  # Number of nodes in g_intermediate

    if step == g_main.num_nodes:  # If it has the same number of nodes as g_main, then the two molecules are equivalent and termination ensues
        return "terminate"

    # print(f'step: {step}')
    node_idx = v[step]  # The main_idx of the node to append next
    node_to_add = g_main.x[int(node_idx)]  # Finding the attr of the new node
    x = torch.vstack(
        (g_intermediate.x, node_to_add))  # The new node appended is at the end of the g.x matrix

    # Update g_intermediate.edge_index by connecting the new node to pre-existing nodes in the graph (can accomodate multiple bonds with pre-existing nodes)
    idx = []
    for i in range(len(g_main.edge_index[0])):
        if g_main.edge_index[0][i] == v[step] or g_main.edge_index[1][i] == v[step]:
            idx.append(g_main.edge_index[:, i].tolist())
    main_idx = np.unique(
        idx).tolist()  # Holds the original main_idx of the new node and the main_idx of the nodes it forms edges with in g_main
    # print(f'99 main_idx: {main_idx}')

    # print(f'visited: {v}')
    intermediate_idx = [v.index(i) for i in
                        main_idx]  # Find intermediate_idx which corresponds to the intermediate_nodes the new node forms connections with
    # print(intermediate_idx)
    for i in reversed(range(len(intermediate_idx))):
        if intermediate_idx[i] >= step:
            intermediate_idx.remove(intermediate_idx[i])  # Remove nodes that have not yet been added
            main_idx.remove(main_idx[i])

    # print(f'main_idx: {main_idx}')
    # print(f'intermediate_idx: {intermediate_idx}')
    # print()

    # Format the idx so it can be appended to edge_index of g_intermediate
    edge_idx = []  # The list to hold the sparse connection matrix for intermediate
    attr_idx = []  # The list to hold the main_idx of the edge attributes
    for i in range(len(intermediate_idx)):
        edge_idx.append([step, intermediate_idx[i]])

        # print(np.swapaxes(g_main.edge_index, 0, 1).tolist())
        # print([step, idx[i]])

        attr_idx.append(np.swapaxes(g_main.edge_index, 0, 1).tolist().index([v[step], main_idx[i]]))
        edge_idx.append([intermediate_idx[i], step])
        attr_idx.append(np.swapaxes(g_main.edge_index, 0, 1).tolist().index(
            [v[step], main_idx[i]]))  # Storing main_idx of the edge attributes
    edge_idx = torch.tensor(np.swapaxes(edge_idx, 0, 1))  # Sparse connection matrix completed

    attr = torch.tensor([g_main.edge_attr[i].tolist() for i in attr_idx])

    # If there is only one node in the intermediate graph
    if step == 1:
        edge_index = edge_idx
        edge_attr = attr
    else:
        edge_index = torch.hstack((g_intermediate.edge_index, edge_idx))
        edge_attr = torch.vstack((g_intermediate.edge_attr, attr))
    return torch_geometric.data.Data(x = x, edge_index = edge_index, edge_attr = edge_attr), intermediate_idx

"""Loading data"""
with open('data/zinc_10_graphs', 'rb') as file:
    graphs = pickle.load(file)

"""Models and Losses"""
node_model = node_predictor()
node_optim = torch.optim.Adam(params=node_model.parameters(), lr=0.0001)

termination_model = termination_predictor()
termination_optim = torch.optim.Adam(params=termination_model.parameters(), lr=0.0001)

edge_model = edge_predictor()
edge_optim = torch.optim.Adam(params=edge_model.parameters(), lr=0.0001)

termination_criterion = torch.nn.MSELoss()
node_criterion = torch.nn.CrossEntropyLoss()
edge_criterion = torch.nn.CrossEntropyLoss()

bank = torch.zeros((13, 27))
for i in range(13):
    bank[i][i] = 1

"""Training Loop"""
for graph_iterator in range(len(graphs)):
    g_final = graphs[graph_iterator]
    v = BFS(g_final)
    g_intermediate = torch_geometric.data.Data(x=torch.reshape(g_final.x[v[0]], (1, 27)), edge_attr=torch.zeros((1, 4)),
                                               edge_index=torch.zeros((2, 1))) # One atom
    g_intermediate, idx = add_node(g_final, g_intermediate, v) # Two atoms

    while add_node(g_final, g_intermediate, v) != "terminate":
        """Find g[]"""
        # Given the output of add_node, create a set of one-edge-at-a-time graphs to train network on
        g = [] # Holds intermediate graphs
        g_prev = g_intermediate
        # print(g_prev.num_nodes)
        g_intermediate, idx = add_node(g_final, g_intermediate, v) # The next graph in the sequence
        # print(g_intermediate.num_nodes)
        # print(idx)

        edge = g_intermediate.edge_index.tolist()
        attr = g_intermediate.edge_attr.tolist()

        # Make it easier when training other models
        removed_idx = []
        removed_attr = []

        # Goes in reverse, gradually removing nodes from graph_intermediate (uninclusive) to reach graph_prev (inclusive)
        for i in range(len(idx)):
            removed_idx.append(idx[i])

            # Remove a new edge
            edge = np.swapaxes(np.array(edge), 0, 1) # Change edge so it is in a form where np.where works
            new_node_idx = g_intermediate.x.shape[0] - 1 # Find index of new node
            edge_idx = np.where(edge == [new_node_idx, idx[i]])[0][1] # Find the index of the point in edges where there is an edge betw new and prev node
            removed_attr.append(attr[edge_idx]) # Append removed attribute ahead of time
            edge = edge.tolist()
            edge.pop(edge_idx)
            edge.pop(edge_idx)

            if i == len(idx) - 1: # If we have reached the end of idx, that means all of the new edges will be removed
                g.append(g_prev)
                break

            # print(f'idx: {idx[i]}')
            # print(f'root node: {g_new.x.shape[0] - 1}')
            # print(f'edge1: {edge.pop(edge_idx)}')
            # print(f'edge2: {edge.pop(edge_idx)}')

            # Remove corresponding attributes
            attr.pop(edge_idx)
            attr.pop(edge_idx)

            edge = torch.tensor(np.swapaxes(edge, 0, 1).tolist())
            attr = torch.tensor(attr)

            graph = torch_geometric.data.Data(x = g_intermediate.x, edge_index = edge, edge_attr = attr)

            # "Refresh" for next loop
            edge = edge.tolist()
            attr = attr.tolist()

            g.append(graph)

        g.reverse()
        removed_idx.reverse()
        removed_attr.reverse()

        # for graph in g:
        #     nx_draw(graph)
        #     print(graph.num_nodes)
        #     print(graph.num_edges)

        """Backpropagation with g"""
        for iterator in range(len(g)):
            # Termination predictor
            termination_prediction = termination_model(g[iterator])[0][0]
            if len(g_prev.x) < len(v): # If there are still more nodes to add
                termination_loss = termination_criterion(termination_prediction, torch.tensor(0.)) # 0 indicates continue
                termination_optim.zero_grad()
                termination_loss.backward()
                termination_optim.step()
            else:
                termination_loss = termination_criterion(termination_prediction, torch.tensor(1.)) # 1 indicates terminate
                termination_optim.zero_grad()
                termination_loss.backward()
                termination_optim.step()

            # Node probability predictor
            node_prediction = node_model(g[iterator], bank)
            node_actual = torch.zeros(len(node_prediction))
            one_idx = []
            if g[iterator].num_nodes < g_intermediate.num_nodes: # Choose a node from the bank
                one_idx.append(np.where(np.array(g_intermediate.x[-1].tolist()) == 1)[0][0] + g[iterator].num_nodes) # find the index of the one-hot encoding of the new node
                one_idx.append(removed_idx[iterator]) # find index of previous node new node was connected to
            else:
                one_idx.append(np.where(np.array(g_intermediate.x[-1].tolist()) == 1)[0][0])
                one_idx.append(removed_idx[iterator])

            for j in one_idx:
                node_actual[j] = 1
            node_loss = node_criterion(torch.reshape(node_prediction, (1, node_prediction.shape[0]))[0], node_actual)
            node_optim.zero_grad()
            node_loss.backward()
            node_optim.step()

            # Edge probability predictor
            edge_prediction = edge_model(g[iterator], bank, one_idx[0], one_idx[1])
            edge_loss = edge_criterion(edge_prediction, torch.tensor(removed_attr[iterator]))
            edge_optim.zero_grad()
            edge_loss.backward()
            edge_optim.step()
    # nx_draw(g_final)