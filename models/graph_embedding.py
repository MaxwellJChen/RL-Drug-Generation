"""Convert ZINC_in_vitro SMILES into simply embedded graphs"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
import pandas as pd
import pickle

"""Embedding Methods"""
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    permitted_list_of_atoms = ['C', 'Cl', 'N', 'O', 'Br', 'S', 'P', 'F', 'I', 'B', 'Si', 'Sn', 'Unknown']

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc

    return np.array(atom_feature_vector)

def get_bond_features(bond):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    return np.array(bond_type_enc)

def graph_from_smiles(x_smiles):
    """
    Inputs:
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)

    Outputs:
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    """

    data_list = []

    for i in range(len(x_smiles)):
        print(i)
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(x_smiles[i])
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct label tensor

        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF))
    return data_list

z_smiles = pd.read_csv('/Users/maxwellchen/Desktop/Drug_Design/Data/ZINC_in_vitro/in-vitro.csv')
z_smiles = z_smiles["smiles"].to_numpy()
print(np.shape(z_smiles))

"""
Compute Graphs for In-Vitro data
"""

z_smiles = np.unique(z_smiles)
# True total of 306253 smiles/molecules

"""
0 – 0:50000
1 – 50000:100000
2 – 100000:150000
3 – 150000:200000
4 – 200000:250000
5 – 250000:306253
"""

n = 5
graphs = graph_from_smiles(z_smiles[250000:306253])

with open(f'zinc_graphs_{n}', 'wb') as file:
    pickle.dump(graphs, file)
