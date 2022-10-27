from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdmolops import GetFormalCharge
import torch
import torch_geometric
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.nn import NNConv, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
import pandas as pd
import pickle

"""Element Analysis"""
def split_formula(smiles):  # Split formula into a list of units, account for charge as well
    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    f = []
    encoded = [identify_char(c) for c in formula]
    unit = ""

    charge = GetFormalCharge(mol)

    for i in range(len(formula)):
        c = encoded[i]
        if i != 0:
            p = encoded[i - 1]
            if (p == 'U' and c == 'L') or (p == 'L' and c == 'L') or (p == 'N' and c == 'N'):
                unit += formula[i]
            else:
                f.append(unit)
                unit = formula[i]
        else:
            unit = formula[i]
        if i + 1 == len(formula):
            f.append(unit)

    # Deal with the charge of the element in the formula
    if charge == 1 or charge == -1: # Charge of 1/-1 means there's just a sign a the end of the formula (+/-)
        f = f[:len(f) - 1]
    elif charge > 1 or charge < -1: # More than one indicates "2+" for instance
        f = f[:len(f) - 2]
    elif charge == 0: # No charge, keep the same
        f = f
    else:
        f = -1

    return f

def identify_char(c):
    """
    Identify if a char is upper case, lower case, or a number, helper function for split_formula
    """
    ascii = ord(c)
    if ascii in range(48, 58):
        return 'N'
    elif ascii in range(65, 91):
        return 'U'
    elif ascii in range(97, 123):
        return 'L'
    else:
        return 'Wtf'

def identify_unit(unit):
    """
    Return whether a unit is an element or number
    A helper function for find_element_distribution_from_smiles
    """
    if ord(unit[0]) <= 57 and ord(unit[0]) >= 48:
        return 'N'
    else:
        return 'E'

def find_element_distribution_from_smiles(smiles):
    """
    Accepts a SMILES string and outputs the distribution of elements

    :param smiles: A SMILES string
    :return: Two lists holding elements and their respective counts in the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    f = split_formula(smiles)

    # Add 1s into their proper spots in the split molecular formula
    # If two elements are side-by-side or if an element is at the end of the formula without anything

    if identify_unit(f[-1]) == 'E': # Add a 1 to the end if the last unit is an element
        f.append('1')

    one_spots = []
    for i in range(len(f)):
        if i + 1 != len(f):
            if identify_unit(f[i]) == 'E' and identify_unit(f[i + 1]) == 'E':
                one_spots.append(i)
            else:
                continue
    one_spots.sort(reverse = True)
    one_spots = [i + 1 for i in one_spots]
    for one in one_spots:
        f.insert(one, '1')

    # Break the list into pairs which should work out perfectly
    element = []
    number = []
    for i in range(0, len(f), 2):
        element.append(f[i])
        number.append(int(f[i + 1]))
    return element, number


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

def get_atom_features(atom,
                      use_chirality=True,
                      hydrogens_implicit=True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    permitted_list_of_atoms = ['C', 'Cl', 'N', 'O', 'Br', 'S', 'P', 'F', 'I', 'B', 'Si', 'Sn', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

def get_bond_features(bond,
                      use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


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