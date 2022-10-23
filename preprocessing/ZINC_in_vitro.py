"""Convert ZINC_in_vitro SMILES into simply embedded graphs"""

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
from rdkit.Chem.rdmolops import GetFormalCharge

import methods

z_smiles = pd.read_csv('/Users/maxwellchen/Desktop/Drug_Design/Data/ZINC_in_vitro/in-vitro.csv')
z_smiles = z_smiles["smiles"].to_numpy()
print(len(z_smiles))



"""
Save Data about the Molecular Distribution
1. Number and Type of Each Element
2. Number of Total Atoms
3. Number of Non-Hydrogen Atoms
4. Charge
"""

"""
elements = []
e_numbers = []

charges = []
c_numbers = []

total_atoms = []
t_numbers = []

heavy_atoms = []
h_numbers = []
for i in range(len(z_smiles)):
    print(i)
    smile = z_smiles[i]

    # Element Distribution
    e, n = methods.find_element_distribution_from_smiles(smile)
    for i in range(len(e)):
        if e[i] in elements:
            idx = elements.index(e[i])
            e_numbers[idx] += n[i]
        else:
            elements.append(e[i])
            e_numbers.append(n[i])

    # Charge
    mol = Chem.MolFromSmiles(smile)
    charge = GetFormalCharge(mol)
    if charge in charges:
        idx = charges.index(charge)
        c_numbers[idx] += 1
    else:
        charges.append(charge)
        c_numbers.append(1)

    # Number of Total Atoms
    total = sum(n)
    if total in total_atoms:
        idx = total_atoms.index(total)
        t_numbers[idx] += 1
    else:
        total_atoms.append(total)
        t_numbers.append(1)

    # Number of Heavy Atoms
    heavy = mol.GetNumAtoms()
    if heavy in heavy_atoms:
        idx = heavy_atoms.index(heavy)
        h_numbers[idx] += 1
    else:
        heavy_atoms.append(heavy)
        h_numbers.append(1)


np.save(f'elements', [elements, e_numbers])
np.save(f'charges', [charges, c_numbers])
np.save(f'total_atoms', [total_atoms, t_numbers])
np.save(f'heavy_atoms', [heavy_atoms, h_numbers])
"""

"""Combine Saved Molecular Data"""
# Charges
# charge = []
# c_number = []
# charge1 = np.load("/Users/maxwellchen/PycharmProjects/Drug_Design/RL-Drug-Generation/preprocessing/charges1.npy")
# charge2 = np.load("/Users/maxwellchen/PycharmProjects/Drug_Design/RL-Drug-Generation/preprocessing/charges2.npy")
# charge3 = np.load("/Users/maxwellchen/PycharmProjects/Drug_Design/RL-Drug-Generation/preprocessing/charges3.npy")



"""Use Embedding Methods to Embed SMILES into Graphs"""