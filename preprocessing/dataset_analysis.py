import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pandas as pd


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

"""
Save data about the Molecular Distribution
1. Number and Type of Each Element
2. Number of Total Atoms
3. Number of Non-Hydrogen Atoms
4. Charge
"""

z_smiles = pd.read_csv('/Users/maxwellchen/Desktop/Drug_Design/Data/ZINC_in_vitro/in-vitro.csv')
z_smiles = z_smiles["smiles"].to_numpy()
print(np.shape(z_smiles))

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
    e, n = find_element_distribution_from_smiles(smile)
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
