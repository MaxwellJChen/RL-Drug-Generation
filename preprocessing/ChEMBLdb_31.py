import pandas
import rdkit


# Create list of SMILES from ChEMBL data
filepath = "/Users/maxwellchen/Desktop/Drug_Design/Data/ChEMBL_31/chembl_31_chemreps.txt"
df = pandas.read_csv(filepath, sep = '/t')
