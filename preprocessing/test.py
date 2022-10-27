import numpy as np
import pickle

asdf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
graphs = []

"""
for i in range(6):
    print(i)
    with open(f'/Users/maxwellchen/PycharmProjects/Drug_Design/RL-Drug-Generation/preprocessing/zinc_graphs_{i}', 'rb') as file:
        b = pickle.load(file)
        file.close()
    if i == 0:
        graphs = b
        print(len(graphs))
    else:
        graphs += b
        print(len(graphs))
    print()

with open(f'zinc_graphs', 'wb') as file:
    pickle.dump(graphs, file)
    file.close()
"""
with open(f'/Users/maxwellchen/PycharmProjects/Drug_Design/RL-Drug-Generation/zinc_graphs', 'rb') as file:
    b = pickle.load(file)
    file.close()
print(len(b))
