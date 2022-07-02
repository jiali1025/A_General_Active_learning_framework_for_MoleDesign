import os
import pandas as pd


data = pd.read_csv("smiles.csv", encoding='utf-8').iloc[:, 0]
data = data.dropna(axis=0, how='any') # read_csv
count = len(data) # lines of txt file
batch = 100 # set numbers of smiles in a batch for saving

with open("smiles.txt", "w+") as f:
    for i in range(0, count):
        f.write(data[i] + "\n")
        if ((i+1) % batch == 0) | ((i+1) >= count):
            f.close()
            os.system("aizynthcli --config config.yml --smiles smiles.txt")
            pd.read_hdf("output.hdf5", "table")[["target", "is_solved"]].to_csv("label.csv", mode="a", header=False, index=None, sep = ",")
            f = open("smiles.txt", "w+")


