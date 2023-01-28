import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

import rdkit
from rdkit import Chem
from poly_hgraph import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
# parser.add_argument('--test', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--nsample', type=int, default=10000)

args = parser.parse_args()

# args.test = [line.strip("\r\n ") for line in open(args.test)]
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
vocab = [ele for ele in vocab if ele != []]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x,y) for x,y,_ in vocab])

model = HierVAE(args).cuda()


model.load_state_dict(torch.load(args.model)[0])
model.eval()

# dataset = MoleculeDataset(args.test, args.vocab, args.atom_vocab, args.batch_size)
# loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

torch.manual_seed(args.seed)
random.seed(args.seed)
output_list = []
# total, acc = 0, 0

# with torch.no_grad():
#     for i,batch in enumerate(loader):
#         orig_smiles = args.test[args.batch_size * i : args.batch_size * (i + 1)]
#         dec_smiles = model.reconstruct(batch)
#         for x, y in zip(orig_smiles, dec_smiles):
#             print(x, y)

with torch.no_grad():
    for _ in tqdm(range(args.nsample // args.batch_size)):
        try:
            smiles_list = model.sample(args.batch_size, greedy=True)
            for _, smiles in enumerate(smiles_list):
                output_list.append(smiles)
        except Exception:
            pass


    np_list = np.array(output_list)
    np.savetxt('/home/pengfei/projects/AL_framework/hgraph2graph/polymers/Generated_results/complex_1226_test.txt', np_list, fmt='%s', delimiter=',')
