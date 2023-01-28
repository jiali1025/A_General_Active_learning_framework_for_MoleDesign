import sys
import argparse 
from collections import Counter
from poly_hgraph import *
from rdkit import Chem
from multiprocessing import Pool
import numpy as np

def process(data):
    vocab = set()
    problem_data = []
    for line in data:
        s = line.strip("\r\n ")
        try:
            hmol = MolGraph(s)
            for node,attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                vocab.add( attr['label'] )
                for i,s in attr['inter_label']:
                    vocab.add( (smiles, s) )
        except Exception:
            problem_data.append(s)
            pass
    return vocab,problem_data


def fragment_process(data):
    counter = Counter()
    problem_frag_data = []
    for smiles in data:
        try:
            mol = get_mol(smiles)
            fragments = find_fragments(mol)
            for fsmiles, _ in fragments:
                counter[fsmiles] += 1
        except Exception:
            problem_frag_data.append(smiles)
            pass

    return counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frequency', type=int, default=200)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    data_file = open("/home/lijiali1025/projects/AL_framework/data/Know_emit_sens_space/final_known_space.csv", "r")
    data = data_file.read()
    data = data.split("\n")
    print('raw_data_amount: '+ str(len(data)))
    data = list(set(data))
    # data = data[:100]
    print('without_duplicate_data_amount: ' + str(len(data)))
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]


    pool = Pool(args.ncpu)
    counter_list = pool.map(fragment_process, batches)
    counter = Counter()
    for cc in counter_list:
        counter += cc
    
    fragments = [fragment for fragment,cnt in counter.most_common() if cnt >= args.min_frequency]
    MolGraph.load_fragments(fragments)

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab_raw = [t[0] for t in vocab_list]
    problem_raw = [t[1] for t in vocab_list]
    vocab = [(x, y) for vocab in vocab_raw for x, y in vocab]
    problem = [p for problem in problem_raw for p in problem]
    vocab = list(set(vocab))
    vocab = sorted(vocab)
    vocab = np.array(vocab)
    problem = np.array(problem)

    fragments = set(fragments)
    final_vocab = []
    for x,y in vocab:
        cx = Chem.MolToSmiles(Chem.MolFromSmiles(x))  # dekekulize
        final_vocab.append((x,y,cx in fragments))

    final_vocab = np.array(final_vocab)

    np.savetxt('/home/lijiali1025/projects/AL_framework/data/Know_emit_sens_space/known_vocab_complex_min_200_0107.txt', final_vocab,fmt='%s', delimiter=',')
    np.savetxt('/home/lijiali1025/projects/AL_framework/data/Know_emit_sens_space/known_vocab_complex_min_200_0107_error.txt',problem, fmt='%s', delimiter=',')


