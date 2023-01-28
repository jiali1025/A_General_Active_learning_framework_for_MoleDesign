import rdkit
import rdkit.Chem as Chem
import copy
import torch

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = [x for x in smiles_list] #copy
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)

class PairVocab(object):

    def __init__(self, smiles_pairs, cuda=True):
        cls = list(zip(*smiles_pairs))[0]
        self.hvocab = sorted( list(set(cls)) )
        self.hmap = {x:i for i,x in enumerate(self.hvocab)}

        self.vocab = [tuple(x) for x in smiles_pairs] #copy
        self.inter_size = [count_inters(x[1]) for x in self.vocab]
        self.vmap = {x:i for i,x in enumerate(self.vocab)}

        self.mask = torch.zeros(len(self.hvocab), len(self.vocab))
        for h,s in smiles_pairs:
            hid = self.hmap[h]
            idx = self.vmap[(h,s)]
            self.mask[hid, idx] = 1000.0

        if cuda: self.mask = self.mask.cuda()
        self.mask = self.mask - 1000.0
            
    def __getitem__(self, x):
        assert type(x) is tuple
        return self.hmap[x[0]], self.vmap[x]

    def get_smiles(self, idx):
        return self.hvocab[idx]

    def get_ismiles(self, idx):
        return self.vocab[idx][1] 

    def size(self):
        return len(self.hvocab), len(self.vocab)

    def get_mask(self, cls_idx):
        return self.mask.index_select(index=cls_idx, dim=0)

    def get_inter_size(self, icls_idx):
        return self.inter_size[icls_idx]

COMMON_ATOMS = [('Na', 0), ('Hg', 0), ('S', 1), ('U', 0), ('Pt', 0), ('Sn', 0), ('As', 0), ('Au', 0), ('K', -1), ('C', -1), ('B', 0), ('Pd', 0), ('Ru', 0), ('Se', -1), ('Mg', 0), ('Al', -3), ('P', 1), ('Si', 0), ('Zn', 0), ('W', 0), ('Lu', 0), ('H', 1), ('K', 0), ('C', 0), ('Se', 0), ('No', 0), ('Hf', 0), ('N', 1), ('Cl', 0), ('Li', -3), ('Ge', 0), ('Sm', 0), ('Ir', -1), ('Rf', 0), ('Ir', -2), ('F', 0), ('Ra', 0), ('S', -1), ('Rh', 0), ('Pb', 0), ('O', 1), ('Sb', 0), ('Ir', 0), ('Th', 0), ('S', 0), ('Sr', 0), ('P', -1), ('Te', 0), ('Rb', 0), ('Ag', 0), ('Fm', 0), ('Na', 1), ('P', 0), ('Bi', 0), ('Cs', -1), ('Sc', 0), ('I', 0), ('Cs', 0), ('Ni', 0), ('Se', 1), ('Ca', 0), ('Re', 0), ('Mn', 0), ('Li', -2), ('Li', -1), ('Br', -1), ('Nd', 0), ('Li', 0), ('N', -1), ('H', 0), ('Br', 0), ('Mg', -1), ('Co', 0), ('O', -1), ('Ti', 0), ('N', 0), ('Fe', -1), ('C', 1), ('Ba', 1), ('Al', 0), ('Cu', -1), ('O', 0), ('Be', 0), ('V', 0), ('Fe', 0), ('Zr', -1), ('Cu', 0), ('Rg', 0), ('Ga', 0), ('B', -1), ('Zr', 0)]
common_atom_vocab = Vocab(COMMON_ATOMS)

def count_inters(s):
    mol = Chem.MolFromSmiles(s)
    inters = [a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
    return max(1, len(inters))


