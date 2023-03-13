from .featurization import MolGraph, Featurization_parameters, atom_features, bond_features, get_atom_fdim, get_bond_fdim
from typing import List, Tuple, Union
import numpy as np
from rdkit import Chem
import torch
from chemprop.rdkit import make_mol
from itertools import zip_longest
PARAMS = Featurization_parameters()
torch.manual_seed(666)

def mask_atom_features(atom_features: List[Union[bool, int, float]]) -> List[Union[bool, int, float]]:

    atom_features_len = len(atom_features)
    assert atom_features_len == PARAMS.ATOM_FDIM
    masked_atom_features = [0] * atom_features_len

    return masked_atom_features




class MolGraph_mask(MolGraph):
    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 mask_auto: bool = False,
                 mask_percentage: float = 0,
                 extra_mask_idx: List[Union[int, float]] = None,
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False, **kwargs):
        super(MolGraph_mask, self).__init__(mol)

        self.mask_percentage = mask_percentage
        self.mask_auto = mask_auto
        self.extra_mask_idx = extra_mask_idx
        self.masked_atom_index = None
        self.final_masked_atom_index = None
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h, self.is_adding_hs), make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs))
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs)

        if not self.is_reaction:
            # Get atom features
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in
                                    zip(self.f_atoms, atom_features_extra)]


            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')

            if self.mask_auto:

                self.masked_atom_index = torch.randperm(len(self.f_atoms))[:int(len(self.f_atoms)*self.mask_percentage)]
                self.masked_atom_index = self.masked_atom_index.tolist()
                self.masked_atom_index = sorted(self.masked_atom_index)

                self.f_atoms = [self.f_atoms[idx] if idx not in self.masked_atom_index else mask_atom_features(self.f_atoms[idx]) for idx in range(len(self.f_atoms))]

                self.final_masked_atom_index = self.masked_atom_index

            else:

                self.f_atoms = [self.f_atoms[idx] if idx not in self.extra_mask_idx else mask_atom_features(self.f_atoms[idx]) for idx in range(len(self.f_atoms))]

                self.final_masked_atom_index = self.extra_mask_idx


            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra bond features')

        else:  # Reaction mode
            if atom_features_extra is not None:
                raise NotImplementedError('Extra atom features are currently not supported for reactions')
            if bond_features_extra is not None:
                raise NotImplementedError('Extra bond features are currently not supported for reactions')

            mol_reac = mol[0]
            mol_prod = mol[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)

            # Get atom features
            if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_prod']:
                # Reactant: regular atom features for each atom in the reactants, as well as zero features for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [
                    atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio]

                # Product: regular atom features for each atom that is in both reactants and products (not in rio), other atom features zero,
                # regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(
                    mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features_zeros(atom) for atom in mol_reac.GetAtoms()] + [
                                   atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            else:  # balance
                # Reactant: regular atom features for each atom in the reactants, copy features from product side for atoms that are only in the products (indices in pio)
                f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [
                    atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

                # Product: regular atom features for each atom that is in both reactants and products (not in rio), copy features from reactant side for
                # other atoms, regular features for atoms that are only in the products (indices in pio)
                f_atoms_prod = [atom_features(
                    mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features(atom) for atom in mol_reac.GetAtoms()] + [
                                   atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

            if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                f_atoms_diff = [list(map(lambda x, y: x - y, ii, jj)) for ii, jj in zip(f_atoms_prod, f_atoms_reac)]
            if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                self.f_atoms = [x + y[PARAMS.MAX_ATOMIC_NUM + 1:] for x, y in zip(f_atoms_reac, f_atoms_prod)]
            elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                self.f_atoms = [x + y[PARAMS.MAX_ATOMIC_NUM + 1:] for x, y in zip(f_atoms_reac, f_atoms_diff)]
            elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                self.f_atoms = [x + y[PARAMS.MAX_ATOMIC_NUM + 1:] for x, y in zip(f_atoms_prod, f_atoms_diff)]
            self.n_atoms = len(self.f_atoms)
            n_atoms_reac = mol_reac.GetNumAtoms()

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    if a1 >= n_atoms_reac and a2 >= n_atoms_reac:  # Both atoms only in product
                        bond_prod = mol_prod.GetBondBetweenAtoms(pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac])
                        if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                            bond_reac = bond_prod
                        else:
                            bond_reac = None
                    elif a1 < n_atoms_reac and a2 >= n_atoms_reac:  # One atom only in product
                        bond_reac = None
                        if a1 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], pio[a2 - n_atoms_reac])
                        else:
                            bond_prod = None  # Atom atom only in reactant, the other only in product
                    else:
                        bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                        if a1 in ri2pi.keys() and a2 in ri2pi.keys():
                            bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], ri2pi[
                                a2])  # Both atoms in both reactant and product
                        else:
                            if self.reaction_mode in ['reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance']:
                                if a1 in ri2pi.keys() or a2 in ri2pi.keys():
                                    bond_prod = None  # One atom only in reactant
                                else:
                                    bond_prod = bond_reac  # Both atoms only in reactant
                            else:
                                bond_prod = None  # One or both atoms only in reactant

                    if bond_reac is None and bond_prod is None:
                        continue

                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    if self.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
                        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
                    if self.reaction_mode in ['reac_prod', 'reac_prod_balance']:
                        f_bond = f_bond_reac + f_bond_prod
                    elif self.reaction_mode in ['reac_diff', 'reac_diff_balance']:
                        f_bond = f_bond_reac + f_bond_diff
                    elif self.reaction_mode in ['prod_diff', 'prod_diff_balance']:
                        f_bond = f_bond_prod + f_bond_diff
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2

class BatchMolGraph_mask:
    """
    A :class:`BatchMolGraph_mask` represents the graph structure and featurization of a batch of masked molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`mask_scope`: A list of index of all the masked atoms inside the batch mol
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph_mask]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph_mask`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
        self.is_reaction = mol_graphs[0].is_reaction
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=self.overwrite_default_atom_features,
                                       is_reaction=self.is_reaction)
        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                      overwrite_default_atom=self.overwrite_default_atom_features,
                                      is_reaction=self.is_reaction)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.mask_scope = [] # list of masked index for the batch

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.mask_scope.extend(map(lambda x: x + self.n_atoms, mol_graph.final_masked_atom_index))
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]],
                                                                   List[Union[int, float]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`
        * :code:`mask_scope`


        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.mask_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph_mask(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              extra_mask_idx_list: List[List[Union[int, float]]] = None,
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph_mask:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param extra_mask_idx_list: A list of list of mask_idx_list for each mols
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph_mask([MolGraph_mask(mol,False,0,extra_mask_idx, af, bf,
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, extra_mask_idx, af, bf
                          in zip_longest(mols, extra_mask_idx_list, atom_features_batch, bond_features_batch)])


