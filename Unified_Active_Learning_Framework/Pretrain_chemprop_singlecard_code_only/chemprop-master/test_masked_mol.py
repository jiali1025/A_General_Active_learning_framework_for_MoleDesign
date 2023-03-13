from chemprop.features import MolGraph, MolGraph_mask, BatchMolGraph, BatchMolGraph_mask, mol2graph_mask
from rdkit import Chem

if __name__ == '__main__':
    '''
    Single mol test
    '''

    # test_mol = 'C1=C(C(=C(C(=C1Cl)Cl)CC2=C(C(=CC(=C2Cl)Cl)Cl)O)O)Cl'
    #
    # test_mol_object = Chem.MolFromSmiles(test_mol)
    #
    # test_mol_graph = MolGraph(test_mol_object)
    #
    # extra_mask_index = [0,3,4]
    #
    # test_mask_mol_graph = MolGraph_mask(mol=test_mol_object, mask_auto=True, mask_percentage=0.2)
    # test_mask_mol_graph_1 = MolGraph_mask(mol=test_mol_object, extra_mask_idx=extra_mask_index)
    #
    # print(test_mol_graph.f_atoms)
    #
    # print(test_mask_mol_graph.f_atoms)
    #
    # print(test_mask_mol_graph.masked_atom_index)
    #
    # print(test_mask_mol_graph_1.f_atoms)
    #
    # print(test_mask_mol_graph_1.masked_atom_index)
    #
    # print(test_mask_mol_graph_1.extra_mask_idx)
    '''
    Batch mol test
    '''

    test_mol_list = ['C1=C(C(=C(C(=C1Cl)Cl)CC2=C(C(=CC(=C2Cl)Cl)Cl)O)O)Cl', 'C1=CC=C2C(=C1)C=CC3=C2C=CC4=C3C=CN=C4',
                     'C1=CC=C2C(=C1)C=C(C(=O)O2)C(=O)O','CC(C)(C)C1=CC(=NN1)C2=CC=CC=N2']

    test_mol_list_object = [Chem.MolFromSmiles(test_mol) for test_mol in test_mol_list]

    test_mol_graph = [MolGraph(test_mol_object) for test_mol_object in test_mol_list_object]

    test_mol_batch = BatchMolGraph(test_mol_graph)

    test_mask_mol_graph = [MolGraph_mask(mol=test_mol_object, mask_auto=True, mask_percentage=0.4) for test_mol_object in test_mol_list_object]

    test_mask_mol_batch = BatchMolGraph_mask(test_mask_mol_graph)
    extra_mask_idx_list = [[0,4,5,6],[4,5,6,7],[1,2,3],[3,4]]
    batch_mask = mol2graph_mask(test_mol_list, extra_mask_idx_list)

    print(test_mol_batch.get_components()[-2])
    print(test_mask_mol_batch.get_components()[-3])

    print(test_mask_mol_batch.get_components()[-1])
    print(batch_mask.get_components()[-1])




