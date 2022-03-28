'''
Here is the code to calculate the molecular weight statistic of the whole design space contained in the
xtb_ML_calcs_compiled_noTest data which getting about 485k data calculated from ML-xtb method.
'''
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
'''
global variable to store the path of the target folders change it to your local path
'''
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data'
figure_save_folder = '/Users/lijiali/Desktop/Active leanring Framework/Figures'
df = pd.read_csv(data_folder_path+'/whole_design_space/xtb_ML_calcs_compiled_noTest.csv')
print('Number of calculated molecules: ' + str(df.shape[0]))

def calculate_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    weight = Descriptors.MolWt(mol)
    return weight
mol_weight_list = []

for i in range(len(df)):
    mol_weight_list.append(calculate_mol_weight(df.iloc[i,1]))
plt.hist(x=mol_weight_list,bins=20,
        color="steelblue",
        edgecolor="black")


plt.xlabel("ratio")
plt.ylabel("frequency")
plt.title("whole_design_space_molW")
plt.savefig(figure_save_folder + '/whole_design_space_molW.png')
