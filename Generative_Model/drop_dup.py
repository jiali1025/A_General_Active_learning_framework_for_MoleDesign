from rdkit import Chem
import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')

# 定义一个函数，用于将SMILES转换为Canonical SMILES
def to_canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # 确保SMILES是有效的
        return Chem.MolToSmiles(mol, canonical=True)
    return None

# 将SMILES转换为Canonical SMILES
df1['Canonical_SMILES'] = df1['SMILES'].apply(to_canonical)
df2['Canonical_SMILES'] = df2['SMILES'].apply(to_canonical)

# 找出第二个文件中出现的Canonical SMILES字符串
canonical_smiles_to_exclude = set(df2['Canonical_SMILES'])

# 从第一个DataFrame中排除这些Canonical SMILES
df1_filtered = df1[~df1['Canonical_SMILES'].isin(canonical_smiles_to_exclude)]

# 可以选择移除'Canonical_SMILES'这一列
df1_filtered = df1_filtered.drop(columns=['Canonical_SMILES'])

# 保存结果到新的CSV文件
df1_filtered.to_csv('filtered_file.csv', index=False)
