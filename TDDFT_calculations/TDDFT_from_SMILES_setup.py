import re
import json
import sys
import time
import logging
from tap import Tap
from rdkit.Chem import Descriptors
from rdkit import Chem
import os
from openbabel import openbabel as ob
from openbabel import pybel
import pandas as pd
from tqdm import tqdm
print('importing')

f = open('progress.out', 'a')
sys.stdout = f


class SimpleArgParse(Tap):
    # xyzdir: str
    # """Directory with xyz files to compute"""
    smiles_path: str
    create_dir: bool = False
    ephemdir: str = os.path.dirname('/pool001/skverma/ephemeral/')
    """Defaults to SCOP_DB directory"""
    log: str = 'warning'


print('setting up')
args = SimpleArgParse().parse_args()
levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}
logLevel = levels.get(args.log.lower())
ephemdir = args.ephemdir
smiles_path = args.smiles_path
create_dir = args.create_dir
smiles_filename = os.path.basename(smiles_path)
smiles_file = os.path.splitext(smiles_filename)[0]
xyzdirname = 'xyzfiles_TDDFT_' + smiles_file
comdirname = 'comfiles_TDDFT_' + smiles_file
TDDFT_multi_dirname = os.path.join(ephemdir, 'TDDFT_multi')
xyzdir = os.path.join(ephemdir, xyzdirname)
comdir = os.path.join(ephemdir, comdirname)
if not os.path.isdir(xyzdir):
    os.mkdir(xyzdir)
if not os.path.isdir(comdir):
    os.mkdir(comdir)
resultsdirname = xyzdirname.replace('xyzfiles_', 'TDDFTresults_')
resultsdir = os.path.join(ephemdir, resultsdirname)
if not os.path.isdir(resultsdir):
    os.mkdir(resultsdir)
errFiles = []
logging.root.setLevel(logLevel)
logging.basicConfig(level=logLevel)


with open(smiles_path, 'r') as file:
    data = file.readlines()
smiInd = -1
for index, val in enumerate(data[0].replace('\n', '').split(',')):
    if val.lower() == 'smiles':
        smiInd = index

print('reading csv')
df = pd.read_csv(smiles_path)
df.info()
if create_dir:
    print('creating directories')
    for ID, smiles in tqdm(enumerate(df.iloc[:, smiInd]), total=len(df.iloc[:, smiInd])):
        IDstr = 'ID' + '%09d' % ID
        IDdir = os.path.join(xyzdir, IDstr)
        if not os.path.isfile(os.path.join(IDdir, 'smiles.txt')):
            try:
                os.mkdir(IDdir)
            except FileExistsError:
                pass
            try:
                with open(os.path.join(IDdir, 'smiles.txt'), 'w') as file:
                    file.write(smiles)
            except TypeError:
                continue

os.system('sbatch --export SMILES_PATH=' + smiles_path + ' job_TDDFT_multi_array')