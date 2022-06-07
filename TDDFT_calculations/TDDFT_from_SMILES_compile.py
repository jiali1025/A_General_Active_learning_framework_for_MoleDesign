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
    ephemdir: str = os.path.dirname('/pool001/skverma/ephemeral/')
    """Defaults to SCOP_DB directory"""
    log: str = 'warning'

def compile_data():
    exDataFile = 'TDDFT_' + smiles_filename
    with open(exDataFile, 'w') as file:
        file.write('SMILES,TDDFT_S1,TDDFT_T1\n')
    allData = {}
    for ID, smiles in tqdm(enumerate(df.iloc[:, smiInd]), total=len(df.iloc[:, smiInd])):
        IDstr = 'ID' + '%09d' % ID
        IDdir = os.path.join(xyzdir, IDstr)
        filename = IDstr
        print(os.path.join(xyzdir, filename, 'excitedstate_S1.out'))
        if os.path.isfile(os.path.join(xyzdir, filename, 'excitedstate_S1.out')) and \
                os.path.isfile(os.path.join(xyzdir, filename, 'excitedstate_T1.out')):
            with open(os.path.join(xyzdir, filename, 'excitedstate_S1.out'), 'r') as file:
                S1data = file.readline()
            with open(os.path.join(xyzdir, filename, 'excitedstate_T1.out'), 'r') as file:
                T1data = file.readline()
            try:
                S1Ex = float(re.search('(.+?)eV', S1data).group(1).split()[-1])
            except:
                continue
            try:
                T1Ex = float(re.search('(.+?)eV', T1data).group(1).split()[-1])
            except:
                continue
            if S1Ex is not None and T1Ex is not None:
                if 0 < S1Ex < 10 and 0 < T1Ex < 10:
                    with open(exDataFile, 'a') as file:
                        file.write(smiles + ',' + str(S1Ex) + ',' + str(T1Ex) + '\n')


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
smiles_filename = os.path.basename(smiles_path)
smiles_file = os.path.splitext(smiles_filename)[0]
xyzdirname = 'xyzfiles_TDDFT_' + smiles_file
comdirname = 'comfiles_TDDFT_' + smiles_file
TDDFT_multi_dirname = os.path.join(ephemdir, 'TDDFT_multi')
xyzdir = os.path.join(ephemdir, xyzdirname)
comdir = os.path.join(ephemdir, xyzdirname)
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
print('compiling data')
compile_data()

print('successfully completed!')
with open('progress.out', 'a') as file:
    file.write('successfully completed!\n')
f.close()
