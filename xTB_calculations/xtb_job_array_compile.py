import numpy as np
import pandas as pd
from tqdm import tqdm
from tap import Tap
import os
import logging


class SimpleArgParse(Tap):
    # xyzdir: str
    # """Directory with xyz files to compute"""
    smiles_path: str
    # basedir: str = os.path.dirname(os.path.realpath(__file__))
    basedir: str = os.path.dirname('/home/gridsan/sverma/xtb_run/xtb_multi_files/')
    ephemdir: str = os.path.dirname('/home/gridsan/sverma/ephemeral/')
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
basedir = args.basedir
ephemdir = args.ephemdir
smiles_path = args.smiles_path
smiles_filename = os.path.basename(smiles_path)
smiles_file = os.path.splitext(smiles_filename)[0]
xyzdirname = 'xyzfiles_' + smiles_file
xtb_multi_dirname = os.path.join(ephemdir, 'xtb_multi')
xyzdir = os.path.join(ephemdir, xyzdirname)
resultsdirname = xyzdirname.replace('xyzfiles_', 'xtbresults_')
resultsdir = os.path.join(ephemdir, resultsdirname)
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

exDataFile = 'xTB_' + smiles_filename
with open(exDataFile, 'w') as file:
    file.write('SMILES,xtbS1,xtbT1\n')

for ID, smiles in tqdm(enumerate(df.iloc[:, smiInd]), total=len(df.iloc[:, smiInd])):
    IDstr = 'ID' + '%09d' % ID
    IDdir = os.path.join(xyzdir, IDstr)
    S1file = os.path.join(xyzdir, IDstr, 'S1Ex.dat')
    T1file = os.path.join(xyzdir, IDstr, 'T1Ex.dat')
    if not os.path.isfile(S1file) or not os.path.isfile(T1file):
        continue
    with open(S1file, 'r') as file:
        S1data = file.readline()
    with open(T1file, 'r') as file:
        T1data = file.readline()
    try:
        S1Ex = float(S1data.split()[1])
    except IndexError:
        S1Ex = None
    except ValueError:
        S1Ex = None
    try:
        T1Ex = float(T1data.split()[1])
    except IndexError:
        T1Ex = None
    except ValueError:
        T1Ex = None
    if S1Ex is not None and T1Ex is not None:
        with open(exDataFile, 'a') as file:
            file.write(smiles + ',' + str(S1Ex) + ',' + str(T1Ex) + '\n')

print('done compiling')
