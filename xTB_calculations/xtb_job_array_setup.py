import os
from tap import Tap
import logging
import pandas as pd
from tqdm import tqdm


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
if not os.path.isdir(xyzdir):
    os.mkdir(xyzdir)
resultsdirname = xyzdirname.replace('xyzfiles_', 'xtbresults_')
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

os.system('sbatch --export SMILES_PATH=' + smiles_path + ' job_xtb_multi_array')
