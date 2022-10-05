import os
import sys
import json
import time
import logging

import pandas as pd
import numpy as np

from tap import Tap
from tqdm import tqdm

from openbabel import openbabel as ob
from openbabel import pybel


class SimpleArgParse(Tap):
    # xyzdir: str
    # """Directory with xyz files to compute"""
    smiles_path: str
    index: int
    nodes: int
    # basedir: str = os.path.dirname(os.path.realpath(__file__))
    basedir: str = os.path.dirname('/home/gridsan/sverma/xtb_run/xtb_multi_files/')
    ephemdir: str = os.path.dirname('/home/gridsan/sverma/ephemeral/')
    """Defaults to SCOP_DB directory"""
    log: str = 'warning'


def gen_sdf(smiles, moldir):
    if os.path.isfile(os.path.join(xyzdir, moldir, moldir + '.xyz')):
        return
    if os.path.isfile(os.path.join(xyzdir, moldir, moldir + '.sdf')):
        return
    mol2 = pybel.readstring('smi', smiles)
    gen3d = ob.OBOp.FindType("gen3D")
    gen3d.Do(mol2.OBMol, "--medium")
    mol2.write('sdf', filename=os.path.join(xyzdir, moldir, moldir + '.sdf'), overwrite=True)
    if not os.path.isfile(os.path.join(xyzdir, moldir, moldir + '.sdf')):
        mol2.write('sdf', filename=os.path.join(xyzdir, moldir, moldir + '.sdf'), overwrite=True)
    with open(os.path.join(xyzdir, moldir, moldir + '.sdf'), 'r') as file:
        data = file.readlines()
    doGen2D = False
    for line in data:
        if '0.0000    0.0000    0.0000' in line:
            doGen2D = True
            break
    if (doGen2D):
        mol2 = pybel.readstring('smi', smiles)
        mol2.addh()
        gen2d = ob.OBOp.FindType("gen2D")
        gen2d.Do(mol2.OBMol, "--medium")
        mol2.write('sdf', filename=os.path.join(xyzdir, moldir, moldir + '.sdf'), overwrite=True)
    with open(os.path.join(xyzdir, moldir, moldir + '.sdf'), 'r') as file:
        data = file.readlines()
    doLocalOpt = False
    for line in data:
        if '0.0000    0.0000    0.0000' in line:
            doLocalOpt = True
            break
    if (doLocalOpt):
        mol2 = pybel.readstring('smi', smiles)
        mol2.addh()
        mol2.make3D()
        mol2.write('sdf', filename=os.path.join(xyzdir, moldir, moldir + '.sdf'), overwrite=True)


def gen_xyz(filename):
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '.xyz')):
        return
    file = filename + ".sdf"
    recompute = True
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '.xyz')):
        recompute = False
    if recompute:
        sdffile = os.path.join(xyzdir, filename, file)
        mol = next(pybel.readfile('sdf', sdffile))
        chrg = str(mol.charge)
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  ' && xtb ' + file + ' --opt tight --gfn 2 --charge ' + chrg + ' --gbsa benzene --norestart > '
                  + 'output_xtb.out 2>&1)')
        if not os.path.isfile(os.path.join(xyzdir, filename, 'xtbopt.sdf')):
            os.system('(cd ' + os.path.join(xyzdir, filename) +
                      ' && xtb ' + file + ' --opt tight --gfn 1 --charge ' + chrg + ' --gbsa benzene --norestart > '
                      + 'output_xtb.out 2>&1)')
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  ' && obabel -isdf ' + 'xtbopt.sdf' + ' -O xtbopt.xyz >> obabelout 2>&1)')

        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  ' && cp xtbopt.xyz ' + filename + '.xyz)')
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -type f -not -name 'ID*.xyz' -exec rm -rf {} \;)")
    else:
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -type f -not -name 'ID*.xyz' -exec rm -rf {} \;)")


def run_xtb_stda(filename):
    xyzfile = os.path.join(xyzdir, filename, filename + '.xyz')
    if not os.path.isfile(xyzfile):
        return
    if os.path.isfile(os.path.join(xyzdir, filename, 'S1Ex.dat')) and \
            os.path.isfile(os.path.join(xyzdir, filename, 'T1Ex.dat')):
        return
    try:
        mol = next(pybel.readfile('xyz', xyzfile))
    except:
        return
    chrg = str(mol.charge)
    os.system('(cd ' + os.path.join(xyzdir, filename) + ' && xtb ' +
              filename + '.xyz' + ' --opt tight --gfn 2 --charge ' + chrg + ' --gbsa benzene --norestart > '
              + 'output_xtb.out 2>&1)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              ' && xtb4stda ' + 'xtbopt.xyz -gbsa benzene > output_xtb4stda.out)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              ' && stda -xtb -e 10 > output_stda.out)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              ' && grep -A1 state output_stda.out | tail -1 > S1Ex.dat)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              ' && stda -xtb -t -e 10 > output_stda_trip.out)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              ' && grep -A1 state output_stda_trip.out | tail -1 > T1Ex.dat)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              ' && rm wfn.xtb && rm tda.dat)')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && find . -type f -not -name '*.dat' -a -not -name 'ID*.xyz' -exec rm -rf {} \;)")


def do_xtb_calcs(ID):
    IDstr = 'ID' + '%09d' % ID
    with open(os.path.join(xyzdir, IDstr, 'smiles.txt')) as file:
        smiles = file.readline()
    gen_sdf(smiles, IDstr)
    gen_xyz(IDstr)
    run_xtb_stda(IDstr)


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

nodes = args.nodes
node_index = args.index
df = pd.read_csv(smiles_path)
smiInd = -1
for index, col in enumerate(df.columns):
    if col.lower() == 'smiles':
        smiInd = index
total = len(df.iloc[:, smiInd])
# total = 10
mols_per_node = int(np.floor(total / nodes) + 1)
for i in np.arange((node_index - 1) * mols_per_node, node_index * mols_per_node):
    do_xtb_calcs(i)
