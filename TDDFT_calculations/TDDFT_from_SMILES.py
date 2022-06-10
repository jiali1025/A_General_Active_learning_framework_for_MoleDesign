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
import numpy as np
print('importing')

f = open('progress.out', 'a')
sys.stdout = f


class SimpleArgParse(Tap):
    # xyzdir: str
    # """Directory with xyz files to compute"""
    smiles_path: str
    index: int
    nodes: int
    ephemdir: str = os.path.dirname('/pool001/skverma/ephemeral/')
    """Defaults to SCOP_DB directory"""
    log: str = 'warning'


# generate 3D coordinates with openbabel
def gen_sdf(smiles, filename):
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '.xyz')):
        return
    if os.path.isfile(os.path.join(xyzdir, filename + '.sdf')):
        return
    mol2 = pybel.readstring('smi', smiles)
    gen3d = ob.OBOp.FindType("gen3D")
    gen3d.Do(mol2.OBMol, "--medium")
    mol2.write('sdf', filename=os.path.join(xyzdir, filename + '.sdf'), overwrite=True)
    if not os.path.isfile(os.path.join(xyzdir, filename + '.sdf')):
        mol2.write('sdf', filename=os.path.join(xyzdir, filename + '.sdf'), overwrite=True)
    with open(os.path.join(xyzdir, filename + '.sdf'), 'r') as file:
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
        mol2.write('sdf', filename=os.path.join(xyzdir, filename + '.sdf'), overwrite=True)
    with open(os.path.join(xyzdir, filename + '.sdf'), 'r') as file:
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
        mol2.write('sdf', filename=os.path.join(xyzdir, filename + '.sdf'), overwrite=True)


# optimize initial 3D coordinates with xTB
def gen_xyz(filename, isFile=True):
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '.xyz')):
        return
    file = filename + ".sdf"
    recompute = True
    try:
        os.mkdir(os.path.join(xyzdir, filename))
    except:
        if os.path.isfile(os.path.join(xyzdir, filename, filename + '.xyz')):
            # recompute = True
            recompute = False
    if recompute:
        sdffile = os.path.join(xyzdir, filename, file)
        if isFile:
            try:
                os.mkdir(os.path.join(xyzdir, filename))
            except:
                pass
            os.system('mv ' + os.path.join(xyzdir, file) + ' ' + sdffile)
        mol = next(pybel.readfile('sdf', sdffile))
        chrg = str(mol.charge)
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  ' && xtb ' + file + ' --opt tight --gfn 2 --charge ' + chrg + ' --gbsa benzene --norestart > '
                  + 'output_xtb.out)')
        if not os.path.isfile(os.path.join(xyzdir, filename, 'xtbopt.sdf')):
            os.system('(cd ' + os.path.join(xyzdir, filename) +
                      ' && xtb ' + file + ' --opt tight --gfn 1 --charge ' + chrg + ' --gbsa benzene --norestart > '
                      + 'output_xtb.out)')
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  ' && obabel -isdf ' + 'xtbopt.sdf' + ' -O xtbopt.xyz)')

        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  ' && cp xtbopt.xyz ' + filename + '.xyz)')
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -type f -not -name 'ID*.xyz' -exec rm -rf {} \;)")
    else:
        if isFile:
            os.system('rm ' + os.path.join(xyzdir, file))
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -type f -not -name 'ID*.xyz' -exec rm -rf {} \;)")


# generate S0 comfiles for DFT
def gen_S0_comfiles(smiles, filename):
    # if os.path.isfile(os.path.join(xyzdir, filename, filename + '_S0.com')):
    #     return
    with open(os.path.join(xyzdir, filename, 'smiles'), 'w') as file:
        file.write(smiles + '\n')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && obabel -ixyz " + filename + ".xyz -O " + filename + "_prelim.com)")
    with open(os.path.join(xyzdir, filename, filename + '_prelim.com'), 'r') as file:
        comdata = file.readlines()
    with open(os.path.join(xyzdir, filename, filename + '_S0.com'), 'w') as file:
        file.write("%chk=gauss_S0.chk\n")
        file.write("%mem="+memGB+"\n")
        file.write("%nproc="+coresCPU+"\n")
        file.write("#b3lyp 6-31G* opt \n\n")
        file.write(' ' + filename + "_S0\n\n")
        mol = Chem.MolFromSmiles(smiles)
        chrg = Chem.GetFormalCharge(mol)
        rads = Descriptors.NumRadicalElectrons(mol)
        if rads % 2 == 0:
            mult = 1
        else:
            mult = 2
        file.write(str(chrg) + ' ' + str(mult) + '\n')
        # f.write("0 1\n")
        writeLine = False
        for index, line in enumerate(comdata):
            if index <= 5:
                continue
            file.write(line)
            # if writeLine:
            #     file.write(line)
            # if '0  1' in line:
            #     writeLine = True
        file.write('\n')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && rm " + filename + "_prelim.com)")


# generate S0 jobfile
# autosubmit jobfile

# run S0 DFT
def run_S0_DFT(filename):
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '_S0.log')):
        with open(os.path.join(xyzdir, filename, filename + '_S0.log'), 'r') as file:
            data = file.readlines()
        if 'Normal termination' in data[-1]:
            # pass
            return
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && g16 " + filename + "_S0.com)")


# get 3D coordinates from DFT output
def gen_coord_for_S1T1(filename):
    if not os.path.isfile(os.path.join(xyzdir, filename, filename + '_S0.log')):
        print('no log file for ' + filename)
        return
    # if os.path.isfile(os.path.join(xyzdir, filename, filename + '_S0.xyz')):
    #     return
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && g092xyz.pl " + filename + "_S0.log)")
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && mv g09-result.xyz " + filename + "_S0.xyz)")


# generate S1,T1 comfiles for TD-DFT
def gen_S1T1_comfiles(smiles, filename):
    # if os.path.isfile(os.path.join(xyzdir, filename, filename + '_S1.com')) and \
    #         os.path.isfile(os.path.join(xyzdir, filename, filename + '_T1.com')):
    #     return
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && obabel -ixyz " + filename + "_S0.xyz -O " + filename + "_prelim.com)")
    with open(os.path.join(xyzdir, filename, filename + '_prelim.com'), 'r') as file:
        comdata = file.readlines()
    with open(os.path.join(xyzdir, filename, filename + '_S1.com'), 'w') as file:
        file.write("%chk=gauss_S1.chk\n")
        file.write("%mem="+memGB+"\n")
        file.write("%nproc="+coresCPU+"\n")
        file.write("#b3lyp/6-31+g*  units=ang scf=(tight) nosym "
                   "td(nstates=10) int=(grid=ultrafine) \n\n")
        file.write(' ' + filename + "_S1\n\n")
        mol = Chem.MolFromSmiles(smiles)
        chrg = Chem.GetFormalCharge(mol)
        rads = Descriptors.NumRadicalElectrons(mol)
        if rads % 2 == 0:
            mult = 1
        else:
            mult = 2
        file.write(str(chrg) + ' ' + str(mult) + '\n')
        for index, line in enumerate(comdata):
            if index <= 5:
                continue
            file.write(line)
        # writeLine = False
        # for line in comdata:
        #     if writeLine:
        #         file.write(line)
        #     if '0  1' in line:
        #         writeLine = True
        file.write('\n')
    with open(os.path.join(xyzdir, filename, filename + '_T1.com'), 'w') as file:
        file.write("%chk=gauss_T1.chk\n")
        file.write("%mem="+memGB+"\n")
        file.write("%nproc="+coresCPU+"\n")
        file.write("#b3lyp/6-31+g*  units=ang scf=(tight) nosym "
                   "td(triplets,nstates=10) int=(grid=ultrafine) \n\n")
        file.write(' ' + filename + "_T1\n\n")
        mol = Chem.MolFromSmiles(smiles)
        chrg = Chem.GetFormalCharge(mol)
        rads = Descriptors.NumRadicalElectrons(mol)
        if rads % 2 == 0:
            mult = 1
        else:
            mult = 2
        file.write(str(chrg) + ' ' + str(mult) + '\n')
        for index, line in enumerate(comdata):
            if index <= 5:
                continue
            file.write(line)
        # writeLine = False
        # for line in comdata:
        #     if writeLine:
        #         file.write(line)
        #     if '0  1' in line:
        #         writeLine = True
        file.write('\n')
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && rm " + filename + "_prelim.com)")


# generate S1,T1 comfiles
# autosubmit comfiles

# run S1 TDDFT
def run_S1_TDDFT(filename):
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '_S1.log')):
        with open(os.path.join(xyzdir, filename, filename + '_S1.log'), 'r') as file:
            data = file.readlines()
        if 'Normal termination' in data[-1]:
            # pass
            return
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && g16 " + filename + "_S1.com)")


# run T1 TDDFT
def run_T1_TDDFT(filename):
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '_T1.log')):
        with open(os.path.join(xyzdir, filename, filename + '_T1.log'), 'r') as file:
            data = file.readlines()
        if 'Normal termination' in data[-1]:
            # pass
            return
    os.system('(cd ' + os.path.join(xyzdir, filename) +
              " && g16 " + filename + "_T1.com)")


# compile results
def extract_data(filename):
    os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -name 'gauss_*.chk' -exec rm -rf {} \;)")
    os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -name 'fort.7' -exec rm -rf {} \;)")
    os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && find . -name 'core*' -exec rm -rf {} \;)")
    if os.path.isfile(os.path.join(xyzdir, filename, filename + '_S1.log')) and \
            os.path.isfile(os.path.join(xyzdir, filename, filename + '_T1.log')):
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && grep 'Excited State   1' " + filename + "_S1.log | tail -1 > excitedstate_S1.out)")
        os.system('(cd ' + os.path.join(xyzdir, filename) +
                  " && grep 'Excited State   1' " + filename + "_T1.log | tail -1 > excitedstate_T1.out)")
    else:
        with open('excitedstate_S1.out', 'w') as file:
            file.write('')
        with open('excitedstate_T1.out', 'w') as file:
            file.write('')


def compile_data():
    allData = {}
    for smiles in SMI2ID:
        filename = SMI2ID[smiles]
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
                    allData[smiles] = {}
                    allData[smiles]['S1'] = S1Ex
                    allData[smiles]['T1'] = T1Ex
    with open('../TDDFT_' + smiles_file + '.csv', 'w') as file:
        file.write('SMILES,S1,T1\n')
        for smiles in allData:
            file.write(smiles + ',' + str(allData[smiles]['S1']) + ',' + str(allData[smiles]['T1']) + '\n')


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


def auto_TDDFT(smiles, filename):
    print('starting calcs for ' + filename)
    with open('progress.out', 'a') as file:
        file.write('starting calcs for ' + filename + '\n')
    print('generating sdf')
    gen_sdf(smiles, filename)
    print('generating xyz')
    gen_xyz(filename)
    print('generating S0 comfile')
    gen_S0_comfiles(smiles, filename)
    print('running ground state DFT')
    run_S0_DFT(filename)
    print('getting ground state coords')
    gen_coord_for_S1T1(filename)
    print('generating S1/T1 comfiles')
    gen_S1T1_comfiles(smiles, filename)
    print('running S1 TDDFT')
    run_S1_TDDFT(filename)
    print('running T1 TDDFT')
    run_T1_TDDFT(filename)
    print('extracting data')
    extract_data(filename)
    with open('progress.out', 'a') as file:
        file.write('finished calcs for ' + filename + '\n')


memGB = '100gb'
coresCPU = '40'
nodes = args.nodes
node_index = args.index
df = pd.read_csv(smiles_path)
smiInd = -1
for index, col in enumerate(df.columns):
    if col.lower() == 'smiles':
        smiInd = index
total = len(df.iloc[:, smiInd])
mols_per_node = int(np.floor(total / nodes) + 1)
for i in np.arange((node_index - 1) * mols_per_node, node_index * mols_per_node):
    IDstr = 'ID' + '%09d' % i
    with open(os.path.join(xyzdir, IDstr, 'smiles.txt')) as file:
        smiles = file.readline()
    auto_TDDFT(smiles, IDstr)


print('successfully completed!')
with open('progress.out', 'a') as file:
    file.write('successfully completed!\n')
f.close()
