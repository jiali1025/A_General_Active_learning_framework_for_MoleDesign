TDDFT from SMILES Instructions

These files are used to run TDDFT with Gaussin as a job array on a SLURM cluster. You can configure your own job scripts as well. The main file with calculation parameters is TDDFT_from_SMILES.py - the others are used to set up directories, submit jobs, and compile results.


Setup and Installation

1. Make sure Gaussian is configured. Try running 'g16' in terminal to check

2. Install the following python packages
typed-argument-parser 
rdkit
openbabel
pandas
numpy
tqdm

3. Install xtb (using a precompiled binary is easiest. One is provided in this folder). If using a precompiled binary, you just need to make sure it is somewhere in your path.

4. Copy the g092xyz.pl script to somewhere in your path.

5. Check the mem and cpu requirements in the job script. The Gaussian script is set to run on 100GB memory and 40 cpus. If that configuration is unavailable on your cluster, you will need to edit those values in both the job script (gaussian mem + 25 gb) AND the TDDFT_from_SMILES.py file (lines 390 and 391)

6. Check the other lines in the job script including partition choice (with -p), module loading, and conda environment activation. Change them to the desired values.

7. Change the ephemdir directory in all 3 TDDFT_from_SMILES* files (either in the python file itself in the SimpleArgParse class, or as an input argument in-line). This will be where all the calculations are done so make sure there is enough storage in this location.


How to Run

1. Run the setup script. This will create all the directories and submit the job array to SLURM. For example:

python TDDFT_from_SMILES_setup.py --smiles_path test.csv --create_dir

For future runs, you can leave out the --create_dir keyword to save some time.

2. Wait for calculations to complete. Check the job queue for progress.

3. Compile results with:

python TDDFT_from_SMILES_compile.py --smiles_path test.csv

4. This will create a file called TDDFT_test.csv - check test results against TDDFT_test_result.csv to ensure everything ran correctly.

