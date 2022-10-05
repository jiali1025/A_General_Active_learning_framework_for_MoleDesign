TODO: edit this for xTB instead of TDDFT. most steps are equivalent.

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

7. Change the ephemdir and storagedir directories in all 3 TDDFT_from_SMILES* files (either in the python file itself in the SimpleArgParse class, or as an input argument in-line). ephemdir is where all the calculations are done so make sure that filesystem is optimized for parallel operation. After calculations are complete the files are moved to storagedir so make there there is enough storage in that directory. (i.e. for engaging /nobackup is used for ephemeral and /pool001 is used for storage.)


How to Run

1. Run the setup script. This will submit the job array to SLURM. For example:

python TDDFT_from_SMILES_setup.py --smiles_path test.csv

2. Wait for calculations to complete. Check the job queue for progress.

3. Compile results with:

python TDDFT_from_SMILES_compile.py --smiles_path test.csv

4. This will create a new file TDDFT_test.csv with all the results!

5. If a run stops unexpectedly, run the TDDFT_from_SMILES_compile.py script to compile all completed calculations. Then you can rerun TDDFT_from_SMILES_setup.py and it will automatically skip over molecules with TDDFT results already completed.


Notes

1. Previous versions created all directories before computation started, but this hit disk quotas of maximum file/folder numbers. Now, directories are created as needed.