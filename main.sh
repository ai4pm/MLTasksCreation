#!/bin/bash
#SBATCH --job-name=BLCA_OS_1YR_BLACK_Protein
#SBATCH --account=ACF-UTHSC0001
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --output=BLCA_OS_1YR_BLACK_Protein.o%j
#SBATCH --error=BLCA_OS_1YR_BLACK_Protein.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01-00:00:00

###########################################

source activate multiethnic

echo "The environment has been activated."

python BLCA_OS_1YR_BLACK_Protein.py BLCA OS 1 BLACK 200

echo "The execution has been done."



