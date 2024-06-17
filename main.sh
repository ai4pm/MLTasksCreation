#!/bin/bash
#SBATCH --job-name=main
#SBATCH --mail-user=tsharma2@uthsc.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=ACF-UTHSC0001
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --output=Protein_100Features_Test/o_e_files/main.o%j
#SBATCH --error=Protein_100Features_Test/o_e_files/main.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01-00:00:00

###########################################

source activate FADA_py36

echo "The environment has been activated."

python Protein_100Features_Test/main.py

echo "The execution has been done."



