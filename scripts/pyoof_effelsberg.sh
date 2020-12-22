#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --time=12:00:00
#SBATCH --job-name pyoof_run
#SBATCH --output=/scratch/v/vanderli/cassane/output/pyoof_%j.txt
#SBATCH --mail-type=ALL

module load gcc/8.3.0
module load openmpi/3.1.3
module load texlive

cd $SCRATCH

# INPUT 
file="pyoof_effelsberg_mpi.py"
# file="create_lookup.py"

directory="/home/v/vanderli/cassane/pyoof/scripts/"
path2file=$directory$file

# python environment
APY3="/home/v/vanderli/cassane/anaconda3/bin/python"

# # EXECUTION COMMAND
mpirun -np 7 $APY3 $path2file
# mpirun -np 4 $APY3 /home/v/vanderli/cassane/pyoof/scripts/create_lookup.py