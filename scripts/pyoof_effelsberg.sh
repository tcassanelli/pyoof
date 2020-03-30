#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name pyoof_run
#SBATCH --output=/scratch/v/vanderli/cassane/output/mpi_ex_%j.txt
#SBATCH --mail-type=ALL

module load texlive

cd $SCRATCH

# INPUT 
file1="pyoof_effelsberg.py"
file2="create_lookup.py"

directory="/home/v/vanderli/cassane/pyoof/scripts/"
path2file=$directory$file

# python environment
APY3="/home/v/vanderli/cassane/anaconda3/bin/python"

# # EXECUTION COMMAND
$APY3 $path2file1
$APY3 $path2file2
