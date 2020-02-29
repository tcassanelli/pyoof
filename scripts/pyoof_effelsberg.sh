#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name coherent_dd
#SBATCH --output=/scratch/v/vanderli/cassane/output/mpi_ex_%j.txt
#SBATCH --mail-type=ALL

cd $SCRATCH

# INPUT 
file="pyoof_effelsberg.py"
directory="/home/v/vanderli/cassane/pyoof/scripts/"
path2file=$directory$file

# python environment
APY3="/home/v/vanderli/cassane/anaconda3/bin/python"

# # EXECUTION COMMAND
$APY3 $path2file
