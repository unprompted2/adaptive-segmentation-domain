#!/bin/bash -l
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=28:00:00
module purge
module load PyTorch
module unload PyTorch

# define context variables
PYTHON_EXE=python

# define environment variables
PROJECT_DIR=$HOME/research/domain-adaptive-segmentation
export PYTHONPATH=$HOME/research/neuralnets:$HOME/