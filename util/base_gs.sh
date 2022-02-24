#!/bin/bash -l
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00

# define context variables
PYTHON_EXE=$PYTHON_DEFAULT

# define environment variables
PROJECT_DIR=$HOME/research/domain-adaptive-segmentation
export PYTHONPATH=$PY