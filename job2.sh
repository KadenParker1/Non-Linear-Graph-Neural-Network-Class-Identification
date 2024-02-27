#!/bin/bash

#SBATCH --time=01:00:00   # walltime
##SBATCH --acount=neo
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -J "nonlineargnnst"   # job name



# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate boydresearch

srun -u python3 polarwauto.py 
