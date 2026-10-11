#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10gb
#SBATCH --partition lem-gpu
#SBATCH --job-name=GPU_test
#SBATCH --mail-user=275340@student.pwr.edu.pl
#SBATCH --gres=gpu:hopper:1
#SBATCH --account=hpc-jarpaw-1791296019

apptainer exec --nv /lustre/software-data/container-images/pytorch_latest.sif ./train.py  --epochs 100 --out Run1

#apptainer exec --nv /lustre/software-data/container-images/pytorch_latest.sif ./train.py  --epochs 2 --steps-per-epoch 4 --n-test 32