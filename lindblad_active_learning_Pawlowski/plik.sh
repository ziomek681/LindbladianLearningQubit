#!/bin/bash
#SBATCH --time=02:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10gb
#SBATCH --partition lem-gpu
#SBATCH --job-name=GPU_test
#SBATCH --mail-user=275340@student.pwr.edu.pl
#SBATCH --gres=gpu:hopper:1
#SBATCH --account=hpc-jarpaw-1791296019

#apptainer exec --nv \
#  --bind "$HOME/python-packages:/opt/python-packages" \
#  /lustre/software-data/container-images/pytorch_latest.sif \
#  bash -c 'export PYTHONPATH=/opt/python-packages:$PYTHONPATH; python ./train.py --epochs 2 --steps-per-epoch 4 --n-test 32 --val-metric all'

apptainer exec --nv \
  --bind "$HOME/python-packages:/opt/python-packages" \
  /lustre/software-data/container-images/pytorch_latest.sif \
  bash -c 'export PYTHONPATH=/opt/python-packages:$PYTHONPATH; python ./train.py --epochs 100 --p-bayes 1.0 --val-metric bayes --out Run4'