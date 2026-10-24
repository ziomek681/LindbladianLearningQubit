#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10gb
#SBATCH --partition lem-gpu
#SBATCH --job-name=GPU_test
#SBATCH --mail-user=275340@student.pwr.edu.pl
#SBATCH --gres=gpu:hopper:1
#SBATCH --account=hpc-jarpaw-1791296019

mkdir -p "$HOME/python-packages"

apptainer exec /lustre/software-data/container-images/pytorch_latest.sif \
  python -m pip install --target "$HOME/python-packages" \
  numpy==1.26.4 scipy==1.13.1 matplotlib==3.8.4