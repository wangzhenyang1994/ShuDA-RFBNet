#!/bin/bash
#SBATCH -J 2d_detection
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:2
module load anaconda3/5.3.0 cuda/9.0
source activate naive
cd ./utils/

python build.py build_ext --inplace

cd ..
