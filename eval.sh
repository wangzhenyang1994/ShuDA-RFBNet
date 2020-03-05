#!/bin/bash
#SBATCH -J 2d_detection
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:2
module load anaconda3/5.3.0
source activate naive
python -u val_RFB.py --batch_size 16 --num_workers 8 --ngpu 2 --resume_net ./weights/RFB_vgg_VOC_epoches_300.pth
