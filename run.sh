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
# python -u train_RFB.py --batch_size 32 --num_workers 8 --ngpu 2 --learning-rate 1e-3
python -u train_RFB.py --batch_size 32 --num_workers 8 --ngpu 2 --learning-rate 1e-3 --resume_net ./weights/RFB_vgg_VOC_epoches_284.pth --resume_epoch 284
