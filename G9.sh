#!/bin/bash -l
# Slurm parameters
#SBATCH --job-name=Cycle
#SBATCH --output=cityscapes_G9_oasisD_44_0.2%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:30:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch









# Activate everything you need
source ~/anaconda3/etc/profile.d/conda.sh
conda activate testenv

# Run your python code

python train.py --dataroot /data/public/cityscapes --mixed_images --add_vgg_loss  --name cityscapes_G9_oasisD_44_0.3 \
--num_epochs 300 --supervised_percentage 0.33 --gpu_ids 0 --batch_size 1  --channels_G 16  --lr_g 0.0004 --lr_d 0.0004     