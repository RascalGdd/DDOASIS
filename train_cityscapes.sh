#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=job_name
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse19

python train.py --name semisupervised_S4U4_no_D_U --dataset_mode cityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes --batch_size_supervised 4 --batch_size_train 4  --no_labelmix \
--Du_patch_size 64 --netDu wavelet --netG 0 \
--channels_G 64 --num_epochs 1500 \
--model_supervision 1 --supervised_percentage 50 \
--results_dir /no_backups/h162/ 



