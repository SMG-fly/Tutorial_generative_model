#!/bin/bash
#SBATCH -J SMG_VAE_train_2
#SBATCH -p 2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o ./slurm_log/VAE/%x_%j.out
#SBATCH -e ./slurm_log/VAE/%x_%j.out

##### Setting basics #####
source ~/.bashrc
conda activate min

mkdir -p ./slurm_log/VAE

##### Main run #####
cd ~/tutorial_model
python3 VAE_train.py --epochs 1000 --batch_size 128 --checkpoint_path ./Checkpoints/VAE-soft1000/checkpoint_epoch_499.pt --output_dir ./Checkpoints/VAE-soft500-softNoSeed500-tf0.8/ --data_path smiles.txt --tokenizer_path tokenizer/tokenizer_w_special_tokens.pkl --wandb_name VAE-training-soft500-softNoSeed500-tf0.8 --num_data 200000 --teacher_forcing_ratio 0.8
