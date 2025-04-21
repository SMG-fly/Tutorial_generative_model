#!/bin/bash
#SBATCH -J SMG_VAE_inference
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
python3 VAE_inference.py --checkpoint_path ./Checkpoints/VAE-soft500-softNoSeed500-tf0.8/checkpoint_epoch_999.pt --output_dir ./Result_VAE/ --tokenizer_path tokenizer/tokenizer_w_special_tokens.pkl 