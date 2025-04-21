#!/bin/bash
#SBATCH -J SMG_MPNN_test_1
#SBATCH -p 2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o ./slurm_log/MPNN/%x_%j.out
#SBATCH -e ./slurm_log/MPNN/%x_%j.out

##### Setting basics #####
source ~/.bashrc
conda activate bindenv

##### Main run #####
cd ~/tutorial_model
python3 MPNN_test.py \
  --graph_path ./Result_MPNN/preprocessed_smiles_graph.pt \
  --smiles_path ./smiles.txt \
  --split_path ./Result_MPNN/split_indices.pkl \
  --ckpt_path ./Result_MPNN/Checkpoints/test100_logP_checkpoint_epoch_9.pt \
  --target logP \
  --threshold 2.5