#!/bin/bash
#SBATCH -J SMG_MPNN_test_QED
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
  --split_path ./Result_MPNN/split_indices-QED-1.pkl \
  --ckpt_path ./Result_MPNN/Checkpoints/MPNN_QED_epoch_49.pt \
  --target QED \
  --threshold 2.5
