#!/bin/bash
#SBATCH -J SMG_MPNN_preprocess_1
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
python3 MPNN_preprocess.py --smiles_path ./smiles.txt --graph_path ./Result_MPNN/preprocessed_smiles_graph.pt 
