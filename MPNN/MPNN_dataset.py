import os

import torch
from torch_geometric.data import Dataset

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

from MPNN_preprocess import smiles_to_graph


class SMILESDataset(Dataset):
    def __init__(self, graph_data_path: str, smiles_path: str, target: str = "logP", num_data=None):
        self.data_list = torch.load(graph_data_path)
        with open(smiles_path, 'r') as f:
            self.smiles_list = list(filter(None, f.read().splitlines()))
        assert len(self.data_list) == len(self.smiles_list), "Mismatch between graphs and SMILES."
        
        if num_data is not None:
            self.data_list = self.data_list[:num_data]
        
        for i, data in enumerate(self.data_list):
            mol = Chem.MolFromSmiles(self.smiles_list[i])
            target_value = self.compute_target_property(mol, target)
            data.y = torch.tensor([target_value], dtype=torch.float)       

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def compute_target_property(self, mol, target: str) -> float:
        if target == "logP":
            return MolLogP(mol)
        elif target == "QED":
            return qed(mol)
        elif target == "MW":
            return Descriptors.MolWt(mol)
        elif target == "TPSA":
            return Descriptors.TPSA(mol)
        elif target == "HBA":
            return Descriptors.NumHAcceptors(mol)
        elif target == "HBD":
            return Descriptors.NumHDonors(mol)
        elif target == "RB":
            return Descriptors.NumRotatableBonds(mol)
        elif target == "RINGS":
            return Descriptors.RingCount(mol)
        else:
            raise ValueError(f"Unsupported target: {target}")