##### SMILES to Graph Data Preprocessing (targets are not included)#####
import argparse

import torch
from torch_geometric.data import Data

from rdkit import Chem

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MPNN on SMILES dataset.")
    parser.add_argument("--smiles_path", type=str, required=True, help="Path to the corresponding SMILES strings (.txt)")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the saved graph data file (.pt)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

# SMILES to Graph Data Preprocessing
def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms(): # mol.GetAtoms() : 분자를 구성하는 원자 리스트
        atom_features.append([atom.GetAtomicNum()]) # 원자 번호 (ex. C=6, O=8)
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index and features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds(): # 분자의 결합들
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j)) # ex. [(0, 1), (1, 0), ...]
        edge_index.append((j, i)) # undirected graph
        bond_type = bond.GetBondTypeAsDouble() # 결합의 종류 (single, double, triple)
        edge_attr.append([bond_type]) 
        edge_attr.append([bond_type]) # 양방향으로 edge를 넣었으니 edge_attr도 두 번씩 추가

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() #.t(): transpose (num_edges, 2) -> (2, num_edges) # contiguous(): 메모리 연속성 보장
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr) # torch_geometric.data.Data 객체 생성 # One graph per SMILES string

def process_smiles(smiles_list: list) -> list:
    graph_data_list = []
    for smiles in smiles_list:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph_data_list.append(graph)
    return graph_data_list

def main():
    args = parse_arguments()

    ##### For Debugging #####
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))  # 포트 번호는 유지 또는 원하는 번호로 변경
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
    #########################

    with open(args.smiles_path, "r") as f:
        smiles_list = list(filter(None, f.read().splitlines()))

    data_list = process_smiles(smiles_list)
    torch.save(data_list, args.graph_path)  # Save the processed graph data
    print(f"{len(data_list)} Graph data saved to {args.graph_path}")


if __name__ == "__main__":
    main()
