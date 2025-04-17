import argparse
import pickle
import torch
import numpy as np

from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from MPNN_model import SmMPNN
from MPNN_dataset import SMILESDataset

# Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MPNN on SMILES dataset.")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the saved graph data file (.pt)")
    parser.add_argument("--smiles_path", type=str, required=True, help="Path to the corresponding SMILES strings (.txt)")
    parser.add_argument("--split_path", type=str, required=True, help="Path to save split index file (train/valid/test)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint for resuming training")
    parser.add_argument("--target", type=str, default="logP", choices=["logP", "QED", "MW", "TPSA", "HBA", "HBD", "RB", "RINGS"], help="Target molecular property for prediction.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for binary classification (optional)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size for MPNN layers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

@torch.no_grad()
def test(model, dataloader, loss_fn, device, threshold=None):
    model.eval()
    y_true = []
    y_pred = []

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        input_batch = batch.to(device)
        output_batch = model(input_batch)

        y_true.append(input_batch.y.detach().cpu())
        y_pred.append(output_batch.detach().cpu())

        loss = loss_fn(output_batch, input_batch.y)
        total_loss += loss.item()
        total_samples += input_batch.num_graphs

    mean_loss = total_loss / total_samples
    print(f"[Test Loss] {mean_loss:.4f}")
    print(f"[Test Samples] {total_samples}")

    # Evaluation (Regression)
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    print(f"[Test Results]")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")

    # Evaluation (Binary Classification)
    if threshold is not None:
        y_true_cls = (y_true >= threshold).astype(int)
        y_pred_cls = (y_pred >= threshold).astype(int)

        conf_matrix = confusion_matrix(y_true_cls, y_pred_cls)
        report = classification_report(y_true_cls, y_pred_cls, digits=4)

        print(f"\n[Threshold = {threshold}] Confusion Matrix:\n{conf_matrix}")
        print(f"\nClassification Report:\n{report}")
    
    print("\n[Predictions vs Ground Truth]")
    for i in range(len(y_true)):
        print(f"{i:3d}: pred = {y_pred[i]:.4f} | true = {y_true[i]:.4f} | error = {abs(y_pred[i] - y_true[i]):.4f}")


def main():
    args = parse_arguments()
    device = torch.device(args.device)

    ##### For Debugging #####
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))  # 포트 번호는 유지 또는 원하는 번호로 변경
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
    #########################

    # Load dataset
    dataset = SMILESDataset(args.graph_path, args.smiles_path, target=args.target)
    with open(args.split_path, 'rb') as f:
        split_dict = pickle.load(f)
    test_indices = split_dict["test"]
    test_set = torch.utils.data.Subset(dataset, test_indices) 
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Load model
    node_dim = dataset.data_list[0].x.shape[1]
    edge_dim = dataset.data_list[0].edge_attr.shape[1]
    model = SmMPNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Define loss
    loss_fn = nn.MSELoss(reduction="sum")

    # Run test
    test(model, test_loader, loss_fn, device, threshold=args.threshold)
    

if __name__ == "__main__":
    main()