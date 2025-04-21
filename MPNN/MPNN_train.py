import argparse
import copy
import os
import pickle
import random
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import random_split

from tqdm import tqdm
import wandb

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

from MPNN_dataset import SMILESDataset
from MPNN_model import SmMPNN

# Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MPNN on SMILES dataset.")
    parser.add_argument("--smiles_path", type=str, required=True, help="Path to the corresponding SMILES strings (.txt)")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the saved graph data file (.pt)")
    parser.add_argument("--target", type=str, default="logP", choices=["logP", "QED", "MW", "TPSA", "HBA", "HBD", "RB", "RINGS"], help="Target molecular property for prediction.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (L2 regularization).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    parser.add_argument("--ckpt_out_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--saved_ckpt_path", type=str, default=None, help="Path to model checkpoint for resuming training")
    parser.add_argument("--valid_ratio", type=float, default=0.01, help="Ratio of validation set.")
    parser.add_argument("--test_ratio", type=float, default=0.01, help="Ratio of test set.")
    parser.add_argument("--num_data", type=int, default=None, help="Number of data samples to use from the dataset.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size for MPNN layers.")
    parser.add_argument("--split_path", type=str, required=True, help="Path to save split index file (train/valid/test)")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--tqdm", action="store_true", help="Show tqdm progress bar")
    # parser.add_argument("--max_len", type=int, default=64) # To do
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

# Set seed
def set_seed(seed=42):
    torch.manual_seed(seed)  # PyTorch의 난수 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU의 난수 고정 (GPU 사용 시)
    random.seed(seed)  # Python 기본 random 라이브러리 고정
    np.random.seed(seed)  # NumPy 난수 고정
    torch.backends.cudnn.deterministic = True  # CUDNN 연산 결정적 사용
    torch.backends.cudnn.benchmark = False  # 성능 최적화 방지 (재현성을 위해)

# Save checkpoint
def save_checkpoint(model, optimizer, epoch, step, loss, output_dir, filename="step"):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if filename == "step":
        torch.save(checkpoint, f"{output_dir}/checkpoint_step_{step}.pt")
    elif filename == "epoch":
        torch.save(checkpoint, f"{output_dir}/checkpoint_epoch_{epoch}.pt")
    elif filename == "final":
        torch.save(checkpoint, f"{output_dir}/final_model.pt")

# 학습 함수
def train(model, train_dataloader, valid_dataloader, optimizer, loss_fn, device, args, start_epoch=0, global_step=0):
    # model.train()
    train_loss_history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        loop = tqdm(train_dataloader, leave=True, disable=not args.tqdm)
        train_loss_list = []

        for batch_idx, batch in enumerate(loop):
            input_batch = batch.to(device)   

            # 모델 Forward & Loss 계산
            output_batch = model(input_batch)

            loss = loss_fn(output_batch, input_batch.y)
            train_loss_list.append(copy.deepcopy(loss.detach().cpu().numpy()))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping # prevent exploding gradient
            optimizer.step()

            # wandb에 로그 기록
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_idx": batch_idx
            })

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

            # Checkpoint saving
            #if global_step % args.checkpoint_step == 0:
            #    save_checkpoint(model, optimizer, epoch, global_step, loss, args.output_dir, "step")

            global_step += 1
                
        epoch_train_avg_loss = np.sum(np.array(train_loss_list)) / len(train_dataloader.dataset) # To do: 계산 맞는지 확인... reduction이 sum에서 mean으로 바꼈는데 그대로인 게 더 이상하지 않아? # divide by the number of dataset samples, not batch samples 
        train_loss_history.append(epoch_train_avg_loss)
        validate(model, valid_dataloader, loss_fn, device, epoch)
        save_checkpoint(model, optimizer, epoch, global_step, loss, args.ckpt_out_dir, "epoch")
        print(f"Epoch {epoch + 1} Loss: {epoch_train_avg_loss:.4f} Time: {time.time() - start:.2f}s")

    # Save final model
    save_checkpoint(model, optimizer, args.epochs, global_step, loss, args.ckpt_out_dir, "final")


# Validation
@torch.no_grad()
def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    valid_loss_history = []
    valid_loss_list = []
    best_valid_loss = float("inf")
    for batch_idx, batch in enumerate(dataloader):
        input_batch = batch.to(device)

        # 모델 Forward & Loss 계산
        output_batch = model(input_batch)
        loss = loss_fn(output_batch, input_batch.y)
        valid_loss_list.append(loss.item())

        # wandb에 로그 기록
        if wandb.run:
            wandb.log({
                "valid_loss": loss.item(),
                "epoch": epoch + 1,
                "batch_idx": batch_idx
            })

    epoch_valid_avg_loss = np.sum(np.array(valid_loss_list)) / len(dataloader.dataset) # divide by the number of dataset samples, not batch samples
    valid_loss_history.append(epoch_valid_avg_loss)
    if epoch_valid_avg_loss < best_valid_loss:
        best_epoch = epoch +1 
        best_valid_loss = epoch_valid_avg_loss
        #best_model_weight = copy.deepcopy(model.state_dict()) # To do: save best model(save checkpoint)
        print(f"Best model at epoch {best_epoch} with loss {best_valid_loss:.4f}")

def main():
    set_seed(42) 
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

    ##### Load dataset ##### # To do: modularize
    full_dataset = SMILESDataset(args.graph_path, args.smiles_path, args.target, args.num_data)
    total_size = len(full_dataset)
    valid_size = int(total_size * args.valid_ratio)
    test_size = int(total_size * args.test_ratio)
    train_size = total_size - valid_size - test_size

    train_set, valid_set, test_set = random_split(full_dataset, [train_size, valid_size, test_size])
    
    # Save split indices (for test)
    splits = {
        "train": train_set.indices,
        "valid": valid_set.indices,
        "test": test_set.indices
    }
    with open(args.split_path, "wb") as f:
        pickle.dump(splits, f)

    # DataLoader
    data_loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        "valid": DataLoader(valid_set, batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    }
    ########################

    # Model Initialization
    node_dim = full_dataset[0].x.shape[1]  # 첫 번째 그래프에서 node_dim 추출
    edge_dim = full_dataset[0].edge_attr.shape[1]

    model = SmMPNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim).to(device)
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.saved_ckpt_path:
        checkpoint = torch.load(args.saved_ckpt_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"]) # checkpoint에 state_dict외 다른 것도 포함된 경우
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("step", 0)
            print(f"Resumed from checkpoint at epoch {start_epoch}, step {global_step}")
        else:
            model.load_state_dict(checkpoint)  # 직접 불러오기 (state_dict만 저장된 경우)
 
    # wandb Initialization
    if args.wandb_name:
        wandb.init(project="Tutorial_ACE", name=args.wandb_name, config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        })
        print(f"[WandB] Run name: {args.wandb_name}")
    else:
        print("[WandB] Run name not provided. WandB logging will be skipped.")

    # 모델 학습 실행
    train(model, data_loaders["train"], data_loaders["valid"], optimizer, loss_fn, device, args, start_epoch=start_epoch, global_step=global_step)

    # Save final model
    

if __name__ =="__main__":
    main()