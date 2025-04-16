import argparse
import copy
import os
import pickle
import random
import time
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import wandb

from rdkit import Chem

from VAE_model import SmVAE

# Parser
# To do: Add wandb + project name
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a protein sequence model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint for resuming training")
    parser.add_argument("--cuda_device", type=int, default=None, help="CUDA device number to use")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    #parser.add_argument("--checkpoint_step", type=int, default=500, help="Step interval for saving checkpoints")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--num_data", type=int, default=20000)
    parser.add_argument("--kl_ratio", type=float, default=10.0)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--tqdm", action="store_true", help="Show tqdm progress bar")
    return parser.parse_args()

# Set seed
def set_seed(seed: int = 42):
    torch.manual_seed(seed)  # PyTorch의 난수 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU의 난수 고정 (GPU 사용 시)
    random.seed(seed)  # Python 기본 random 라이브러리 고정
    np.random.seed(seed)  # NumPy 난수 고정
    torch.backends.cudnn.deterministic = True  # CUDNN 연산 결정적 사용
    torch.backends.cudnn.benchmark = False  # 성능 최적화 방지 (재현성을 위해)

##### Preprocess #####
def load_data(file_path: str, max_len: int, num: int):
    smiles_list = []
    with open(file_path, 'r') as f:
        for line in f:
            smiles = line.strip().split('\t')[-1]
            
            if smiles == "" or len(smiles) < 2:
                continue
            
            if len(smiles) <= max_len - 2: # -2: <sos>, <eos> tokens
                smiles_list.append(smiles)

            if len(smiles_list) >= num:
                break

        return smiles_list

def load_tokenizer(tokenizer_path : str) -> dict: 
    """Load the tokenizer from a pkl file."""
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file) 
    
    if not isinstance(tokenizer, dict):
        raise TypeError("Tokenizer file must contain a dictionary.")
    
    # check if there are special tokens
    if 'X' not in tokenizer:
        raise ValueError("Tokenizer must include a padding token 'X'.")

    index_to_char = {idx: char for char, idx in tokenizer.items()}

    return tokenizer, index_to_char


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer):
        self.smiles_list = smiles_list # self.data
        self.tokenizer = tokenizer

        # Encode SMILES strings to index sequences
        encoded_smiles_list = self.encode_smiles()
        self.seq_list = encoded_smiles_list

        # Compute lengths + Convert list to tensor
        length_list = [len(smiles) for smiles in encoded_smiles_list]
        self.length_list = torch.from_numpy(np.array(length_list))
        
    def encode_smiles(self):
        encoded_smiles_list = []
        for smiles in self.smiles_list:
            tokens = [self.tokenizer[char] for char in smiles] 
            encoded_tensor = torch.from_numpy(np.array(tokens)) # Convert to tensor
            encoded_smiles_list.append(encoded_tensor)
        return encoded_smiles_list

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__ (self, index):
        return {
            "seq": self.seq_list[index],              # seq
            "length": self.length_list[index].item() # valid sequence length
        }
    

def add_special_tokens(smiles_list, sos_token='^', eos_token='X'): # To do: Modify to <eos> token (now: <pad>)
    """Add special tokens to each SMILES string."""
    return [f"{sos_token}{smiles}{eos_token}" for smiles in smiles_list]

# Collate function # To do: revise
def collate_fn(batch):
    sequence_batch = []
    length_batch = []

    for sample in batch:
        sequence_batch.append(sample["seq"])     # SMILES sequence
        length_batch.append(sample["length"])    # valid sequence length 

    padded_seq = pad_sequence(
        sequence_batch,
        batch_first=True,
        padding_value=0  # 0: 'X' token
    )

    return {
        "seq": padded_seq,                                 # padded SMILES sequence
        "length": torch.tensor(length_batch),              # valid sequence length (doesn't need to be dynamic padded)
    }

# Save checkpoint
def save_checkpoint(model, optimizer, epoch, step, loss, output_dir, filename="step"):
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
def train(model, train_dataloader, valid_dataloader, optimizer, device, args, start_epoch=0, global_step=0):
    # model.train() # Activate when not valid per epoch
    for epoch in range(start_epoch, args.epochs):
        model.train() # Activate when valid per epoch
        start = time.time()
        loop = tqdm(train_dataloader, leave=True, disable=not args.tqdm)

        recon_loss_list, kl_loss_list = [], []

        for batch_idx, batch in enumerate(loop):
            input_batch = batch["seq"].to(device)
            length_batch = batch["length"].to(device)

            logits, mean, logvar = model(input_token_indices=input_batch, teacher_forcing_ratio=args.teacher_forcing_ratio) # [batch_size, max_len, vocab_size]
            recon_loss, kl_loss = model.compute_loss(logits, input_batch, mean, logvar, length_batch) # [batch_size, max_len]
            loss = recon_loss + args.kl_ratio * kl_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping # prevent exploding gradient
            optimizer.step()

            recon_loss_list.append(recon_loss.item())
            kl_loss_list.append(kl_loss.item())

            # wandb에 로그 기록
            if wandb.run:
                wandb.log({
                    "train_recon_loss": recon_loss.item(),
                    "train_kl_loss": kl_loss.item(),
                    "train_total_loss": loss.item(),
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batch_idx": batch_idx
                })

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

            global_step += 1
            # Checkpoint saving
            #if global_step % args.checkpoint_step == 0:
            #    save_checkpoint(model, optimizer, epoch, global_step, loss, args.output_dir, "step")

        recon_loss_avg = np.mean(recon_loss_list)
        kl_loss_avg = np.mean(kl_loss_list)
        print(f"Epoch {epoch+1}: recon={recon_loss_avg:.5f}, kl={kl_loss_avg:.5f}, time={time.time() - start:.2f}s")
        
        validate(model, valid_dataloader, device, epoch, args)

        save_checkpoint(model, optimizer, epoch, global_step, loss, args.output_dir, "epoch")

# Validation
@torch.no_grad()
def validate(model, dataloader, device, epoch, args):
    model.eval()
    recon_loss_list, kl_loss_list, total_loss_list = [], [], []
    for batch_idx, batch in enumerate(dataloader):
        input_batch = batch["seq"].to(device)
        length_batch = batch["length"].to(device)

        # 모델 Forward & Loss 계산
        logits, mean, logvar = model(input_token_indices=input_batch, teacher_forcing_ratio=args.teacher_forcing_ratio) # [batch_size, max_len, vocab_size]
        recon_loss, kl_loss = model.compute_loss(logits, input_batch, mean, logvar, length_batch) # [batch_size, max_len]
        total_loss = recon_loss + args.kl_ratio * kl_loss

        recon_loss_list.append(recon_loss.item())
        kl_loss_list.append(kl_loss.item())
        total_loss_list.append(total_loss.item())
        
        # wandb에 로그 기록
        if wandb.run:
            wandb.log({
                "valid_loss": total_loss.item(),
                "epoch": epoch + 1,
                "batch_idx": batch_idx
            })

    avg_recon = np.mean(recon_loss_list)
    avg_kl = np.mean(kl_loss_list)
    avg_loss = np.mean(total_loss_list)

    print(f"Validation: recon={avg_recon:.5f}, kl={avg_kl:.5f}, total={avg_loss:.5f}")
    return avg_recon, avg_kl, avg_loss

def main():
    # Setting
    args = parse_arguments()
    set_seed(42) 

    if args.cuda_device:
        device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data # To do: move to argparse
    smiles_list = load_data(args.data_path, args.max_len, args.num_data)
    smiles_list = add_special_tokens(smiles_list) # Add special tokens
    tokenizer, index_to_char = load_tokenizer(args.tokenizer_path)
    print(f'n_char: {len(tokenizer)}')
    print(f'index of <pad>: {tokenizer["X"]}')

    # Split data # To do: Modify to train_test_split
    #train_data = SMILESDataset(smiles_list[:16000], tokenizer)
    #valid_data = SMILESDataset(smiles_list[16000:18000], tokenizer)
    #test_data = SMILESDataset(smiles_list[18000:20000], tokenizer)
    valid_data = SMILESDataset(smiles_list[1:1000], tokenizer)
    test_data = SMILESDataset(smiles_list[1000:2000], tokenizer)
    train_data = SMILESDataset(smiles_list[2000:], tokenizer)
   
    # DataLoader
    dataloaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),
        "valid": DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    }

    # Prepare model 
    model = SmVAE(vocab_size=len(tokenizer), embedding_dim=128, hidden_dim=128, latent_dim=128, num_rnn_layers=3)
    model.to(device)

    # Optimizer 설정
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # checkpoint 불러오기
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"]) # checkpoint에 state_dict외 다른 것도 포함된 경우
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("step", 0)
            print(f"Resumed from checkpoint at epoch {start_epoch}, step {global_step}")
        else:
            model.load_state_dict(checkpoint)  # 직접 불러오기 (state_dict만 저장된 경우)

    # wandb 초기화
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
    train(model, dataloaders["train"], dataloaders["valid"], optimizer, device, args, start_epoch=0, global_step=0)

if __name__ =="__main__":
    main()


