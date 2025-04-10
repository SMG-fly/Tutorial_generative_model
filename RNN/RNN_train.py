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
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import wandb

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

from RNN_model import RNNRegressor

# Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a protein sequence model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training dataset(parquet)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint for resuming training")
    #parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device number to use")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory")
    #parser.add_argument("--checkpoint_step", type=int, default=500, help="Step interval for saving checkpoints")
    return parser.parse_args()

# Set seed
def set_seed(seed=42):
    torch.manual_seed(seed)  # PyTorch의 난수 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU의 난수 고정 (GPU 사용 시)
    random.seed(seed)  # Python 기본 random 라이브러리 고정
    np.random.seed(seed)  # NumPy 난수 고정
    torch.backends.cudnn.deterministic = True  # CUDNN 연산 결정적 사용
    torch.backends.cudnn.benchmark = False  # 성능 최적화 방지 (재현성을 위해)

##### Preprocess #####
def load_data(file_path, max_len, num):
    f = open(file_path, 'r')
    smiles_list = []
    cnt = 0
    while cnt < num:
        line = f.readline()
        smi = line.strip().split('\t')[-1]
        if len(smi) <= max_len-1:
            smiles_list.append(smi)
            cnt += 1
    return smiles_list

def load_tokenizer(tokenizer_path : str) -> dict: # tokenizer_path: c_to_i.pkl
    """Load the tokenizer from a pkl file."""
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file) 
    
    if not isinstance(tokenizer, dict):
        raise TypeError("Tokenizer file must contain a dictionary.")
    
    if 'X' not in tokenizer:
        raise ValueError("Tokenizer must include a padding token 'X'.")

    return tokenizer

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, logp_list, tokenizer):
        self.smiles_list = smiles_list # self.data
        self.tokenizer = tokenizer

        # Encode SMILES strings to index sequences
        encoded_smiles_list = self.encode_smiles()
        self.seq_list = encoded_smiles_list

        # Compute lengths + Convert list to tensor
        length_list = [len(smiles) for smiles in encoded_smiles_list]
        self.length_list = torch.from_numpy(np.array(length_list))

        # Convert logP list to tensor
        self.prop_list = torch.from_numpy(np.array(logp_list))
        
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
            "length": self.length_list[index].item(), # valid sequence length
            "logp": self.prop_list[index]             # logP
        }

# padding
def add_X_padding(smiles_list):
    """Add one 'X' to the end of each SMILES string.""" # why? is it <eos> token?
    for i in range(len(smiles_list)):
        smiles_list[i] += 'X'

def max_padding(smiles_list, max_len): # RNN에서 사용 X, CNN에서 사용 (기본 방식)
    """Pad each SMILES string to the maximum length with 'X'."""
    for i in range(len(smiles_list)):
        smiles_list[i] = smiles_list[i].ljust(max_len, 'X')

# Calculate logP
def calculate_logp(smiles_list):
    """Return a list of logP values for each SMILES string."""
    logp_values = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles) # Convert a SMILES string to a molecule object
        logp = MolLogP(molecule) # Calculate the logP value of the molecule object
        logp_values.append(logp)
    return logp_values

# Collate function # To do: revise
def collate_fn(batch):
    sequence_batch = []
    length_batch = []
    logp_batch = []

    for sample in batch:
        sequence_batch.append(sample["seq"])     # SMILES sequence
        length_batch.append(sample["length"])    # valid sequence length
        logp_batch.append(sample["logp"])        # logP value(target value)

    padded_seq = pad_sequence(
        sequence_batch,
        batch_first=True,
        padding_value=45  # 45: 'X' token
    )

    return {
        "seq": padded_seq,                                 # padded SMILES sequence
        "length": torch.tensor(length_batch),              # valid sequence length (not padded)
        "logp": torch.tensor(logp_batch, dtype=torch.float)  # logP value (not padded)
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
def train(model, train_dataloader, valid_dataloader, optimizer, device, args):
    model.train()
    train_loss_history = []
    train_start_time = time.time()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        since = time.time()

        loop = tqdm(train_dataloader, leave=True)
        train_loss_list = []

        for batch_idx, batch in enumerate(loop):
            input_batch = batch["seq"].to(device)
            length_batch = batch["length"].to(device)
            label_batch = batch["logp"].to(device)     

            # 모델 Forward & Loss 계산
            output_batch = model(input_batch, length_batch)
            output_batch = output_batch.squeeze(-1)

            loss_fct = nn.MSELoss(reduction="sum") # To do: Move to main
            loss = loss_fct(output_batch, label_batch)
            train_loss_list.append(copy.deepcopy(loss.detach().cpu().numpy()))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping # prevent exploding gradient
            optimizer.step()

            # evaluation

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
                
        epoch_train_avg_loss = np.sum(np.array(train_loss_list)) / len(train_dataloader.dataset) # divide by the number of dataset samples, not batch samples 
        train_loss_history.append(epoch_train_avg_loss)
        validate(model, valid_dataloader, device, epoch)
        save_checkpoint(model, optimizer, epoch, global_step, loss, args.output_dir, "epoch")
        print(f"Epoch {epoch + 1} Loss: {epoch_train_avg_loss:.4f} Time: {time.time() - since:.2f}s")

# Validation
def validate(model, dataloader, device, epoch):
    model.eval()
    valid_loss_history = []
    valid_loss_list = []
    best_valid_loss = float("inf")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_batch = batch["seq"].to(device)
            length_batch = batch["length"].to(device)
            label_batch = batch["logp"].to(device)

            # 모델 Forward & Loss 계산
            output_batch = model(input_batch, length_batch)
            output_batch = output_batch.squeeze(-1)

            loss_fct = nn.MSELoss(reduction="sum") # To do: Move to main
            loss = loss_fct(output_batch, label_batch)
            valid_loss_list.append(copy.deepcopy(loss.detach().cpu().numpy()))

            # wandb에 로그 기록
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
        best_model_weight = copy.deepcopy(model.state_dict()) # To do: save best model(save checkpoint)
        print(f"Best model at epoch {best_epoch} with loss {best_valid_loss:.4f}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # 난수 고정
    set_seed(42) 

    # Load data # To do: move to argparse
    max_len = 64
    num_data = 20000
    smiles_list = load_data(args.data_path, max_len, num_data)
    logp_list = calculate_logp(smiles_list)
    add_X_padding(smiles_list)
    #max_padding(smiles_list, max_len)
    tokenizer = load_tokenizer(args.tokenizer_path)
    #print(f'n_char: {len(tokenizer)}') # 46
    #print(f'X: {tokenizer['X']}') # 45

    # Split data # To do: Modify to train_test_split
    train_data = SMILESDataset(smiles_list[:16000], logp_list[:16000], tokenizer)
    valid_data = SMILESDataset(smiles_list[16000:18000], logp_list[16000:18000], tokenizer)
    test_data = SMILESDataset(smiles_list[18000:20000], logp_list[18000:20000], tokenizer)

    # DataLoader
    data_loaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),
        "valid": DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    }

    # Prepare model 
    model = RNNRegressor(n_feature=128, n_rnn_layer=1, n_char=len(tokenizer), layer_type='GRU')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # checkpoint 불러오기
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=f"cuda:{args.cuda_device}")

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"]) # checkpoint에 state_dict외 다른 것도 포함된 경우
        else:
            model.load_state_dict(checkpoint)  # 직접 불러오기 (state_dict만 저장된 경우)

    # GPU 사용 가능 여부 확인
    #device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu") # GPU 번호 지정
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # wandb 초기화
    wandb.init(project="Tutorial_ACE", name="RNN-training", config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })
    

    # Optimizer 설정
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 모델 학습 실행
    train(model, data_loaders["train"], data_loaders["valid"], optimizer, device, args)

    # Save final model # To do: Add

if __name__ =="__main__":
    main()