import argparse
import os
import pickle
import random

import numpy as np
import torch
from rdkit import Chem

from VAE_model import SmVAE

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on a trained SMILES VAE model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer pkl file")
    parser.add_argument("--num_generate", type=int, default=100, help="Number of SMILES to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save generated SMILES")
    #parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device number to use")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_rnn_layers", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=64)
    return parser.parse_args()

# Set seed
def set_seed(seed: int = 42):
    torch.manual_seed(seed)  # PyTorch의 난수 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU의 난수 고정 (GPU 사용 시)
    random.seed(seed)  # Python 기본 random 라이브러리 고정
    np.random.seed(seed)  # NumPy 난수 고정
    torch.backends.cudnn.deterministic = True  # CUDNN 연산 결정적 사용
    torch.backends.cudnn.benchmark = False  # 성능 최적화 방지 (재현성을 위해)

def load_tokenizer(tokenizer_path : str) -> tuple[dict, dict]: 
    """Load the tokenizer from a pkl file."""
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file) 
    
    if not isinstance(tokenizer, dict):
        raise TypeError("Tokenizer file must contain a dictionary.")

    index_to_char = {idx: char for char, idx in tokenizer.items()}

    return tokenizer, index_to_char

def decode_smiles(index_sequence: torch.Tensor, index_to_char: dict) -> str:
    """Convert a sequence of indices to a SMILES string."""
    decoded_smiles = []

    for idx in index_sequence:
        index = idx.item() # tensor -> int
        char = index_to_char.get(index, '') # get the character corresponding to the index # if index not in index_to_char, return empty string
        decoded_smiles.append(char)

    decoded_smiles = ''.join(decoded_smiles) # Combine characters from a list into a single string
    
    return decoded_smiles

def sample_latent_vectors(latent_dim: int, num_samples: int, device: torch.device) -> torch.Tensor:
    """Sample latent vectors from a standard normal distribution."""
    return torch.randn(num_samples, latent_dim).to(device)

def generate_smiles_from_latents(model: SmVAE, latent_vectors: torch.Tensor, index_to_char: dict, max_len: int) -> list[str]:
    model.eval()
    generated_smiles = []
    generated_sequences = model.inference(latent_vectors, max_length=max_len)  # [batch_size, seq_len]

    for sequence in generated_sequences:
        smiles = decode_smiles(sequence, index_to_char)
        smiles = smiles.split('X')[0]  # Cut at <eos> or padding token # To do: cut <sos> token
        generated_smiles.append(smiles)

    return generated_smiles

def validate_smiles(smiles_list: list[str]) -> list[str]:
    valid_smiles = []
    for smi in smiles_list:
        if Chem.MolFromSmiles(smi):
            valid_smiles.append(smi)
    return valid_smiles

def save_generated_smiles(all_smiles: list[str], valid_smiles: list[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_path = os.path.join(output_dir, "generated_all.txt")
    valid_path = os.path.join(output_dir, "generated_valid.txt")
    
    with open(all_path, 'w') as f:
        for smiles in all_smiles:
            f.write(f"{smiles}\n")
    print(f"Saved all generated SMILES to: {all_path}")        

    with open(valid_path, 'w') as f:
        for smiles in valid_smiles:
            f.write(f"{smiles}\n")
    print(f"Saved valid SMILES to: {valid_path}")

def inference(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer, index_to_char = load_tokenizer(args.tokenizer_path)

    # Load model
    model = SmVAE(vocab_size=len(tokenizer), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim, num_rnn_layers=args.num_rnn_layers)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"]) # checkpoint에 state_dict외 다른 것도 포함된 경우
    else:
        model.load_state_dict(checkpoint)  # 직접 불러오기 (state_dict만 저장된 경우)
    model.to(device)

    # Sample latent vectors
    latent_vectors = sample_latent_vectors(latent_dim=args.latent_dim, num_samples=args.num_generate, device=device)

    # Generate SMILES
    generated = generate_smiles_from_latents(model, latent_vectors, index_to_char, args.max_len)
    valid = validate_smiles(generated)

    # Save outputs
    save_generated_smiles(generated, valid, args.output_dir)

    # Print summary
    print(f"\nSuccess Rate: {len(valid)}/{args.num_generate} ({len(valid)/args.num_generate*100:.2f}%)")

def main():
    #set_seed(42) 
    args = parse_arguments()
    inference(args)

if __name__ =="__main__":
    main()
