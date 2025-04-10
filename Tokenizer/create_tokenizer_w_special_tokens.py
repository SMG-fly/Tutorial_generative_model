import argparse
import pickle
import json
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert a tokenizer pickle file to a JSON file.")
    parser.add_argument("-i", "--smiles_path", type=str, required=True, help="Path to input SMILES text file")
    parser.add_argument("-o", "--save_path", type=str, default="./tokenizer/tokenizer.pkl", help="Path to save the tokenizer .pkl file")
    return parser.parse_args()

def build_tokenizer_from_file(smiles_path: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(smiles_path, "r") as f:
        lines = f.readlines()

    smiles_chars = sorted(set("".join(line.strip() for line in lines)))

    # Define special tokens (order matters!)
    special_tokens = ['X', '^', '$', '?'] # ['<pad>', '<sos>', '<eos>', '<unk>']

    # Remove duplicates if any special tokens are already in SMILES chars
    smiles_chars = [ch for ch in smiles_chars if ch not in special_tokens]

    # Combine special tokens and SMILES characters
    all_tokens = special_tokens + smiles_chars

    # Build tokenizer dict
    char_to_index = {char: idx for idx, char in enumerate(all_tokens)}

    # Save as pickle
    with open(save_path, "wb") as f:
        pickle.dump(char_to_index, f)

    # Optional: also save as JSON for human readability
    with open(save_path.replace(".pkl", ".json"), "w") as f:
        json.dump(char_to_index, f, indent=2)

    print(f"Tokenizer saved to {save_path} and JSON version to {save_path.replace('.pkl', '.json')}")

    return char_to_index

def main():
    args = parse_arguments()
    build_tokenizer_from_file(args.smiles_path, args.save_path)

if __name__ == "__main__":
    main()
