import os
import json
import pathlib
import pickle
from typing import Dict

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

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

def collate_fn(batch, pad_token_idx: int):
    """
    batch: List of dicts from SMILESDataset
    Returns:
        - padded_input: (B, L)
        - pad_mask: (B, L) where 1 = real token, 0 = pad
    """
    seqs = [item['input_ids'] for item in batch]    # List[Tensor] # input_ids는 SMILESDataset에서 반환하는 키
    padded_x0 = pad_sequence(seqs, batch_first=True, padding_value=pad_token_idx)
    pad_mask = (padded_x0 != pad_token_idx)
    return {'x_0': padded_x0, 'pad_mask': pad_mask}

class SMILESDataset(Dataset): # To do: init으로 어떤 게 필요한지, getitem 할 때 어떤 형태가 필요한지 확인하고 수정하기 (지금은 GPT 코드)
    def __init__(self, path, tokenizer, max_length):
        self.smiles_list = self._load_smiles(path)
        self.tokenizer = tokenizer
        self.max_length = max_length # 여기에 y 정보를 추가해야 한다는 거지? condition 용으로?

    def _load_smiles(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        token_ids = self.tokenizer.encode(smiles)
        token_ids = token_ids[:self.max_length]
        pad_len = self.max_length - len(token_ids)
        input_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
        pad_mask = [1] * len(token_ids) + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids),
            "pad_mask": torch.tensor(pad_mask),
        }

class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer(cfg.dataset.tokenizer_path)  # 예: .pkl or .json
        self.max_length = cfg.dataset.max_length

        base_path = pathlib.Path(__file__).resolve().parents[2]
        self.root_path = os.path.join(base_path, cfg.dataset.datadir)

    def setup(self, stage=None):
        self.train_dataset = SMILESDataset(
            file_path=os.path.join(self.root_path, "train.smi"),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.val_dataset = SMILESDataset(
            file_path=os.path.join(self.root_path, "val.smi"),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.test_dataset = SMILESDataset(
            file_path=os.path.join(self.root_path, "test.smi"),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)


class SequenceDatasetInfos: # pytorch lightning과는 무관, DatasetInfo class를 사용하면서 vocab_size, pad_idx 등등을 따로따로 넘기지 않고 하나의 객체로 넘길 수 있게 됨.
    # To do: 어떤 정보를 넘겨주면 좋은지 확인하고 수정하기
    """
    시퀀스 데이터셋(SMILES 등)을 위한 메타정보 클래스.
    모델 입력/출력 차원, vocab size, pad token 등 저장.
    """
    def __init__(self, tokenizer: Dict[str, int], max_seq_len: int):
        self.tokenizer = tokenizer                      # 문자 -> 인덱스
        self.inverse_tokenizer = {v: k for k, v in tokenizer.items()}  # 인덱스 -> 문자
        self.max_seq_len = max_seq_len

        self.pad_token = 'X'
        self.pad_token_idx = tokenizer[self.pad_token]  # 예: 53
        self.vocab_size = len(tokenizer)

        # 시퀀스 기반이므로 X만 input/output feature로 사용
        self.input_dims = {'X': self.vocab_size, 'y': 0}
        self.output_dims = {'X': self.vocab_size, 'y': 0}

    def idx_to_token(self, idx: int) -> str:
        return self.inverse_tokenizer.get(idx, '?')

    def token_to_idx(self, token: str) -> int:
        return self.tokenizer.get(token, self.pad_token_idx)

    def decode_sequence(self, idx_seq) -> str:
        """인덱스 시퀀스를 문자열로 디코딩"""
        return ''.join([self.idx_to_token(idx) for idx in idx_seq if idx != self.pad_token_idx])

    def encode_sequence(self, smiles: str) -> list:
        """문자열을 인덱스 시퀀스로 인코딩 (패딩 없음)"""
        return [self.token_to_idx(ch) for ch in smiles]

    def save(self, path: str):
        """info 객체를 JSON으로 저장"""
        info = {
            'tokenizer': self.tokenizer,
            'max_seq_len': self.max_seq_len,
            'pad_token': self.pad_token
        }
        with open(path, 'w') as f:
            json.dump(info, f)

    @classmethod
    def load(cls, path: str):
        """info 객체를 JSON에서 불러오기"""
        with open(path, 'r') as f:
            info = json.load(f)
        return cls(tokenizer=info['tokenizer'], max_seq_len=info['max_seq_len'])
