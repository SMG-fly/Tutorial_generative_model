import os
import json
import pathlib
import pickle
from typing import Dict
from functools import partial
from types import SimpleNamespace

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

def load_tokenizer(tokenizer_path : str) -> 'TokenizerWrapper': 
    """Load the tokenizer from a pkl file."""
    with open(tokenizer_path, "rb") as file:
        vocab_dict = pickle.load(file) 
    
    if not isinstance(vocab_dict, dict):
        raise TypeError("Tokenizer file must contain a dictionary.")
    
    # check if there are special tokens
    if 'X' not in vocab_dict:
        raise ValueError("Tokenizer must include a padding token 'X'.")

    return TokenizerWrapper(vocab_dict)

class TokenizerWrapper:
    def __init__(self, vocab_dict: dict, pad_token='X', mask_token='?'):
        self.vocab = vocab_dict
        self.vocab_size = len(vocab_dict)
        self.inverse_vocab = {v: k for k, v in vocab_dict.items()}

        self.pad_token = pad_token
        self.mask_token = mask_token
        self.pad_token_idx = self.vocab[pad_token]
        self.mask_token_idx = self.vocab[mask_token]

    def token_to_idx(self, token: str) -> int:
        return self.vocab.get(token, self.pad_token_idx)  # fallback: pad

    def idx_to_token(self, idx: int) -> str:
        return self.inverse_vocab.get(idx, '?')

    def encode(self, smiles: str) -> list:
        encode = [self.token_to_idx(ch) for ch in smiles]
        return encode

    def decode(self, idx_seq: list) -> str:
        decode = ''.join([self.idx_to_token(idx) for idx in idx_seq if idx != self.pad_token_idx])
        return decode


class SMILESDataset(Dataset): # To do: init으로 어떤 게 필요한지, getitem 할 때 어떤 형태가 필요한지 확인하고 수정하기 (지금은 GPT 코드)
    def __init__(self, file_path, tokenizer, max_length):
        self.data = self._load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def _load_data(self, path: str):
        data = []
        with open(path, "r") as f:
            for line in f:
                splits = line.strip().split(",") # ,이 있으면 뒷 부분을 y 함 # 없으면 None처리 # 그냥 Dataset 폴더만 갈아끼우면 y 적용할 수 있음 
                smiles = splits[0]
                y = float(splits[1]) if len(splits) > 1 else None
                data.append((smiles, y)) # None 값도 append가 되나? 
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles, y_val = self.data[idx]
        token_ids = self.tokenizer.encode(smiles)[:self.max_length]
        pad_len = self.max_length - len(token_ids)
        input_ids = token_ids + [self.tokenizer.pad_token_idx] * pad_len
        pad_mask = [1] * len(token_ids) + [0] * pad_len

        sample = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
        }
        if y_val is not None:
            sample["y"] = torch.tensor(y_val, dtype=torch.float32) 

        return sample


class SequenceCollator: 
    def __init__(self, pad_token_idx):
        self.pad_token_idx = pad_token_idx

    def __call__(self, batch): # collate_fn으로 사용됨
        seqs = [item['input_ids'] for item in batch]
        padded_x0 = pad_sequence(seqs, batch_first=True, padding_value=self.pad_token_idx)
        pad_mask = (padded_x0 != self.pad_token_idx)
 
        # optional y
        if 'y' in batch[0]:
            y = torch.stack([item['y'] for item in batch])
        else:
            y = None

        return SimpleNamespace(x=padded_x0, pad_mask=pad_mask, y=y)


class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer(cfg.dataset.tokenizer_path)  # 예: .pkl or .json
        self.max_length = cfg.dataset.max_length
        self.root_path = cfg.dataset.datadir

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
        collator = SequenceCollator(pad_token_idx=self.tokenizer.pad_token_idx)
        return DataLoader(self.train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True, collate_fn=collator)

    def val_dataloader(self):
        collator = SequenceCollator(pad_token_idx=self.tokenizer.pad_token_idx)
        return DataLoader(self.val_dataset, batch_size=self.cfg.train.batch_size, shuffle=False, collate_fn=collator)

    def test_dataloader(self):
        collator = SequenceCollator(pad_token_idx=self.tokenizer.pad_token_idx)
        return DataLoader(self.test_dataset, batch_size=self.cfg.train.batch_size, shuffle=False, collate_fn=collator)


class SequenceDatasetInfos: # pytorch lightning과는 무관, DatasetInfo class를 사용하면서 vocab_size, pad_idx 등등을 따로따로 넘기지 않고 하나의 객체로 넘길 수 있게 됨.
    # To do: 어떤 정보를 넘겨주면 좋은지 확인하고 수정하기
    """
    시퀀스 데이터셋(SMILES 등)을 위한 메타정보 클래스.
    모델 입력/출력 차원, vocab size, pad token 등 저장.
    """
    def __init__(self, datamodule, cfg): # complete info로 고쳐야할 수도? (Github 참고)
        self.tokenizer = datamodule.tokenizer     # already TokenizerWrapper
        self.max_seq_len = cfg.dataset.max_length

        # 속성은 tokenizer에서 바로 가져옴
        self.pad_token_idx = self.tokenizer.pad_token_idx
        self.mask_token_idx = self.tokenizer.mask_token_idx
        self.vocab_size = self.tokenizer.vocab_size

        # 시퀀스 기반이므로 X만 input/output feature로 사용 # 아래 input_dim, output_dim이랑 겹치는데 model에서 제대로 사용하는지 확인 필요할듯 # To do: check # y도 feature가 있을 경우에는 수정 필요
        self.input_dims = {'X': self.vocab_size, 'y': 0}
        self.output_dims = {'X': self.vocab_size, 'y': 0}

    def compute_input_output_dims(self, extra_features=None, domain_features=None): # datamodule 지웠음 -> 어디서 인자 오류가 나는지 확인해야함 (검거 완.) # 또 있을지 모르니 일단 남겨둠
            """
            datamodule을 통해 input/output 차원을 계산해 저장
            - input_dims: {'X': vocab_size + extra_dim, 'y': ...}
            - output_dims: {'X': vocab_size, ...}
            """
            # SMILES이므로 'X'에만 적용
            X_dim = self.vocab_size
            extra_dim = extra_features.get_input_dim() if extra_features else 0
            domain_dim = domain_features.get_input_dim() if domain_features else 0 # domain_features를 어떻게 해야할지가 고민이네... # To do: check

            # 보통 output은 vocab_size 고정, input은 조건 추가
            self.input_dims = {'X': X_dim + extra_dim, 'y': domain_dim} # To do: check # domain_features....이거 필요한가...? # condition을 어떻게 줘야할까?
            self.output_dims = {'X': X_dim, 'y': domain_dim}

