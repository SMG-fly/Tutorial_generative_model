import numpy as np
import torch
from torchmetrics import Metric, MeanSquaredError

import os
import wandb
import omegaconf
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

# Utils --------------------------------
class PlaceHolder: # To do: Check And Modify argmax
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def pad(self, pad_mask, convert_to_idx=False): # padding 부분을 학습에서 무시하게 해주는 역할
        if convert_to_idx:
            self.X = torch.argmax(self.X, dim=-1)  # 확률 최대값 → 클래스 인덱스 # (B, L, V) -> (B, L)
            self.X[pad_mask == 0] = -1 # padding 부분은 -1로 마스킹
            # -1은 "이 위치는 학습/평가에서 제외해야 한다"는 신호
            # To do: F.cross_entropy(input, target, ignore_index=-1) 로 -1인 부분은 무시하고 loss 계산

            if self.y is not None:
                if self.y.dim() == 3: # (B, L, d)
                    self.y[pad_mask == 0] = 0
                else:
                    pass

        else: # softmax 상태에서 마스킹
            self.X = self.X * pad_mask.unsqueeze(-1) 
            # self.S.shape = (B, L, V) ← 각 위치에서 V개의 클래스 확률을 가짐
            # # seq_mask.shape = (B, L) ← 1은 유효한 위치, 0은 padding
            # soft 분포인 상태에서 padding 위치는 0으로 날리는 것

            if self.y is not None:
                if self.y.dim() ==3: # (B, L, V)
                    self.y = self.y * pad_mask.unsqueeze(-1) 
        return self

def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) # .yaml 파일을 python dict로 변환
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict, # wandb.init()에 전달할 설정 값
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt') # To do: Check whethere this is necessary # maybe memory waste

def update_config_with_new_keys(cfg, saved_cfg):
    # 중요한 keys만 명시적으로 병합
    for section in ['general', 'train', 'model']:
        omegaconf.OmegaConf.set_struct(getattr(cfg, section), True) # 구조 고정 모드 → 새로운 key를 직접 추가하면 에러가 나게 설정 # 전체 구조 보호
        with omegaconf.open_dict(getattr(cfg, section)): # 위 구조고정을 with open_dict(...) 안에서만 일시적으로 해제해서 수동으로 key를 넣을 수 있게 해줌
            for key, value in getattr(saved_cfg, section).items():
                if key not in getattr(cfg, section).keys(): # ckpt에 있는 key가 test의 cfg에 없으면 # 이미 존재하는 key는 덮어쓰지 않음 (덮으면 위험할 수 있으니까)
                    setattr(getattr(cfg, section), key, value) # cfg.section.key = value와 같은 역할 # python 내장 함수 

    return cfg # 업데이트된 cfg 반환

def create_folders(cfg): # 필요한 folder를 미리 만들어두는 함수 # To do: customize
    os.makedirs(os.path.join('checkpoints', cfg.general.name), exist_ok=True)

    # 필요한 경우만
    if cfg.general.save_chain_steps:
        os.makedirs(os.path.join('chains', cfg.general.name), exist_ok=True)

# Diffusion Utils ----------------------

def absorbing_beta_schedule_discrete(timesteps: int):
    """ Absorbing-state beta schedule: beta_t = 1 / (T - t + 1) """
    betas = np.array([1 / (timesteps - t + 1) for t in range(timesteps)], dtype=np.float32)
    return betas

def pad_distributions(true_X, pred_X, pad_mask, pad_token_idx):
    """
    Padding된 토큰 위치는 loss에 영향을 주지 않도록 정해진 분포로 대체하고 정규화.

    Args:
        true_X: (batch, L, vocab_size)
        pred_X: (batch, L, vocab_size)
        pad_mask: (batch, L) — True: 유효한 토큰, False: padding 위치

    Returns:
        masked true_X, pred_X (shape 동일)
    """
    # padding token만 1인 one-hot vector
    padding_row = torch.zeros(true_X.size(-1), device=true_X.device) # (vocab_size,) 
    padding_row[pad_token_idx] = 1. # padding token

    # 마스킹 적용(padding 위치에 one-hot 분포 덮어쓰기)
    # pad_mask는 bool 타입
    true_X[~pad_mask] = padding_row
    pred_X[~pad_mask] = padding_row

    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7

    # vocab_size 차원에서 정규화 (합이 1이 되도록)
    true_X = true_X / true_X.sum(dim=-1, keepdim=True)
    pred_X = pred_X / pred_X.sum(dim=-1, keepdim=True)

    return true_X, pred_X # (batch, L, vocab_size)

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def sample_discrete_features(probX, pad_mask, pad_idx = None):
    '''
    Sample discrete token indices from a multinomial distribution over vocabulary at each timestep.

    Args:
        probX: (bs, L, vocab_size)
            Probability distribution over vocabulary at each position.
        pad_mask: (bs, L)
            Boolean mask indicating valid (True) vs padding (False) positions.
    
    Returns:
        PlaceHolder with:
            X: (bs, L) - token indices sampled from probX
            y: (bs, 0) - dummy global features (empty)
    '''
    batch, length, vocab_size = probX.shape

    # 1. Masking 처리
    if pad_idx: # padding 위치는 전부 pad만 뽑도록
        probX[~pad_mask] = 0
        probX[~pad_mask, pad_idx] = 1
    else: # padding 위치는 uniform distribution으로 sampling
        probX[~pad_mask] = 1.0 / vocab_size

    # 2. Flatten 후 multinomial sampling
    probX_flat = probX.reshape(batch * length, vocab_size) # torch.multinomial()은 2D tensor만 받으므로

    # 3. 각 위치에서 1개 token을 sampling (Multinomial distribution)
    X_t = probX_flat.multinomial(1) # 각 위치에서 token 하나만 뽑고 싶으니까 1 # (batch * length, 1)
    X_t = X_t.reshape(batch, length) 

    # 4. global feature y는 사용하지 않으므로 빈 텐서로 대체 
    # placeholder 자리 채우기용
    y_dummy = torch.zeros(batch, 0).type_as(X_t)

    return PlaceHolder(X=X_t, y=y_dummy)

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    # Flatten featrue tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32) # (Batch, Length, Vocab) # X는 3차원이라서 차원 변화 없음 # flatten은 edge를 위한 것
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  

    Qt_M_T = torch.transpose(Qt_M, -2, -1) # (Batch, Vocab, Vocab)

    left_term = M_t @ Qt_M_T # (Batch, Length, Vocab)
    right_term = M @ Qsb_M # (Batch, Length, Vocab)
    product = left_term * right_term # (Batch, Length, Vocab)

    denom = M @ Qtb_M
    denom = (denom * M_t).sum(dim=-1) # 이게... matmul이 아니었다고?

    prob = product / denom.unsqueeze(-1)

    return prob

def posterior_distribution(X, y, X_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt, Qsb_M=Qsb, Qtb_M=Qtb) # Debugging할 때 Qt가 들어가는 게 맞는지 확인
    
    return PlaceHolder(X=prob_X, y=y_t)

def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb): # train 때 구한 posterior랑 구현 방식(수식)이 다른데? 어떻게 된 거지? # To do: 그냥 내 맘대로 posterior 하나 구현하는 게 나을지도...
    """ 
        M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 # To do: 아니 앞에서는 elemwise 곱이었는데 왜 여기서는 matmul이지?
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    Qt_T = Qt.transpose(-1, -2) # (bs, d_t, d_t-1)
    left_term = X_t @ Qt_T # (bs, n, dt-1) # X_t는 (bs, n, dt) # X_t @ Qt_T = X_t @ Qt.transpose(-1, -2)
    left_term = left_term.unsqueeze(dim=2) # (bs, n, 1, dt-1) # broadcasting을 위해서

    right_term = Qsb.unsqueeze(1) # (bs, 1, d_0, d_t-1) # Qsb는 (bs, d0, d_t-1)
    numerator = left_term * right_term # (bs, n, d_0, d_t-1) # broadcasting을 통해서 곱해짐

    X_t_transposed = X_t.transpose(-1, -2) # (bs, d_t, n) # To do: check # 여기서 부터 train, paper와 다른 방식으로 posterior 구현됨

    prod = Qtb @ X_t_transposed # (bs, d_0, n) # Qtb는 (bs, d0, dt) # Q. train에서는 elementwise로 하더만!!! # X_0는 또 어디갔어? # x_0를 pred해서 곱해줘야 하는 거 아냐?
    prod = prod.transpose(-1, -2) # (bs, n, d_0)
    denominator = prod.unsqueeze(dim=-1) # (bs, n, d_0, 1) # broadcasting을 위해서
    denominator[denominator == 0] = 1e-6 # 0인 부분은 1e-7로 대체

    out = numerator / denominator # (bs, n, d_0, d_t-1) # 각 위치에서 vocab_size에 대한 확률 분포를 구함
    return out # 모든 가능한 x0에 대해 q(x_{t-1}|x_t, x_0)를 계산한 것

# Noise_schedule -----------------------
class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == 'absorbing':
            betas = absorbing_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999) # clamp를 통해 beta가 1에 너무 가까워지는 것을 방지 (정보 보존이 하나도 안됨 -> 학습이 안됨)

        log_alpha = torch.log(self.alphas) # 로그로 변환
        log_alpha_bar = torch.cumsum(log_alpha, dim=0) # 로그의 누적합
        self.alphas_bar = torch.exp(log_alpha_bar) # 다시 원래 스케일로 변환 (alpha의 누적곱 완성)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        return self.get_beta(t_normalized, t_int)

    def get_beta(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1 # 둘 중에 하나만 들어와야 함.
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()] # 정수형 인덱스의 beta를 가져온다.

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()] # 정수형 인덱스의 alpha_bar를 가져온다.
    

class AbsorbingStateTransition:
    def __init__(self, abs_state: int, num_classes: int):
        self.abs_state = abs_state
        self.num_classes = num_classes

        # u: 모든 state가 abs_state로 전이될 확률이 1인 행렬
        self.u = torch.zeros(1, self.num_classes, self.num_classes)
        self.u[:, :, self.abs_state] = 1 # Absorbing state만 1

    def get_Qt(self, beta_t):
        """ 
        Returns transition matrix 

        beta_t: (B,) 확률 (t 시점에서 noise를 얼마나 섞을지) # B: batch size
        return: q_t: (B, C, C) transition matrix at time t
        """
        beta_t = beta_t.unsqueeze(1) # (B, 1)
        eye = torch.eye(self.num_classes).unsqueeze(0) # (1, C, C)
        q_t = beta_t * self.u + (1 - beta_t) * eye # (B, C, C)
        return q_t

    def get_Qt_bar(self, alpha_bar_t): # 논문엔 이렇게 구한다는 말 없었던 거 같은데... 어쨌든 그 논문 코드니까 ㅇㅋ
        """
        alpha_bar_t: (B,) 누적 alpha
        return: (B, C, C) cumulative transition matrix
        """

        alpha_bar_t = alpha_bar_t.unsqueeze(1) # (B, 1)
        eye = torch.eye(self.num_classes).unsqueeze(0) # (1, C, C)
        q_t_bar = alpha_bar_t * eye + (1 - alpha_bar_t) * self.u # (B, C, C)
        return q_t_bar


# Abstract metrics ----------------------
class SumExceptBatchMetric(Metric): # mean over batch
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default = torch.tensor(0.), dist_reduce_fx = "sum") # Metric의 내장 함수 # (변수 이름, 초기값, aggregation 방법)
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0] # batch size

    def compute(self) -> torch.Tensor:
        return self.total_value / self.total_samples
    
class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self) -> torch.Tensor:
        return self.total_value / self.total_samples

class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:    
        self.total_nll += torch.sum(batch_nll) # 현재 batch의 nll 누적
        self.total_samples += batch_nll.numel() # tensor의 전체 원소 개수

    def compute(self) -> torch.Tensor:
        return self.total_nll / self.total_samples

class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None: # In training phase, pred : \hat{x}_0, target : x_0
        """ 
        Update state with predictions and targets.
        preds: (B, L, V) # softmax된 확률 분포
        targets: (B, L, V) # Ground Truth values 
        """
        target = torch.argmax(target, dim=-1) # target도 분포로 받나봐... 근데 굳이 argmax로 해줘야 하나? 그냥 분포끼리 비교하는 게 더 낫지 않나? # To do: Check
        output = F.cross_entropy(preds, target, reduction='sum') # (B, L) # 각 위치에서의 cross entropy loss
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples

# Train metrics ----------------------
class TrainLossDiscrete(nn.Module):
    """ Train with Cross Entropy Loss """
    def __init__(self, lambda_train): # lambda_train: loss_x, loss_y 비율 조정
        super().__init__()
        self.X_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train
    
    def forward(self, masked_pred_X, pred_y, true_X, true_y, log: bool):
        """ 
        Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_y : tensor -- (bs, )
        log : boolean. 
        """
        # To do: 아래 부분 아직 분석 안 함. 이해하고 아래 reset으로 넘어가기
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx) # 마지막 차원 유지, 앞 차원은 자동 계산 # 시퀀스 전체를 token 단위로 비교하기 위해
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1) # Vocab 차원에서 0이 아닌 값이 하나라도 있으면 True # pred는 이미 masking 되어 있어서 필요 없음 

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        loss_X = self.X_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0 # numel: tensor 안에 있는 모든 원소의 개수를 반환하는 함수 # 데이터가 없으면 0으로 대체
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0 # To do: Check # y는 전처리 필요 없나?

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_y).detach(),
                      "train_loss/X_CE": self.X_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train * loss_y 

    def reset(self):
        for metric in [self.X_loss, self.y_loss]:
            metric.reset() # torchmetrics.Metric의 reset() 메서드 호출

    def log_epoch_metrics(self):
        epoch_node_loss = self.X_loss.compute() if self.X_loss.total_samples > 0 else -1 # total ce (mean) 도출
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1 

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log


# spectre_utils.py --------------------
class TrainSequenceMetrics: # 아니 근데 이러면 앞에 했던 것들이랑 중복 아닌가? 그리고 이 함수도 확인 후 수정 필요 # To do: check
    def __init__(self, dataset_infos):
        self.pad_token_idx = dataset_infos.pad_token_idx
        self.vocab_size = dataset_infos.vocab_size
        self.reset() # 이거 내장 함수야?

    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.correct_tokens = 0

    def update(self, pred_logits, x_0, pad_mask): # model에서 다 계산을 해놨을텐데 왜 이렇게 하는 거지? # To do: check
        """
        pred_logits: (B, L, V)
        x_0: (B, L)
        pad_mask: (B, L) where 1 = real token, 0 = pad
        """
        B, L, V = pred_logits.shape

        # flatten
        logits = pred_logits.view(-1, V)       # (B*L, V)
        targets = x_0.view(-1)                 # (B*L,)
        mask = pad_mask.view(-1).bool()        # (B*L,)

        # loss
        loss = F.cross_entropy(logits[mask], targets[mask], reduction='sum')
        self.total_loss += loss.item()

        # accuracy
        preds = logits.argmax(dim=-1)          # (B*L,)
        correct = (preds == targets) & mask
        self.correct_tokens += correct.sum().item()
        self.total_tokens += mask.sum().item()

    def compute(self):
        avg_loss = self.total_loss / max(self.total_tokens, 1)
        acc = self.correct_tokens / max(self.total_tokens, 1)
        return {
            "train/loss": avg_loss,
            "train/acc": acc,
            "train/tokens": self.total_tokens
        }
    
    
class SamplingSequenceMetrics: # 이 함수도 확인 후 수정 필요 # To do: check
    def __init__(self, dataset_infos, reference_set=None):
        self.tokenizer = dataset_infos.inverse_tokenizer  # index → token
        self.pad_token_idx = dataset_infos.pad_token_idx
        self.vocab_size = dataset_infos.vocab_size
        self.reference_set = set(reference_set or [])
        self.reset()

    def reset(self):
        self.decoded = []
        self.valid_set = set()
        self.total = 0

    def decode_sequence(self, idx_seq):
        # idx_seq: (L,)
        tokens = [self.tokenizer.get(idx, '') for idx in idx_seq if idx != self.pad_token_idx]
        return ''.join(tokens)

    def update(self, generated_batch):
        """
        generated_batch: Tensor (B, L)
        """
        for i in range(generated_batch.size(0)):
            seq = generated_batch[i].tolist()
            decoded = self.decode_sequence(seq)
            self.total += 1
            self.decoded.append(decoded)
            if self.is_valid(decoded):
                self.valid_set.add(decoded)

    def is_valid(self, seq: str):
        # 예시: SMILES 유효성 검사 → RDKit 또는 길이 기준 등
        return len(seq) > 0 and all(c.isalnum() or c in "=#()" for c in seq)

    def compute(self):
        total = self.total
        valid = len(self.valid_set)
        unique = len(set(self.decoded))

        novelty = 0
        if self.reference_set:
            novel_set = set(self.decoded) - self.reference_set
            novelty = len(novel_set) / total

        return {
            "sampling/validity": valid / total,
            "sampling/uniqueness": unique / total,
            "sampling/novelty": novelty,
            "sampling/total": total
        }


# extra_features ----------------------
class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0)) # 이렇게 empty tensor를 줘도 되는 거야? 기존 x값까지 지워지겠어. extra라서 상관없나?
        empty_y = y.new_zeros((y.shape[0], 0))
        return PlaceHolder(X=empty_x, y=empty_y)


class ExtraFeatures: # To do: Check and modify
    def __init__(self, features_type: str, dataset_info):
        self.features_type = features_type
        self.max_length = dataset_info.max_seq_len

    def __call__(self, noisy_data):
        """
        Input:
            - noisy_data['X_t']: (B, L)      # tokenized input
            - noisy_data['pad_mask']: (B, L)
            - noisy_data['y_t']: (B, D)      # ex: logP, label
        Output:
            - PlaceHolder with X, E, y
        """
        X_t = noisy_data['X_t']
        pad_mask = noisy_data['pad_mask']
        y_t = noisy_data['y_t']  # logP (B, 1) or other label (B, D)
        B, L = X_t.shape

        seq_lens = pad_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        length_ratio = seq_lens / self.max_length             # (B, 1)

        # -------- Feature type 분기 -------- #
        if self.features_type == 'length':
            return PlaceHolder(
                X=X_t.new_zeros((B, L, 0)),
                E=None,
                y=length_ratio  # (B, 1)
            )

        elif self.features_type == 'logp':
            return PlaceHolder(
                X=X_t.new_zeros((B, L, 0)),
                E=None,
                y=y_t  # logP as global condition (B, 1)
            )

        elif self.features_type == 'length+logp':
            return PlaceHolder(
                X=X_t.new_zeros((B, L, 0)),
                E=None,
                y=torch.cat([length_ratio, y_t], dim=-1)  # (B, 2)
            )

        elif self.features_type == 'none':
            return PlaceHolder(
                X=X_t.new_zeros((B, L, 0)),
                E=None,
                y=None
            )

        else:
            raise ValueError(f"Unsupported extra feature type: {self.features_type}")
