import numpy as np
import torch
from torchmetrics import Metric, MeanSquaredError

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