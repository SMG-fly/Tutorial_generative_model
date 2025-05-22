import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import os

import pytorch_lightning as pl

'''
from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
'''
from SmD3PM_utils import PlaceHolder # utils
from SmD3PM_utils import pad_distributions, sum_except_batch, sample_discrete_features, posterior_distribution, reverse_tensor, compute_batched_over0_posterior_distribution # diffusion_utils
from SmD3PM_utils import AbsorbingStateTransition, PredefinedNoiseScheduleDiscrete # noise_schedule
from SmD3PM_utils import SumExceptBatchMetric, SumExceptBatchKL, NLL # abstract_metrics

class X0Model(nn.Module):
    def __init__(self, n_channel: int, N: int = 16) -> None:
        super().__init__()
        pass # 모델 구조 # for predicting x0 at each time step # \hat p(x_0)


class SmD3PM(nn.Module):
    def __init__(self, cfg, tokenizer, dataset_infos, train_metrics, sampling_metrics, visualization_metrics, extra_features, domain_features): # To do: Modify for my model
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps # diffusion step 수 (1000 (paper)

        # tokenizer 기반 input/output dim 정의
        self.vocab_size = len(tokenizer) # tokenizer: dict {char: idx}
        self.pad_idx = 0
        self.start_token_idx = 1
        self.end_token_idx = 2
        
        self.mask_idx = 3

        self.tokenizer = tokenizer

        self.train_loss = TrainLossDiscrete() # To do: make loss function
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        # To do: 만들어야 함.
        self.extra_features = extra_features
        self.domain_features = domain_features

        # Diffusion model structure
        # GPT 추천 구조 -> DiGress 보면서 고치기
        self.embedding = nn.Embedding(self.vocab_size, cfg.model.embedding_dim, pad_idx=self.pad_idx)
        self.x0_model = nn.Transformer(
            d_model=cfg.model.embedding_dim,
            nhead=cfg.model.nhead, 
            num_encoder_layers=cfg.model.num_encoder_layers,
            dropout=cfg.model.dropout
        )
        self.output_head = nn.Linear(cfg.model.embedding_dim, self.vocab_size)

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=cfg.model.noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        if cfg.model.transition == "absorbing":
            abs_state = cfg.model.abs_state # "?" token
            self.transition_model = AbsorbingStateTransition(abs_state=abs_state, num_classes=self.vocab_size)

            # limit_dist: noise가 최대로 적용되었을 때 분포 abs_state만 1, 나머지 class는 0인 one-hot vector
            limit_dist = torch.zeros(self.vocab_size) # x_T가 도달해야하는 목적지
            limit_dist[abs_state] = 1.0
            
        # To do: Check
        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    # train.py ---------------------------
    def training_step(self, data, batch_idx):
        # data.x: (batch_size, seq_len) -> tokenized input sequences
        # data.pad_mask: (batch_size, seq_len) -> True if real token, False if pad
        input_seq = data.x # To do: check where is data
        pad_mask = data.pad_mask

        # 1. Apply noise to input data
        noisy_seq = self.apply_noise(input_seq, pad_mask, data.y)

        # 2. Compute additional features if needed (ex. t, time-step embedding)






    # training model ----------------------
    def kl_prior(self, X, pad_mask): # L_T
        """
        Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        모델이 생성한 분포가 얼마나 prior와 가까운지 계산 

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.

        Args:
        X: (bs, L, vocab_size)         # 시퀀스의 확률 분포 (softmax or one-hot)
        pad_mask: (bs, L)             # 유효 토큰 위치 (1), 패딩 (0)

        Returns:
        Scalar: 전체 시퀀스에 대한 KL divergence 합
        """
        ones = torch.ones((X.size(0), 1), device=X.device) # (Batch, 1)
        Ts = self.T * ones # (Batch, 1) # T: diffusion step 수 
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts) # (Batch, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device) # (Batch, Vocab, Vocab)

        probX = X @ Qtb # (Batch, Length, Vocab) # X_noise

        limit_X = self.limit_dist.X[None, None, :].expand_as(probX).type_as(probX) # (1, 1, Vocab) -> (Batch, Length, Vocab) broadcasting # 가정한 X_T의 목표 분포

        limit_X, probX = pad_distributions( # padding된 부분은 KL 계산에서 제외
            true_X=limit_X.clone(), # (Batch, Length, Vocab) # limit_X는 정적이고 재사용되므로 원본이 수정되지 않도록 clone() 
            pred_X=probX, # (Batch, Length, Vocab)
            pad_mask=pad_mask, # (Batch, Length)
        )

        kl_distance_X = F.kl_div( # 각 시점, 각 토큰 위치에서의 KL divergence (token 단위)
            input=probX.log(), target=limit_X, reduction='none'
        )

        return sum_except_batch(kl_distance_X) # Batch size 제외 전체 합산

    def compute_Lt(self, X, y, pred, noisy_data, pad_mask, test): # to do: feature y 추가 # To do: 질문 해결하기(x_0)
        pred_probs_X = F.softmax(pred.X, dim=-1) # pred.x_0의 logit
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], device=self.device) # (Batch, Vocab, Vocab)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], device=self.device) # (Batch, Vocab, Vocab)
        Qt = self.transition_model.get_Qt_bar(noisy_data['beta_t'], device=self.device) # (Batch, Vocab, Vocab)

        # Compute distributions to compare with KL
        batch, length, vocab = X.shape

        # 둘 다 true 같은데???? 뭐지???? x_t로 예측한 x_0가 전혀 들어가지 않았는데??
        # 그냥 x_0의 값이랑 X_0의 softmax를 넣은 차이 뿐인데?
        # To do: x0 model 추가 필요할듯 -> pred로 들어오는 거 같은데... 그럼 pred는 어디서 들어오는 거야?
        # 생각해보니 x0 끼리 비교하는 건데 posterior는 왜 구하는 거지?
        prob_true = posterior_distribution(X=X, y=y, X_t=noisy_data['X_t'], y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred = posterior_distribution(X=pred_probs_X, y=pred_probs_y, X_t=noisy_data['X_t'], y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)

        # Reshape and filter masked rows
        prob_true.X, prob_pred.X = pad_distributions(true_X=prob_true.X, pred_X=prob_pred.X, pad_mask=pad_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X)) # To do: 아니 근데 test랑 val이랑 다른가? 확인하기

        return self.T * kl_x # 전체 ELBO/NLL 추정을 위해 scale 복원

    def reconstruction_logp(self, t, X, pad_mask): 
        # compute noise values for t = 0
        # noise가 거의 없는 Q0를 곱한 x0를 기반으로 얼마나 x0를 잘 복원할 수 있는지 평가
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt_bar(beta_t=beta_0, device=self.device) # (Batch, Vocab, Vocab)

        probX0 = X @ Q0

        sampled0 = sample_discrete_features(probX=probX0, pad_mask=pad_mask, device=self.device) # (bs, L) # sampled t for each token

        X0 = F.one_hot(sampled0, num_classes=self.vocab_size).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape)

        sampled_0 = PlaceHolder(X=X0, y=y0).type_as(X0).pad(pad_mask=pad_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'y_t': sampled_0.y, 'pad_mask': pad_mask, 't': torch.zeros(X0.shape[0], 1).type_as(sampled0.y0)}
        extra_data = self.compute_extra_data(noisy_data)

        pred0 = self.forward(noisy_data, extra_data, pad_mask) # forward?? x_t -> x_0 예측하는 걸 보통 forward라고 하나?

        # Normalize prediction
        probX0 = F.softmax(pred0.X, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to the loss
        probX0[~pad_mask] = torch.ones(self.Xdim_output).type_as(probX0)

        return PlaceHolder(X=probX0, y=proby0) # -log p_θ(x₀) 계산을 위한 확률 분포 출력
        
    def apply_noise(self, X, y, pad_mask):
        """ Sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately \therefore lowest_t = 1
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(low=lowest_t, high=self.T + 1, size=(X.size(0), 1), device=X.device).float() # (bs, 1) # t ~ Uniform(0, T) for each sample in batch
        s_int = t_int - 1 # (bs, 1) # s = t - 1

        t_normalized = t_int / self.T # (bs, 1) # t / T
        s_normalized = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule.get_beta(t_normalized = t_normalized) # (bs, 1) # beta_t
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized = s_normalized) # (bs, 1) # alpha_s_bar
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized = t_normalized) # (bs, 1) # alpha_t_bar

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1 # row의 총 합이 1인가? (== normalize가 잘 되어 있는가?)

        # Compute tranisition probabilities
        probX = X @ Qtb.X # (bs, L, vocab) # X_noise

        # Sample noisy sequence
        sampled_t = sample_discrete_features(probX=probX, pad_mask=pad_mask, device=self.device) # (bs, L) # sampled t for each token

        # One-hot encoding
        X_t = F.one_hot(sampled_t, num_classes=self.vocab_size) # num classes
        assert (X.shape == X_t.shape)

        z_t = PlaceHolder(X=X_t, y=y).type_as(X_t).pad(pad_mask=pad_mask) # pad mask로 수정

        noisy_data = {'t_int': t_int, 't': t_normalized, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'y_t': z_t.y, 'pad_mask': pad_mask}

        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, y, pad_mask, test=False):
        """
        Computes sequence-based variational lower bound (VLB) loss.

        Args:
            pred: model output (contains pred.X: (bs, L, vocab), pred.y: (bs, dy))
            noisy_data: dict containing 't', 'alpha_t_bar', 'alpha_s_bar', 'beta_t', 'X_t', 'y_t'
            X:       (bs, L, vocab_size)     - input token one-hot or smoothed distribution
            y:       (bs, dy)                - label vector (e.g., molecular property)
            pad_mask: (bs, L)                - True for real token, False for padding
            test:    bool                    - whether to use test or validation loss metric

        Returns:
            nll: scalar, average NLL for the batch (nlls = -log_pN + kl_prior + loss_all_t - loss_term_0)
        """
        t = noisy_data['t'] # diffusion time t per sample

        # 1. Estimate log_p(N): node count log-prob
        # -log_p(N) 항 : 논문에는 언급되지 않은 항. 샘플링 시 길이 분포가 train set의 길이 분포와 비슷해지도록 유도
        # 구체적인 구현 방법을 Github에서 확인할 수 없음 -> 일단 논문에 있는 것 위주로 구현 위해 삭제 -> 샘플링할 때 길이 분포를 뽑으므로 그 때는 길이 분포 p(N)이 필요하긴 함 (To do)
        '''
        N = pad_mask.sum(1).long() # (bs,) # 각 시퀀스의 유효 토큰 개수
        log_pN = self.node_dist.log_prob(N)
        '''

        # 2. KL term: KL[q(x_T | x₀) || p(x_T)] using fixed limit_dist prior # L_T
        # q(x_T | x_0): forward process output
        # p(x_T): limit_dist (학습하고자 하는 모델의 목표 분포)
        kl_prior = self.kl_prior(X, pad_mask) # (bs,)

        # 3. Diffusion posterior KL: L_t = KL[q(x₀|x_t) || p_θ(x₀|x_t)] # L_t
        loss_all_t = self.compute_Lt(X, y, pred, noisy_data, pad_mask, test)

        # 4. Reconstruction loss: L₀ = -log p(x₀ | z₀)
        prob0 = self.reconstruction_logp(t, X, pad_mask) 

        # Assume X, y are one-hot or smoothed probability vectors
        loss_term_0 = self.val_X_logp(X * prob0.X.log()) # + self.val_y_logp(y * prob0.y.log())  # (bs,)

        # 5. Total loss per sample
        # nlls = -log_pN + kl_prior + loss_all_t - loss_term_0 # 왜 reconstruction loss를 안 쓰고, loss_term_0를 쓰지?
        nlls = kl_prior + loss_all_t - loss_term_0 # (bs,)
        assert nlls.ndim == 1, f'{nlls.shape} has more than batch dimension'

        # 6. Aggregate nlls (mean over batch)
        nll = (self.test_nll if test else self.val_nll)(nlls)

        # 8. Logging
        if wandb.run:
            wandb.log({
                "kl_prior": kl_prior.mean(),
                "Estimator loss_terms": loss_all_t.mean(),
                #"log_pn": log_pN.mean(),
                "loss_term_0": loss_term_0.mean(),
                "batch_test_nll" if test else "val_nll": nll
            }, commit=False)

        return nll

    def forward(self, noisy_data, extra_data, pad_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=-1).float() # (bs, L, vocab_size + extra_features)
        y = torch.hstack((noisy_data['y_t'], extra_data.y))
        return self.x0_model(X, y)

    # sampling ----------------------------
    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int, save_final: int, seq_lens=None, max_len=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        # 1. Generate sequence lengths
        if seq_lens is None:
            seq_lens = self.length_dist.sample_n(batch_size, self.device)  # To do: Define self.length_dist -> 이거 하면 log p(N)도 구할 수 있을듯!! 
        elif isinstance(seq_lens, int):
            seq_lens = torch.full((batch_size,), seq_lens, dtype=torch.int, device=self.device)
        else:
            assert isinstance(seq_lens, torch.Tensor)
        
        if max_len is None:
            max_len = torch.max(seq_lens).item()
        else:
            max_len = min(max_len, torch.max(seq_lens).item())

        # 2. Make pad mask
        arange = torch.arange(max_len, device=self.device).unsqueeze(0)
        pad_mask = arange < seq_lens.unsqueeze(1)

        # 3. Sample initial noise z_T ~ limit_dist
        X_t = F.one_hot(torch.full((batch_size, max_len), fill_value=self.transition_model.abs_state), num_classes=self.vocab_size).float() # abs_state 호환시키기
        y_t = torch.zeros(batch_size, 0) # To do: Modify

        # 4. Reverse sampling ### most important part
        chain_X = torch.zeros((number_chain_steps, keep_chain, max_len), dtype=torch.long, device=X_t.device)
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1), device=self.device)
            t_array = s_array + 1 
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X_t, y_t, pad_mask)
            X_t, y_t = sampled_s.X, sampled_s.y

            write_index = (s_int * number_chain_steps) // self.T # number_chain_steps: 몇 개 step을 저장할 것인지 # 여러 숫자가 한 숫자로 몰림 <- 그냥 x%y ==0 조건을 주는 게 깔끔하지 않나?
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain] # keep_chain개만 chain_X에 기록

        # Argmax sampling # To do: Modify to sample from the distribution (high temperature)
        sampled_s = sampled_s.pad(pad_mask=pad_mask, convert_to_idx=True) # (bs, L, vocab_size)
        X_0, y_0 = sampled_s.X, sampled_s.y

        # Save chains
        if keep_chain > 0:
            final_X_chain = X_0[:keep_chain]
            chain_X[0] = final_X_chain # 아니 0에다가 집어놓고 reverse tensor?

            chain_X = reverse_tensor(chain_X)

        sequence_list = []
        for i in range(batch_size):
            Length = seq_lens[i].item()
            token_ids = X_0[i, :Length].cpu()
            sequence_list.append(token_ids)

        return sequence_list
    
    def sample_p_zs_given_zt(self, s_norm, t_norm, X_t, y_t, pad_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        Batch, Length, Vocab = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t_norm) # (Batch, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_norm) # (Batch, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_norm)

        # Retrieve transition matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)
        Qt = self.transition_model.get_Qt_bar(beta_t, device=self.device)

        # Neural net prediction # p(x0 | xt)
        noisy_data = {'X_t': X_t, 'y_t': y_t, 'pad_mask': pad_mask, 't': t_norm}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, pad_mask)
        pred_X = F.softmax(pred.X, dim=-1) # (Batch, Length, Vocab) # Normalize

        # q(x_{t-1}|x_t, x_0)
        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution()

        # p(xs | xt) = sum q(x_{t-1}|x_t, x_0) * p(x0 | xt)
        weigthed_X = pred_X * p_s_and_t_given_0_X # (Batch, Length, Vocab)
        unnormalized_prob_X = weigthed_X.sum(dim=2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True) # normalize
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all() # Vocab의 총합이 1인가? # normalize가 잘 되어 있는가?

        # X_s ~ p(xs | xt)
        sampled_s = sample_discrete_features(probX=prob_X, pad_mask=pad_mask, device=self.device) # (bs, L) # sampled s for each token

        X_s = F.one_hot(sampled_s, num_classes=self.vocab_size).float()
        assert (X_t.shape == X_s.shape)

        # y = torch.zeros(y_t.shape[0], 0) # DiGress code에 없어서 주석처리. 실행해보고 필요없으면 삭제
        out_one_hot = PlaceHolder(X=X_s, y=y).type_as(X_s).pad(pad_mask=pad_mask) # 이렇게 one-hot으로 나눌 거면 PlaceHolder에서 convert_to_idx 왜 쓰는 거야?
        out_discrete = PlaceHolder(X=X_s, y=y).type_as(X_s).pad(pad_mask=pad_mask) # (bs, L, vocab_size) # sampled s for each token

        return out_one_hot, out_discrete
    
    def compute_extra_data(self, noisy_data): # 넣을지 말지 선택할 수 있게 해주기
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        
        extra_features = self.extra_featuers(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=-1) # diffusion step 추가

        PlaceHolder = PlaceHolder(X=extra_X, y=extra_y)


        

            














