import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import os
from omegaconf import OmegaConf

import pytorch_lightning as pl

from SmD3PM_utils import PlaceHolder, setup_wandb, move_to_device, save_generated_sequences # utils
from SmD3PM_utils import pad_distributions, sum_except_batch, sample_discrete_features, posterior_distribution, reverse_tensor, compute_batched_over0_posterior_distribution # diffusion_utils
from SmD3PM_utils import AbsorbingStateTransition, PredefinedNoiseScheduleDiscrete # noise_schedule
from SmD3PM_utils import SumExceptBatchMetric, SumExceptBatchKL, NLL # abstract_metrics
from SmD3PM_utils import TrainLossDiscrete # train_metrics
from SmD3PM_utils import NormalLengthDistribution # sample_utils

class X0Model(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim) # input(X_T)이 index가 아니라 이미 3차원임. 그래서 Embedding이 아니라 Linear

        encoder_layer = nn.TransformerEncoderLayer( # 이게 그냥 Transformer가 아니라 TransformerEncoder인 이유가 있을까?
            d_model=model_dim,
            nhead=num_heads, # num_head 여러 개 하면 디버거가 작동이 안 되려나? # To do: check
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, input_dim)  # or output_dim

    def forward(self, x_input, pad_mask):
        device = self.input_projection.weight.device
        x_input = x_input.to(device).float()
        pad_mask = move_to_device(pad_mask, x_input.device)

        x = self.input_projection(x_input)  # (batch_size, seq_len, model_dim)
        x = self.encoder(x, src_key_padding_mask=~pad_mask if pad_mask is not None else None)  # (batch_size, seq_len, model_dim) # src: source_input # transformer는 pad_mask가 true인 부분이 pad
        return self.output_head(x) 


class YModel(nn.Module): # for guidance # domain features # 궁극적으로는 logp를 예측해야 함(mask가 있는 상태에서)
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 1)) # 굉장히... 굉장히... 기본 구조네.... # 수정 필요 하려나? # To do: check

    def forward(self, x_input, pad_mask):
        device = self.model[0].weight.device
        x_input = x_input.to(device).float() # (B, L, D) # input sequence
        pad_mask = move_to_device(pad_mask, x_input.device)

        if pad_mask is not None:
            pad_mask = pad_mask.to(device)
            lengths = pad_mask.sum(dim=1)
            x_pooled = (x_input * pad_mask.unsqueeze(-1)).sum(dim=1)/lengths # log p 는 스칼라값이므로 압축 필요 # (B, L, D) → (B, D)
        else:
            x_pooled = x_input.mean(dim=1)
        return self.model(x_pooled) # (batch_size, 1) # 스칼라 값
        

class SmD3PM(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, extra_features, domain_features): # To do: Modify for my model
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps # diffusion step 수 (1000 (paper))

        # dataset_infos에서 정보 가져오기
        self.tokenizer = dataset_infos.tokenizer

        self.vocab_size = dataset_infos.vocab_size
        self.pad_idx = dataset_infos.pad_token_idx
        self.mask_idx = dataset_infos.mask_token_idx

        self.Xdim = dataset_infos.input_dims['X'] # input feature dimension
        self.ydim = dataset_infos.input_dims['y']

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train) # To do: make loss function
        # evaluate metrics -------------------------------
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL() # val와 계산 방법은 같지만, 기록하는 공책이 다르다!
        self.test_X_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        # --------------------------------------------------

        self.extra_features = extra_features
        self.domain_features = domain_features

        # Diffusion model structure
        self.x0_model = X0Model(input_dim=self.Xdim, 
                                model_dim=cfg.model.embedding_dim,
                                num_layers=cfg.model.num_layers,
                                num_heads=cfg.model.num_heads)

        self.y_model = YModel(input_dim=self.Xdim, # y 예측값은 x로부터 나오니까! # 추후 y를 추가할 수도 있겠지!
                              hidden_dim=cfg.model.y_hidden_dim) 

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=cfg.model.noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        if cfg.model.transition == "absorbing": # noise_schedule에서 absorbing을 해줬는데 transtion에 왜 또 absorbing을 넣어줘야 하지? 그냥 한 번에 하면 되는 걸 여러 번에 걸쳐서 하는 느낌이라...
            abs_state = cfg.model.abs_state # "?" token
            self.transition_model = AbsorbingStateTransition(abs_state=abs_state, num_classes=self.vocab_size) # for get Qt, Qt_bar

            # limit_dist: noise가 최대로 적용되었을 때 분포 abs_state만 1, 나머지 class는 0인 one-hot vector
            limit_dist = torch.zeros(self.vocab_size) # x_T가 도달해야하는 목적지
            limit_dist[abs_state] = 1.0
            self.limit_dist = PlaceHolder(X=limit_dist) # AttributeError 나는지 확인하고 주석 제거하기 

        self.length_dist = NormalLengthDistribution(device=self.device) # 정규 길이 분포 # default값은 zinc에 맞춰두었다.       

        flat_cfg = OmegaConf.to_container(cfg, resolve=True)  # flatten 된 dict
        self.save_hyperparameters(flat_cfg, ignore=["train_metrics", "sampling_metrics"])
        self.start_epoch_time = time.time()
        self.train_iterations = None
        #self.val_iterations = None
        self.log_every_steps = cfg.trainer.log_every_steps
        self.number_chain_steps = cfg.trainer.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def forward(self, noisy_data, extra_data, pad_mask=None, y_target=None):
        """
        forward(noisy_data, extra_data, pad_mask, y_target):
            noisy_data: dict with X_t, y_t, t
            extra_data: features from compute_extra_data
            pad_mask: padding mask
            y_target: optional, for computing guidance during training
        Return:
            PlaceHolder(X=x0_pred, y=y_pred)
        Args:
            noisy_data.X_t: (B, L, D) - noised sequence
            extra_data.X: (B, L, D_extra) - extra features (optional)
            extra_data.y: (B, D_y) - optional y-related features
            pad_mask: (B, L) - True if valid token
            y_target: (B, 1) - optional, for regression guidance
        """
        # 1. x0_model input 구성
        if extra_data is not None and extra_data.X is not None:
            x_input = torch.cat((noisy_data['X_t'], extra_data.X), dim=-1) # X_T
        else:
            x_input = noisy_data['X_t']

        pad_mask = pad_mask.to(x_input.device)
        x0_pred = self.x0_model(x_input, pad_mask=pad_mask) # shape?

        # 2. y_model input 구성
        if y_target is not None:  
            y_pred = self.y_model(x_input, pad_mask=pad_mask)
            return PlaceHolder(X=x0_pred, y=y_pred)
        else:
            return PlaceHolder(X=x0_pred, y=None)

    # train.py ---------------------------
    
    def configure_optimizers(self): 
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=self.cfg.train.weight_decay)
    
    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader()) # 훈련 데이터 배치 수 계산해서 저장
        self.print("Size of the input features", self.Xdim, self.ydim)
        if self.local_rank == 0: # 메인 프로세스만 실행 # 외부 로깅 툴은 여러 프로세스에서 동시에 호출하면 오류가 생기므로, 한 프로세스만 실행
            setup_wandb(self.cfg) # lightning으로 자동 초기화할 수 있으나 수동 구현으로 유연성 보장

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...") # self.print : main process에서만 실행 -> log가 꼬이거나 중복되지 않도록
        self.start_epoch_time = time.time() 
        self.train_loss.reset() # train_loss 초기화
        self.train_metrics.reset() # train_metrics 초기화 # 정확도, F1-score, recall, precision 등 평가 지표(metric)을 관리하는 클래스

    def training_step(self, batch, batch_idx): # To do: i 가 batch index인지 확인하고 맞다면 batch_idx로 바꾸기
        # data.x: (batch_size, seq_len) -> tokenized input sequences
        # data.pad_mask: (batch_size, seq_len) -> True if real token, False if pad
        batch = move_to_device(batch, self.device)
        input_seq = batch.x # To do: check where is data
        pad_mask = batch.pad_mask
        true_y = batch.y

        noisy_seq = self.apply_noise(input_seq, pad_mask, true_y)
        extra_features = self.compute_extra_data(noisy_seq)

        noisy_seq = move_to_device(noisy_seq, self.device)
        extra_features = move_to_device(extra_features, self.device)

        # Forward pass   # 근데 여기서 말하는 forward가 x0가 맞나? 일단 내 forward pass는 x0를 예측하는 건데
        pred = self.forward(noisy_seq, extra_features, pad_mask)

        # Compute loss # To do:  이 놈의 train_loss는 도대체 어디서 정의하는 건데!!!! # train_metrics는 또 뭔데요? 
        loss = self.train_loss(
            masked_pred_X=pred.X,  # pred.seq: (batch_size, seq_len, vocab_size)
            pred_y=pred.y,
            true_X=input_seq,
            true_y=true_y,
            pad_mask=pad_mask,
            log=(batch_idx % self.log_every_steps == 0)
        )

        # log training metrics # To do: Check # Optioinal
        self.train_metrics(
            masked_pred_X=pred.X,
            true_X=input_seq,
            pad_mask=pad_mask,
            log=(batch_idx % self.log_every_steps == 0)
        )

        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        # 1. loss (x_CE, y_CE) 로깅
        loss_log = self.train_loss.log_epoch_metrics()  # {'train_epoch/x_CE': ..., 'train_epoch/y_CE': ...}

        # 2. metrics (loss, acc 등)
        seq_metrics = self.train_metrics.log_epoch_metrics()  # {'train/loss': ..., 'train/acc': ..., 'train/tokens': ...}

        # 3. 출력 구성
        self.print(
            f"Epoch {self.current_epoch} Summary:"
            f"\n  Train Loss:"
            f"\n  x_CE: {loss_log['train_epoch/x_CE']:.3f}"
            f" | y_CE: {loss_log['train_epoch/y_CE']:.3f}"
            f"\n  Train Metrics:"
            f"\n  SeqLoss: {seq_metrics['train/loss']:.3f}"
            f" | Acc: {seq_metrics['train/acc']:.3f}"
            f" | Tokens: {seq_metrics['train/tokens']}"
            f"\n  Time Elapsed: {time.time() - self.start_epoch_time:.1f}s"
        )

        if torch.cuda.is_available():
            print(torch.cuda.memory_summary()) # 메모리 사용 현황 출력
        else:
            print("CUDA is not available. Skipping memory summary.")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_X_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        input_seq = batch.x
        pad_mask = batch.pad_mask
        true_y = batch.y

        pad_mask = pad_mask.to(input_seq.device) # 같은 batch에서 나왔는데 굳이?

        noisy_seq = self.apply_noise(input_seq, pad_mask, true_y)
        extra_features = self.compute_extra_data(noisy_seq)
        noisy_seq = move_to_device(noisy_seq, self.device)
        extra_features = move_to_device(extra_features, self.device)

        pred = self.forward(noisy_seq, extra_features, pad_mask)
        nll = self.compute_val_loss(pred, noisy_seq, input_seq, true_y,  pad_mask, test=False) # To do: Check # 아니 update를 시켜줘야 하는 거 아냐? 이건 validation이구나... 그럼 지금까지 한 게 train loss가 아니었구나...
        
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_X_logp.compute() * self.T] # To do: Check # log p는 뭐야? # diffusion 수만큼 누적되는 것 반영
        if wandb.run:
            wandb.log({
                "val/epoch_nll": metrics[0],
                "val/X_kl": metrics[1],
                "val/X_logp": metrics[2]
            }, commit=False) # 로그 그룹화 가능하게 commit=False로 설정
        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0]: .2f} -- Val KL {metrics[1]: .2f} -- Val log p {metrics[2]: .2f} -- {time.time() - self.start_epoch_time:.1f}s ")

        # Lightning 내부 로거에 NLL 등록 (Checkpoint callback이 모니터할 수 있게)
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))
        
        self.val_counter += 1
        if self.val_counter % self.cfg.trainer.early_stopping == 0: # 일정 주기마다 샘플 생성
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            id = 0
            while samples_left_to_generate > 0:
                batch_size = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, batch_size)
                to_save = min(samples_left_to_save, batch_size) # To do : Check # Batch size랑 비교를 하는 게 맞나? # 나머지랑 같은 거면 그냥 하나만 써도 되잖아.
                chains_save = min(chains_left_to_save, batch_size) # To do : Check # Batch size랑 비교를 하는 게 맞나?
                samples.extend(self.sample_batch(batch_id=id, batch_size=to_generate, 
                                                 keep_chain=chains_save, number_chain_steps=self.number_chain_steps, 
                                                 save_final=to_save,
                                                 seq_lens=None, max_len=None))
                id += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            
            self.print("Computing sampling metrics...")
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank) # To do: Check # Forward가 왜 여기서 나와? 내가 정의한 forward가 이런 게 아니었던 거 같은데? 그냥 x_0 예측하는 거였던 거 같은데...
            
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_X_logp.reset()
        self.sampling_metrics.reset()
        if self.local_rank == 0: # test_only가 아닐 때도 또 setup이 되면 곤란하지 않을까... # To do: check
            setup_wandb(self.cfg)

    def test_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        input_seq = batch.x 
        pad_mask = batch.pad_mask
        true_y = batch.y

        noisy_seq = self.apply_noise(input_seq, pad_mask, batch.y) # 여기는 그냥 train 성능 보고 sampling은 뒤에서 따로 함
        extra_features = self.compute_extra_data(noisy_seq)
        noisy_seq = move_to_device(noisy_seq, self.device)
        extra_features = move_to_device(extra_features, self.device)

        pred = self.forward(noisy_seq, extra_features, pad_mask)
        nll = self.compute_val_loss(pred, noisy_seq, input_seq, batch.y,  pad_mask, test=True) 
        return {'loss': nll}
    
    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_X_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_nll": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/X_logp": metrics[2]}, commit=False)
        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0]: .2f} -- Test KL {metrics[1]: .2f} -- Test log p {metrics[2]: .2f} -- {time.time() - self.start_epoch_time:.1f}s ")
        
        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll:.4f}\n') # 그냥 출력용 (not logging)

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='\n', flush=True)
            batch_size = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, batch_size) # 각각 하고 싶은 수량이 다를 수 있으니까~~
            to_save = min(samples_left_to_save, batch_size)
            chains_save = min(chains_left_to_save, batch_size)
            samples.extend(self.sample_batch(batch_id=id, batch_size=to_generate, 
                                                 keep_chain=chains_save, number_chain_steps=self.number_chain_steps, 
                                                 save_final=to_save,
                                                 seq_lens=None, max_len=None))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        # Saving
        self.print("Saving the generated sequences...\n")
        save_generated_sequences(samples, self.cfg, self.tokenizer)

        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        self.print("Done testing.")

    # data processing ----------------------
    def apply_noise(self, X, pad_mask, y):
        """ Sample noise and apply it to the data. """
        pad_mask = pad_mask.to(X.device)

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately \therefore lowest_t = 1
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(low=lowest_t, high=self.T, size=(X.size(0), 1), device=X.device).float() # (bs, 1) # t ~ Uniform(0, T) for each sample in batch
        s_int = t_int - 1 # (bs, 1) # s = t - 1

        t_normalized = t_int / self.T # (bs, 1) # t / T
        s_normalized = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule.get_beta(t_normalized = t_normalized) # (bs, 1) # beta_t
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized = s_normalized) # (bs, 1) # alpha_s_bar
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized = t_normalized) # (bs, 1) # alpha_t_bar

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)
        assert (abs(Qtb.sum(dim=2) - 1.) < 1e-4).all(), Qtb.sum(dim=2) - 1 # row의 총 합이 1인가? (== normalize가 잘 되어 있는가?)

        # Compute tranisition probabilities
        X = F.one_hot(X, num_classes=self.vocab_size).float() # 2dim -> 3dim 
        assert X.ndim == 3, f"X should be one-hot with shape (B, L, V), got {X.shape}"
        
        probX = torch.einsum("blc,bcd->bld", X, Qtb) # (bs, L, vocab) # X_noise

        # Sample noisy sequence
        sampled_t = sample_discrete_features(probX=probX, pad_mask=pad_mask) # (bs, L) # sampled t for each token

        # One-hot encoding
        X_t = F.one_hot(sampled_t.X, num_classes=self.vocab_size) # num classes
        assert (X.shape == X_t.shape)

        z_t = PlaceHolder(X=X_t).type_as(X_t).pad(pad_mask=pad_mask) # pad mask로 수정

        noisy_data = {'t_int': t_int, 't': t_normalized, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'y_t': z_t.y, 'pad_mask': pad_mask}

        return noisy_data

    # validation loss ----------------------
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
        Ts = (self.T - 1) * ones # (Batch, 1) # T: diffusion step 수 
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts) # (Batch, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar) # (Batch, Vocab, Vocab)

        assert X.ndim == 3, f"X should be one-hot with shape (B, L, V), got {X.shape}"
        probX = torch.einsum("blc,bcd->bld", X, Qtb) # batch matmul # (Batch, Length, Vocab) # X_noise

        limit_X = self.limit_dist.X[None, None, :].expand_as(probX).type_as(probX) # broadcasting # (1, 1, Vocab) -> (Batch, Length, Vocab) # 가정한 X_T의 목표 분포

        limit_X, probX = pad_distributions( # padding된 부분은 KL 계산에서 제외
            true_X=limit_X.clone(), # (Batch, Length, Vocab) # limit_X는 정적이고 재사용되므로 원본이 수정되지 않도록 clone() 
            pred_X=probX, # (Batch, Length, Vocab)
            pad_mask=pad_mask, # (Batch, Length)
            pad_token_idx=self.pad_idx 
        )

        kl_distance_X = F.kl_div( # 각 시점, 각 토큰 위치에서의 KL divergence (token 단위)
            input=probX.log(), target=limit_X, reduction='none'
        ) # input은 log(Q), target은 P 형태로 넣어야 한다.

        return sum_except_batch(kl_distance_X) # Batch size 제외 전체 합산

    def compute_Lt(self, X, y, pred, noisy_data, pad_mask, test): # to do: feature y 추가 # To do: 질문 해결하기(x_0)
        pred_probs_X = F.softmax(pred.X, dim=-1).to(self.device) # pred.x_0의 logit
        has_y = pred.y is not None
        pred_probs_y = F.softmax(pred.y, dim=-1).to(self.device) if has_y else None

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar']).to(self.device) # (Batch, Vocab, Vocab)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar']).to(self.device) # (Batch, Vocab, Vocab)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t']).to(self.device) # (Batch, Vocab, Vocab)

        # Compute distributions to compare with KL
        batch, length, vocab = X.shape

        # 둘 다 true 같은데???? 뭐지???? x_t로 예측한 x_0가 전혀 들어가지 않았는데??
        # 그냥 x_0의 값이랑 X_0의 softmax를 넣은 차이 뿐인데?
        # To do: x0 model 추가 필요할듯 -> pred로 들어오는 거 같은데... 그럼 pred는 어디서 들어오는 거야?
        # 생각해보니 x0 끼리 비교하는 건데 posterior는 왜 구하는 거지?
        prob_true = posterior_distribution(X=X, y=y if has_y else None, X_t=noisy_data['X_t'], y_t=noisy_data['y_t'] if has_y else None, Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred = posterior_distribution(X=pred_probs_X, y=pred_probs_y if has_y else None, X_t=noisy_data['X_t'], y_t=noisy_data['y_t'] if has_y else None, Qt=Qt, Qsb=Qsb, Qtb=Qtb)

        # Reshape and filter masked rows
        prob_true.X, prob_pred.X = pad_distributions(true_X=prob_true.X, pred_X=prob_pred.X, pad_mask=pad_mask, pad_token_idx=self.pad_idx) # (Batch, Length, Vocab) # padding된 부분은 KL 계산에서 제외
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X)) # To do: 아니 근데 test랑 val이랑 다른가? 확인하기

        return self.T * kl_x # 전체 ELBO/NLL 추정을 위해 scale 복원

    def reconstruction_logp(self, t, X, pad_mask): 
        # compute noise values for t = 0
        # noise가 거의 없는 Q0를 곱한 x0를 기반으로 얼마나 x0를 잘 복원할 수 있는지 평가
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0).to(X.device) # (Batch, Vocab, Vocab)

        assert X.ndim == 3, f"X should be one-hot with shape (B, L, V), got {X.shape}"

        probX0 = torch.einsum("blc,bcd->bld", X, Q0)

        sampled0 = sample_discrete_features(probX=probX0, pad_mask=pad_mask) # X_1 (bs, L) # sampled t for each token

        X0 = F.one_hot(sampled0.X, num_classes=self.vocab_size).float() # X_1
        y0 = sampled0.y # zero tensor
        assert (X.shape == X0.shape)

        sampled_0 = PlaceHolder(X=X0, y=y0).type_as(X0).pad(pad_mask=pad_mask) 

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'y_t': sampled_0.y, 'pad_mask': pad_mask, 't': torch.zeros(X0.shape[0], 1).type_as(sampled0.y)} 
        extra_data = self.compute_extra_data(noisy_data)

        pred0 = self.forward(noisy_data, extra_data, pad_mask) # x1을 보고 x0 예측

        # Normalize prediction
        probX0 = F.softmax(pred0.X, dim=-1)
        if pred0.y is not None: # y가 있다면
            proby0 = F.softmax(pred0.y, dim=-1)
        else:
            proby0 = None

        # Set masked rows to arbitrary values that don't contribute to the loss
        probX0[~pad_mask] = torch.ones(self.Xdim).type_as(probX0)

        return PlaceHolder(X=probX0, y=proby0) # -log p_θ(x₀) 계산을 위한 확률 분포 출력

    def compute_val_loss(self, pred, noisy_data, X, y, pad_mask, test=False): # To do: check # 그럼 training loss는 어디에 있는 거야?
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
        X = F.one_hot(X, num_classes=self.vocab_size).float().to(self.device)
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

        # Assume X, y are one-hot or smoothed probability vectors # 위의 식이랑 아래 식 합쳐서 loss_term_0 한 번에 구할 것 # To do
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

    # sampling model ----------------------------
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
        X_t = F.one_hot(torch.full((batch_size, max_len), fill_value=self.transition_model.abs_state, device=self.device), num_classes=self.vocab_size).float() # abs_state 호환시키기
        y_t = torch.zeros(batch_size, 0, device=self.device) # To do: Modify # condition을 줄 때 y_t는 어떤 값으로 초기화할지 생각하기

        # 4. Reverse sampling ### most important part
        chain_X = torch.zeros((number_chain_steps, keep_chain, max_len), dtype=torch.long, device=X_t.device)
        for t_int in reversed(range(0, self.T)):
            t_array = t_int * torch.ones((batch_size, 1), device=self.device)
            t_norm = t_array / self.T
            
            s_int = t_int - 1
            s_array = s_int * torch.ones((batch_size, 1), device=self.device)
            s_norm = s_array / self.T 
            
            sampled_oh, sampled_idx = self.sample_p_zs_given_zt(s_norm, t_norm, X_t, y_t, pad_mask)
            X_t, y_t = sampled_oh.X, sampled_oh.y

            # 기록 조건 → evenly spaced steps
            if (t_int * number_chain_steps) % self.T == 0:
                write_index = (t_int * number_chain_steps) // self.T # number_chain_steps: 몇 개 step을 저장할 것인지 
                chain_X[write_index] = sampled_idx.X[:keep_chain]  # 정수 인덱스로된 X 저장

        # Argmax sampling # To do: Modify to sample from the distribution (high temperature)
        final_idx = sampled_idx.X  # (B, L)

        # Save chains
        if keep_chain > 0:
            chain_X[0] = final_idx[:keep_chain]
            chain_X = reverse_tensor(chain_X)

        sequence_list = []
        for i in range(batch_size):
            Length = seq_lens[i].item()
            token_ids = final_idx[i, :Length].cpu()
            sequence_list.append(token_ids)

        return sequence_list # list of tensors
    
    def sample_p_zs_given_zt(self, s_norm, t_norm, X_t, y_t, pad_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        Batch, Length, Vocab = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t_norm) # (Batch, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_norm) # (Batch, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_norm)

        # Retrieve transition matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar)
        Qt = self.transition_model.get_Qt(beta_t)

        # Neural net prediction # p(x0 | xt)
        noisy_data = {'X_t': X_t, 'y_t': y_t, 'pad_mask': pad_mask, 't': t_norm}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, pad_mask)
        pred_X = F.softmax(pred.X, dim=-1) # (Batch, Length, Vocab) # Normalize

        # q(x_{t-1}|x_t, x_0)
        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb)

        # p(xs | xt) = q(x_{t-1}|x_t, x_0) * p(x0 | xt)
        weighted_X = torch.einsum('bnd,bndk->bnk', pred_X, p_s_and_t_given_0_X) # (Batch, Length, Vocab)
        prob_X = F.softmax(weighted_X, dim=-1) # for normalization
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all() # Vocab의 총합이 1인가? # normalize가 잘 되어 있는가?

        # X_s ~ p(xs | xt)
        sampled_idx = sample_discrete_features(probX=prob_X, pad_mask=pad_mask) # (bs, L) # sampled s for each token
        sampled_X_oh = F.one_hot(sampled_idx.X, num_classes=self.vocab_size).float() # 그저 sample한 걸 one-hot으로 변환했을 뿐 (-> 다양성 걱정할 필요 없을듯듯)
        assert (X_t.shape == sampled_X_oh.shape)

        y = y_t.clone() if y_t is not None else None # To do: y sampling 함수 만들기
         
        out_one_hot = PlaceHolder(X=sampled_X_oh, y=y).type_as(X_t).pad(pad_mask=pad_mask)
        return out_one_hot, sampled_idx
    
    def compute_extra_data(self, noisy_data):
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)

        if extra_features.y is not None and extra_molecular_features.y is not None:
            extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)
        else:
            extra_y = None

        # Always add timestep embedding to y if y is being used
        if not (extra_y is None or (isinstance(extra_y, torch.Tensor) and extra_y.numel() == 0)):
            extra_y = torch.cat((extra_y, noisy_data['t']), dim=-1) # diffusion step 정보 추가

        return PlaceHolder(X=extra_X, y=extra_y)


            














