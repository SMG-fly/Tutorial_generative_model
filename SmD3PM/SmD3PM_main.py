import argparse
import os
import hydra
from hydra import initialize, compose

import torch
from torch.serialization import safe_globals
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint # To do: check # ModelCheckpoint는 모델 저장을 위한 콜백, EMA는 Exponential Moving Average를 적용하기 위한 콜백

from SmD3PM_model import SmD3PM
from SmD3PM_utils import update_config_with_new_keys, create_folders # utils

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train DiGress on SMILES dataset(Sequence).")
    parser.add_argument('--config', type=str, default='config/SmD3PM_config.yaml')
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    return parser.parse_args()

def load_config_from_args() -> DictConfig:
    args = parse_arguments()
    config_path, config_file = os.path.split(args.config)
    config_name = os.path.splitext(config_file)[0]

    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name=config_name, overrides=args.overrides)
    return cfg

def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    # test할 때 쓰기 위한 cfg 불러와서 저장하기
    saved_cfg = cfg.copy()  # cfg는 DictConfig라서 copy로 복사 가능
    name = cfg.general.name + '_resume' # 현재 실험 이름에 _resume suffix를 붙여서 새로운 이름 생성 (logging, checkpoint 경로 관리에 유리)
    resume = cfg.general.test_only # test_only는 테스트할 ckpt 경로를 담고 있다. 

    # 모델 cfg 불러온 후 위에서 저장한 값으로 보완하기
    model = SmD3PM.load_from_checkpoint(resume, **model_kwargs) # PL(LightningModule) 기반 모델에서 제공하는 load_from_checkpoint
    cfg = model.cfg # 훈련 당시 설정을 복원 # cfg 자체가 완전히 새롭게 바뀜
    cfg.general.test_only = resume  # 훈련 당시 저장된 설정 값
    cfg.general.name = name # resume 버전임을 명시
    cfg = update_config_with_new_keys(cfg, saved_cfg) # 현재 설정에서 빠진 key만 보완
    return cfg, model

def get_resume_adaptive(cfg, model_kwargs): # cfg: 현재 실행된 Hydra 설정 # model_kwargs: 모델 초기화 인자 (dataset info 등) # ckpt의 cfg를 기준으로 하되, 실행 시점의 config 일부를 덮어씌움
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    # 현재 실행된 config를 백업
    saved_cfg = cfg.copy()

    # Fetch path to this file to get base path 
    current_path = os.path.dirname(os.path.realpath(__file__)) # 현재 py 파일의 경로를 가져옴 
    resume_path = os.path.join(current_path, cfg.general.resume) # resume_path는 절대 경로로 바뀌게 된다. # cfg.general.resume는 상대 경로로 저장되어 있음

    model = SmD3PM.load_from_checkpoint(resume_path, **model_kwargs) # 체크포인트로부터 모델 파라미터와 함께 config도 불러옴
    new_cfg = model.cfg # 훈련 당시 저장된 설정값

    # ckpt의 config에 현재 설정값 덮어쓰기 # 충돌하는 cfg 키는 현재 config로 덮어쓰게 된다. <- get_resume(ckpt의 config가 우선)과의 차이점
    for category in cfg: 
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg] 

    new_cfg.general.resume = resume_path # 현재 어떤 ckpt로부터 이어받았는지 명시
    new_cfg.general.name = new_cfg.general.name + '_resume' 

    new_cfg = update_config_with_new_keys(new_cfg, saved_cfg) # 현재 설정에서 빠진 key만 보완
    return new_cfg, model 

# main -----------------
from SmD3PM_dataset import SequenceDataModule, SequenceDatasetInfos   # 수정하기
from SmD3PM_utils import TrainSequenceMetrics, SamplingSequenceMetrics # spectre_utils
from SmD3PM_utils import ExtraFeatures, DummyExtraFeatures # extra_features

def main(cfg: DictConfig): 

    ##### For Debugging #####
    if cfg.general.debug:  # 또는 args.debug
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
    #########################
        
    datamodule = SequenceDataModule(cfg) # load dataset # train/val/test DataLoader를 생성하는 PyTorch Lightning 모듈
    dataset_infos = SequenceDatasetInfos(datamodule, cfg) # datset 정보 구성 # 토크나이저, vocab size, padding 정보 등 → 모델 입력/출력 차원 계산, 마스킹, 조건 생성 등에 사용
 
    # Metric 설정 # 학습 중 loss 추적, 샘플 생성 후 평가 등에 필요한 metric 클래스 # 여기도 함수 만들어 줘야 함!
    train_metrics = TrainSequenceMetrics(dataset_infos)
    sampling_metrics = SamplingSequenceMetrics(dataset_infos)


    # extra feature 쓸건지 안 쓸건지 결정 # 원래 코드에는 discrete인지도 보는데 우리는 discrete인 경우만 보고 있으니 생략
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos) # To do: Check # dataset_info는 뭐야? # Extra features는 conditional generate랑 관련있는 거야?
        # domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos) # 샘플링 이후 평가용 # condition으로 넣을 거면 extra_features 사용 # 근데 domain_features도 넣어줬던 거 같은데 뭐지...? # To do: check
        domain_features = DummyExtraFeatures()
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures() # Domain features 빼도 되는지 확인

    # Input/Output dimension 계산 # input_dims, output_dims update # 혹시 extra_features가 추가했을지 모르니까!
    dataset_infos.compute_input_output_dims(extra_features=extra_features, domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics, 'sampling_metrics': sampling_metrics, 
                    'extra_features': extra_features, 'domain_features': domain_features}

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded 
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    create_folders(cfg) # To do: make it # utils.py 에 있음

    # model 불러오기~~~ ### # 그러고보니 앞에 test 부분은 sampling으로 넘기는 게 좋을듯? 어려우면 말고... 일단 후순위임
    model = SmD3PM(cfg=cfg, **model_kwargs)
    # model_kwargs: 위에서 만든 모든 구성 요소 (dataset_infos, train_metrics, extra_features, …)
    # cfg에는 학습 세팅이 들어감 (lr, dropout, optimizer 등)

    # To do: check # 왜 model save부터 해?
    callbacks = [] # 이건 왜 나온 거야? # To do: check
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"{cfg.save.checkpoint_path}/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL', # logger에 기록된 값 중 val/epoch_NLL을 모니터링
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1) # Every_n_epochs: 몇 epoch마다 저장할지
        last_ckpt_save = ModelCheckpoint(dirpath=f"{cfg.save.checkpoint_path}/{cfg.general.name}",
                                         filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback) # To do: check # callback list에 추가하면 무슨 일이 생기지?

    if cfg.general.debug:
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    # 드디어 trainer ###
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    # pytorch ligthning 에서 Trainier를 불러오게 됨 # To do: 보통 trainer 어떻게 쓰는지 알아둘 필요가 있을듯
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                    strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints # 분산 학습 방식...?? 이게 뭔데 # DDP+unused_parameters=True가 무슨 의미야? DDP도 모르고 unused_parameter도 몰라
                    accelerator='gpu' if use_gpu else 'cpu',
                    devices=cfg.general.gpus if use_gpu else 1,
                    max_epochs=cfg.train.n_epochs,
                    check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                    fast_dev_run=cfg.general.debug,
                    enable_progress_bar=False, # True로 해봤지만... 여전히 아무것도 돌아가지 않는다...
                    callbacks=callbacks, # callback들이 아래 작업을 자동으로 수행: # 1) 모델 저장, 2) 학습 중간에 EMA 적용, etc.
                    log_every_n_steps=cfg.trainer.log_every_steps,
                    logger = None) # To do: Trainer 어떻게 쓰는 건지 알아보기 # wandblogger를 연동하는 것 고려해보기

    # test_only == False이면 학습 후 best 모델로 test
    if not cfg.general.test_only: 
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume) # Trainer.fit()은 모델을 학습시키는 함수 # datamodule: train/val/test DataLoader를 생성하는 PyTorch Lightning 모듈
        
        if not cfg.general.debug: # To do: 이게 무슨 조건이고 아래 코드는 무슨 코드지?
            best_ckpt_path = checkpoint_callback.best_model_path
            if not best_ckpt_path or not os.path.exists(best_ckpt_path): # best checkpoint가 없다면 last.ckpt로 fallback
                print("[INFO] No best checkpoint found. Using last.ckpt as fallback.")
                best_ckpt_path = f"{cfg.save.checkpoint_path}/{cfg.general.name}/last.ckpt"
            
            trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)

    else: # test_only인 경우 # 학습(trainer.fit) 없이 바로 test를 진행
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

if __name__ == "__main__":
    cfg = load_config_from_args()
    main(cfg)