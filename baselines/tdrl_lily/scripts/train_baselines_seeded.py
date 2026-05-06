"""Wrapper around train_baselines.py that injects a per-seed log_dir suffix.

Usage:
  python train_baselines_seeded.py --exp synthetic_ivae --seed 42 \
      --log_root experiments/synth_crl_baselines
"""
import warnings
warnings.filterwarnings('ignore')

import argparse, os, sys, yaml, pwd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from LiLY.tools.utils import load_yaml
from LiLY.datasets.sim_dataset import (
    SimulationDatasetTSTwoSample,
    SimulationDatasetTSTwoSampleNS,
)
from LiLY.baselines.iVAE.model import iVAE
from LiLY.baselines.BetaVAE.model import BetaVAE
from LiLY.baselines.SlowVAE.model import SlowVAE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    help='Config stem: synthetic_ivae | synthetic_svae | ...')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--log_root', required=True,
                    help='Override LOG from yaml: final dir = {log_root}/{exp}/seed_{seed}')
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_cfg = os.path.join(script_dir, '../LiLY/configs', f'{args.exp}.yaml')
    cfg = load_yaml(abs_cfg)
    print('######### Configuration #########')
    print(yaml.dump(cfg, default_flow_style=False))
    print('#################################')

    pl.seed_everything(args.seed)

    if cfg['NS']:
        data = SimulationDatasetTSTwoSampleNS(directory=cfg['ROOT'], transition=cfg['DATASET'])
    else:
        data = SimulationDatasetTSTwoSample(directory=cfg['ROOT'], transition=cfg['DATASET'])

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data) - num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, batch_size=cfg['VAE']['TRAIN_BS'],
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg['VAE']['VAL_BS'],
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'], shuffle=False)

    model_type = cfg['MODEL']
    if model_type == 'iVAE':
        model = iVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                     z_dim=cfg['VAE']['LATENT_DIM'],
                     hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                     lr=cfg['iVAE']['LR'],
                     correlation=cfg['MCC']['CORR'])
    elif model_type == 'BetaVAE':
        model = BetaVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                        z_dim=cfg['VAE']['LATENT_DIM'],
                        hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                        beta=cfg['BetaVAE']['BETA'],
                        beta1=cfg['SlowVAE']['beta1_VAE'],
                        beta2=cfg['SlowVAE']['beta2_VAE'],
                        lr=cfg['BetaVAE']['LR'],
                        correlation=cfg['MCC']['CORR'])
    elif model_type == 'SlowVAE':
        model = SlowVAE(input_dim=cfg['VAE']['INPUT_DIM'],
                        z_dim=cfg['VAE']['LATENT_DIM'],
                        hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                        beta=cfg['SlowVAE']['BETA'],
                        gamma=cfg['SlowVAE']['GAMMA'],
                        beta1=cfg['SlowVAE']['beta1_VAE'],
                        beta2=cfg['SlowVAE']['beta2_VAE'],
                        lr=cfg['SlowVAE']['LR'],
                        rate_prior=cfg['SlowVAE']['RATE_PRIOR'],
                        correlation=cfg['MCC']['CORR'])
    else:
        raise ValueError(f'Unsupported MODEL: {model_type}')

    log_dir = os.path.join(args.log_root, args.exp, f'seed_{args.seed}')
    os.makedirs(log_dir, exist_ok=True)
    print(f'[log_dir] {log_dir}')

    checkpoint_callback = ModelCheckpoint(monitor='val_vae_loss', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_vae_loss', min_delta=0.0,
                                        patience=50, verbose=False, mode='min')

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['VAE']['GPU'],
                         val_check_interval=cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         deterministic=True,
                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
