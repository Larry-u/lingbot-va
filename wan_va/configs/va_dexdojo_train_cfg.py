# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_dexdojo_cfg import va_dexdojo_cfg
import os

va_dexdojo_train_cfg = EasyDict(__name__='Config: VA dexdojo train')
va_dexdojo_train_cfg.update(va_dexdojo_cfg)

va_dexdojo_train_cfg.dataset_path = '/root/nas/yuxianggang/data/dexdojo_lerobot_v21'
va_dexdojo_train_cfg.empty_emb_path = os.path.join(va_dexdojo_train_cfg.dataset_path, 'empty_emb.pt')
va_dexdojo_train_cfg.enable_wandb = False
va_dexdojo_train_cfg.load_worker = 8
va_dexdojo_train_cfg.save_interval = 2000
va_dexdojo_train_cfg.gc_interval = 50
va_dexdojo_train_cfg.cfg_prob = 0.1

# Training parameters (1x B200-180G: batch 4 x grad-accum 8 -> GBS 32)
va_dexdojo_train_cfg.learning_rate = 1e-5
va_dexdojo_train_cfg.beta1 = 0.9
va_dexdojo_train_cfg.beta2 = 0.95
va_dexdojo_train_cfg.weight_decay = 0.1
va_dexdojo_train_cfg.warmup_steps = 100
va_dexdojo_train_cfg.batch_size = 8
va_dexdojo_train_cfg.gradient_accumulation_steps = 4
va_dexdojo_train_cfg.num_steps = 10000
