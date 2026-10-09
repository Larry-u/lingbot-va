# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

# DexDojo (Xspark DexBench: tianji dual-arm + WujiHand2, 6 tasks).
# 54-D absolute joint-space actions in corpus block order
# [left_arm(7), left_hand(20), right_arm(7), right_hand(20)] @ 25 Hz.
# Videos sampled every 4th frame (6.25 fps) so each sampled video frame owns
# exactly 4 native action steps, matching the 4-actions-per-video-frame
# contract of the dataset / server code.
va_dexdojo_cfg = EasyDict(__name__='Config: VA dexdojo')
va_dexdojo_cfg.update(va_shared_cfg)

va_dexdojo_cfg.wan22_pretrained_model_name_or_path = "/root/nas/yuxianggang/weights/lingbot-va-base-dexdojo54"

va_dexdojo_cfg.attn_window = 72
va_dexdojo_cfg.frame_chunk_size = 2
va_dexdojo_cfg.env_type = 'none'

va_dexdojo_cfg.height = 256
va_dexdojo_cfg.width = 320
va_dexdojo_cfg.action_dim = 54
va_dexdojo_cfg.action_per_frame = 16
va_dexdojo_cfg.obs_cam_keys = [
    'observation.images.stereo', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_dexdojo_cfg.guidance_scale = 5
va_dexdojo_cfg.action_guidance_scale = 1

va_dexdojo_cfg.num_inference_steps = 25
va_dexdojo_cfg.video_exec_step = -1
va_dexdojo_cfg.action_num_inference_steps = 50

va_dexdojo_cfg.snr_shift = 5.0
va_dexdojo_cfg.action_snr_shift = 1.0

va_dexdojo_cfg.used_action_channel_ids = list(range(54))
inverse_used_action_channel_ids = [
    len(va_dexdojo_cfg.used_action_channel_ids)
] * va_dexdojo_cfg.action_dim
for i, j in enumerate(va_dexdojo_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_dexdojo_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_dexdojo_cfg.action_norm_method = 'quantiles'
# q01/q99 over the full 6-bucket 54-D action pool, produced by
# tools_dexdojo/calc_norm_stats.py. Sourced from a JSON file so the same
# config works on the training box and the eval cluster without edits:
# set LINGBOT_DEXDOJO_STATS=/path/to/norm_stats.json to override the default.
# If the file is absent at import time a placeholder is installed and any
# actual dexdojo use fails fast (see _norm_stat_placeholder consumers).
import json as _json
import os as _os

_norm_stat_path = _os.environ.get(
    'LINGBOT_DEXDOJO_STATS',
    '/root/nas/yuxianggang/data/dexdojo_lerobot_v21/norm_stats.json')
try:
    with open(_norm_stat_path) as _f:
        _stats = _json.load(_f)
    assert len(_stats['q01']) == 54 and len(_stats['q99']) == 54, _norm_stat_path
    va_dexdojo_cfg.norm_stat = _stats
except FileNotFoundError:
    va_dexdojo_cfg.norm_stat = {"q01": [0.0] * 54, "q99": [0.0] * 54}
    va_dexdojo_cfg._norm_stat_placeholder = True
