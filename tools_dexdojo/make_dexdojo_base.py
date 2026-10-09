# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Create a DexDojo (54-D action) base checkpoint from lingbot-va-base (30-D).

Copies the base dir, rewrites transformer/config.json action_dim 30 -> 54 and
re-initializes the two action-head projections whose shapes depend on it
(`action_embedder`, `action_proj_out`). Everything else — including the
action-stream `condition_embedder_action` — keeps its pretrained weights,
so SFT only has to learn the new 54-D action head on top of the frozen-
semantics world model. VAE / tokenizer / text_encoder are symlinked.

Output layout mirrors the base (usable directly as
`wan22_pretrained_model_name_or_path` for both training and inference).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ACTION_HEAD_KEYS = ("action_embedder", "action_proj_out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="lingbot-va-base dir")
    ap.add_argument("--dst", required=True, help="output dir")
    ap.add_argument("--action-dim", type=int, default=54)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    t_src = src / "transformer"
    t_dst = dst / "transformer"
    assert (t_src / "config.json").is_file(), t_src
    t_dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((t_src / "config.json").read_text())
    old_dim = cfg.get("action_dim", 30)
    cfg["action_dim"] = args.action_dim
    (t_dst / "config.json").write_text(json.dumps(cfg, indent=2))

    shards = sorted(t_src.glob("diffusion_pytorch_model*.safetensors"))
    assert shards, f"no safetensors under {t_src}"

    torch.manual_seed(args.seed)

    for index_file in t_src.glob("*.index.json"):
        shutil.copy(index_file, t_dst / index_file.name)
        print(f"[surgery] copied index {index_file.name}")

    for shard in shards:
        sd = load_file(str(shard))
        new_sd = {}
        replaced = []
        for k, v in sd.items():
            if any(k.startswith(p) or k == p for p in ACTION_HEAD_KEYS) or \
               any("." + p + "." in k for p in ACTION_HEAD_KEYS):
                new_shape = list(v.shape)
                changed = False
                if new_shape[0] == old_dim:
                    new_shape[0] = args.action_dim
                    changed = True
                elif new_shape[-1] == old_dim:
                    new_shape[-1] = args.action_dim
                    changed = True
                if not changed:
                    # e.g. action_embedder.bias [inner_dim] — shape does not
                    # depend on action_dim; keep the pretrained value.
                    new_sd[k] = v
                    continue
                nv = torch.empty(new_shape, dtype=v.dtype)
                torch.manual_seed(args.seed + hash(k) % 10000)
                fan_in = nv.reshape(-1, nv.shape[-1]).shape[-1] \
                    if nv.dim() == 2 else nv.shape[-1]
                bound = 1 / fan_in ** 0.5
                nv.uniform_(-bound, bound)
                new_sd[k] = nv
                replaced.append((k, tuple(v.shape), tuple(new_shape)))
            else:
                new_sd[k] = v
        out = t_dst / shard.name
        save_file(new_sd, str(out))
        for k, old, new in replaced:
            print(f"[surgery] {k}: {old} -> {new}")
        if not replaced:
            print(f"[surgery] {shard.name}: no action-head keys found "
                  f"(already {args.action_dim}-D?)")

    for sub in ("vae", "tokenizer", "text_encoder"):
        s, d = src / sub, dst / sub
        if d.exists() or d.is_symlink():
            continue
        os.symlink(s, d)
    print(f"[surgery] done -> {dst}")


if __name__ == "__main__":
    main()
