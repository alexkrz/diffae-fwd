import json

import torch

from src.diffae_unet import BeatGANsAutoencModel

# ckpt = torch.load("checkpoints/ffhq256_autoenc/last.ckpt", weights_only=False)
# state_dict = ckpt["state_dict"]
# torch.save(state_dict, "checkpoints/diffae-ffhq256-pt/ffhq_autoenc_model.pt")

with open("configs/diffae-ffhq256/autoenc_model.json", "r", encoding="utf-8") as f:
    autoenc_cfg = json.load(f)
model = BeatGANsAutoencModel(**autoenc_cfg)
state_dict = torch.load("checkpoints/diffae-ffhq256-pt/ffhq_autoenc_model.pt")
ema_state_dict = {k[len("ema_model.") :]: v for k, v in state_dict.items() if k.startswith("ema_model.")}
model.load_state_dict(ema_state_dict, strict=True)
