import json

import torch

from src.diffae_unet import BeatGANsAutoencModel, ClsModel


def convert():
    ckpt = torch.load("checkpoints/ffhq256_autoenc_cls/last.ckpt", weights_only=False)
    state_dict = ckpt["state_dict"]
    torch.save(state_dict, "checkpoints/diffae-ffhq256-pt/ffhq_autoenc_cls.pt")


def test_autoenc():
    with open("configs/diffae-ffhq256/autoenc_model.json", "r", encoding="utf-8") as f:
        autoenc_cfg = json.load(f)
    model = BeatGANsAutoencModel(**autoenc_cfg)
    state_dict = torch.load("checkpoints/diffae-ffhq256-pt/ffhq_autoenc_model.pt")
    ema_state_dict = {k[len("ema_model.") :]: v for k, v in state_dict.items() if k.startswith("ema_model.")}
    model.load_state_dict(ema_state_dict, strict=True)


def test_cls():
    cls_model = ClsModel(
        style_ch=512,
        num_classes=40,
        manipulate_znormalize=True,
    )
    state_dict = torch.load("checkpoints/diffae-ffhq256-pt/ffhq_autoenc_cls.pt")
    cls_model.load_state_dict(state_dict, strict=False)
    latent_stats = torch.load("checkpoints/diffae-ffhq256-pt/ffhq_autoenc_latent.pt")
    cls_model.set_latent_stats(latent_stats["conds_mean"], latent_stats["conds_std"])


if __name__ == "__main__":
    convert()
    # test_autoenc()
    test_cls()
