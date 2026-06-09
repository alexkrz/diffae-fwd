# %%
# Imports
import json
import math

import huggingface_hub
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from src.dataset import CelebAttrDataset, ImageDataset
from src.diffae_diffusion import DiffAEScheduler
from src.diffae_unet import BeatGANsAutoencModel, ClsModel

# %%
# Download model checkpoints
huggingface_hub.snapshot_download(
    repo_id="alexkrz/diffae-ffhq256",
    local_dir="checkpoints/diffae-ffhq256",
)

# %%
# Load model
device = "cuda"
# conf = ffhq256_autoenc()
with open("configs/diffae-ffhq256/autoenc_model.json", "r", encoding="utf-8") as f:
    autoenc_cfg = json.load(f)
with open("configs/diffae-ffhq256/scheduler.json", "r", encoding="utf-8") as f:
    scheduler_cfg = json.load(f)
model = BeatGANsAutoencModel(**autoenc_cfg)
scheduler = DiffAEScheduler(**scheduler_cfg)
state_dict = load_file("checkpoints/diffae-ffhq256/ffhq256_autoenc_ema.safetensors", device="cpu")
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
model.requires_grad_(False)
print("Loaded DiffAEModel + DiffAEScheduler")

# %%
# Load cls_model
cls_model = ClsModel(
    style_ch=512,
    num_classes=40,
    manipulate_znormalize=True,
)
state_dict = load_file("checkpoints/diffae-ffhq256/ffhq256_autoenc_cls.safetensors", device="cpu")
cls_model.load_state_dict(state_dict, strict=False)
# Load latent stats from autoencoder
latent_stats = load_file("checkpoints/diffae-ffhq256/ffhq256_autoenc_latent.safetensors", device="cpu")
cls_model.set_latent_stats(latent_stats["conds_mean"], latent_stats["conds_std"])
cls_model.to(device)
print("Loaded cls_model")

# %%
# Load data
data = ImageDataset(
    "data/imgs_align",
    image_size=autoenc_cfg["image_size"],
    exts=["jpg", "JPG", "png"],
    do_augment=False,
)
batch = data[0]["img"][None]
print(type(batch[0]))
print(batch.shape)  # N, C, H, W

# %%
# Encode images
batch_device = batch.to(device)
cond_dict = model.encode(batch_device)
cond = cond_dict["cond"]
xT = scheduler.reverse_sample_loop(model, batch_device, cond=cond, T=250, progress=True)

# %%
# Add condition on cls_id
cls_id = CelebAttrDataset.cls_to_id["Wavy_Hair"]
cond2 = cls_model.normalize(cond)
cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
cond2 = cls_model.denormalize(cond2)

# %%
# Render conditioned image
img: list[torch.Tensor] = (scheduler.sample_loop(model, xT, cond=cond2, T=100, progress=True) + 1) / 2

# %%
# Plot original and rendered image side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax: list[plt.Axes]
ori: torch.Tensor = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(img[0].permute(1, 2, 0).cpu())
results_dir = "results/compare_manipulated.png"
plt.savefig(results_dir)
print(f"Results saved at: {results_dir}")

# %%
