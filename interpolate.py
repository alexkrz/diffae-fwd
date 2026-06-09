# %%
# Imports
import json

import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from src.dataset import ImageDataset
from src.diffae_diffusion import SpacedDiffusionBeatGans
from src.diffae_unet import BeatGANsAutoencModel
from src.model import DiffAEModel, DiffAEScheduler, ffhq256_autoenc

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
model = BeatGANsAutoencModel(**autoenc_cfg)
scheduler = SpacedDiffusionBeatGans(
    gen_type="ddim",
    betas=np.linspace(0.0001, 0.02, 1000, dtype=np.float64),
    model_type="autoencoder",
    model_mean_type="eps",
    model_var_type="fixed_large",
    loss_type="mse",
    rescale_timesteps=False,
    fp16=True,
    train_pred_xstart_detach=True,
    use_timesteps=tuple(range(0, 1000, 50)),
)
state_dict = load_file("checkpoints/diffae-ffhq256/ffhq256_autoenc_ema.safetensors", device="cpu")
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
model.requires_grad_(False)
print("Loaded DiffAEModel + DiffAEScheduler")

# %%
# Load dataset
data = ImageDataset(
    "data/imgs_interpolate",
    image_size=autoenc_cfg["image_size"],
    exts=["jpg", "JPG", "png"],
    do_augment=False,
)
batch = torch.stack([data[0]["img"], data[1]["img"]])
print(type(batch[0]))
print(batch[0].shape)  # C, H, W
ori = (batch + 1) / 2  # Undo normalization

# %%
# Encode images
batch_device = batch.to(device)
cond_dict = model.encode(batch_device)
cond = cond_dict["cond"]
xT = scheduler.reverse_sample_loop(model, batch_device, cond=cond, T=250, progress=True)

# %%
# Perform interpolation
# Semantic codes are interpolated using convex combination, while stochastic codes are interpolated using spherical linear interpolation
alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device)
intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]


def cos(a: torch.Tensor, b: torch.Tensor):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


theta = torch.arccos(cos(xT[0], xT[1]))
x_shape = xT[0].shape
intp_x = (
    torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None]
    + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]
) / torch.sin(theta)
intp_x = intp_x.view(-1, *x_shape)

pred: list[torch.Tensor] = (scheduler.sample_loop(model, intp_x, cond=intp, T=20, progress=True) + 1) / 2

# %%
# Plot interpolation results
fig, ax = plt.subplots(1, 10, figsize=(5 * 10, 5))
ax: list[plt.Axes]
for i in range(len(alpha)):
    ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
results_dir = "results/compare_interpolated.png"
plt.savefig(results_dir)
print(f"Results saved at: {results_dir}")

# %%
