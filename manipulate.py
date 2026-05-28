# %%
# Imports
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from src.dataset import CelebAttrDataset, ImageDataset
from src.model import ClsModel, DiffAEModel, DiffAEScheduler, ffhq256_autoenc, ffhq256_autoenc_cls

# %%
# Load ema_model
device = "cuda:0"
conf = ffhq256_autoenc()
unet = DiffAEModel(conf)
scheduler = DiffAEScheduler(conf)
state_dict = load_file("checkpoints/safetensors/ffhq256_autoenc_ema.safetensors", device="cpu")
unet.load_ema_state_dict(state_dict, strict=False)
unet.to(device).eval()
print("Loaded DiffAEModel + DiffAEScheduler")

# %%
# Load cls_model
cls_conf = ffhq256_autoenc_cls()
cls_model = ClsModel(cls_conf)
state_dict = load_file("checkpoints/safetensors/ffhq256_autoenc_cls.safetensors", device="cpu")
cls_model.load_state_dict(state_dict, strict=False)
# Load latent stats from autoencoder
latent_stats = load_file("checkpoints/safetensors/ffhq256_autoenc_latent.safetensors", device="cpu")
cls_model.set_latent_stats(latent_stats["conds_mean"], latent_stats["conds_std"])
cls_model.to(device)
print("Loaded cls_model")

# %%
# Load data
data = ImageDataset("data/imgs_align", image_size=conf.img_size, exts=["jpg", "JPG", "png"], do_augment=False)
batch = data[0]["img"][None]
print(type(batch[0]))
print(batch.shape)  # N, C, H, W

# %%
# Encode images
batch_device = batch.to(device)
cond = unet.encode(batch_device)
xT = scheduler.reverse_sample_loop(unet, batch_device, cond=cond, T=250)

# %%
# Add condition on cls_id
cls_id = CelebAttrDataset.cls_to_id["Wavy_Hair"]
cond2 = cls_model.normalize(cond)
cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
cond2 = cls_model.denormalize(cond2)

# %%
# Render conditioned image
img = (scheduler.sample_loop(unet, xT, cond=cond2, T=100) + 1) / 2

# %%
# Plot original and rendered image side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax: list[plt.Axes]
ori: torch.Tensor = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(img[0].permute(1, 2, 0).cpu())
plt.savefig("results/compare_manipulated.png")

# %%
