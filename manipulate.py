# %%
# Imports
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.dataset import CelebAttrDataset, ImageDataset
from src.model import ClsModel, MinLitModel, ffhq256_autoenc, ffhq256_autoenc_cls

# %%
# Load ema_model
device = "cuda:0"
conf = ffhq256_autoenc()
model = MinLitModel(conf)
state = torch.load(
    f"checkpoints/{conf.name}/last.ckpt",
    map_location="cpu",
    weights_only=False,  # The model checkpoint contains Pytorch Lightning metadata, so we need to load it with weights_only=False
)
model.load_state_dict(state["state_dict"], strict=False)
model.ema_model.to(device).eval()
print("Loaded ema_model")

# %%
# Load cls_model
cls_conf = ffhq256_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(
    f"checkpoints/{cls_conf.name}/last.ckpt",
    map_location="cpu",
    weights_only=False,
)
print("latent step:", state["global_step"])
cls_model.load_state_dict(state["state_dict"], strict=False)
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
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

# %%
# Add condition on cls_id
cls_id = CelebAttrDataset.cls_to_id["Wavy_Hair"]
cond2 = cls_model.normalize(cond)
cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
cond2 = cls_model.denormalize(cond2)

# %%
# Render conditioned image
img = model.render(xT, cond2, T=100)

# %%
# Plot original and rendered image side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax: list[plt.Axes]
ori: torch.Tensor = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(img[0].permute(1, 2, 0).cpu())
plt.savefig("results/compare_manipulated.png")

# %%
