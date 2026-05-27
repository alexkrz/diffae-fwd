# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import ImageDataset
from src.model import MinLitModel, ffhq256_autoenc

# %%
# Load model
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
# Load dataset
data = ImageDataset("data/imgs_interpolate", image_size=conf.img_size, exts=["jpg", "JPG", "png"], do_augment=False)
batch = torch.stack(
    [
        data[0]["img"],
        data[1]["img"],
    ]
)
print(type(batch[0]))
print(batch[0].shape)  # C, H, W
ori = (batch + 1) / 2  # Undo normalization

# %%
# Encode images
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

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

pred: list[torch.Tensor] = model.render(intp_x, intp, T=20)

# %%
# Plot interpolation results
fig, ax = plt.subplots(1, 10, figsize=(5 * 10, 5))
ax: list[plt.Axes]
for i in range(len(alpha)):
    ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
plt.savefig("results/compare_interpolated.png")

# %%
