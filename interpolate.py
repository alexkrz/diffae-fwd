# %%
# Imports
import torch

from src.dataset import ImageDataset
from src.model import build_interpolator, ffhq256_autoenc

# %%
# Load model
conf = ffhq256_autoenc()
model = build_interpolator(
    conf=conf,
    checkpoint_path=f"checkpoints/{conf.name}/last.ckpt",
    device="cuda:0",
    strict=False,
)

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
