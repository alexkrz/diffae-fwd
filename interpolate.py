# %%
# Imports
import torch

from src.model import ffhq256_autoenc, build_interpolator


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
