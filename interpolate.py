# %%
# Imports
import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from src.dataset import ImageDataset
from src.diffae_unet import BeatGANsAutoencConfig, BeatGANsAutoencModel
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
conf = ffhq256_autoenc()
autoenc_cfg = BeatGANsAutoencConfig(
    image_size=256,
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=(16,),
    dropout=0.1,
    channel_mult=(1, 1, 2, 2, 4, 4),
    conv_resample=True,
    dims=2,
    use_checkpoint=False,
    num_heads=1,
    num_head_channels=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    # Additional BeatGANsAutoencConfig / BeatGANsUNetConfig args.
    num_classes=None,
    num_heads_upsample=-1,
    num_input_res_blocks=None,
    embed_channels=512,
    resnet_two_cond=True,
    resnet_use_zero_module=True,
    resnet_cond_channels=None,
    enc_out_channels=512,
    enc_attn_resolutions=None,
    enc_pool="adaptivenonzero",
    enc_num_res_block=2,
    enc_channel_mult=(1, 1, 2, 2, 4, 4, 4),
    enc_grad_checkpoint=False,
    latent_net_conf=None,
)
model = BeatGANsAutoencModel(autoenc_cfg)
scheduler = DiffAEScheduler(conf)
state_dict = load_file("checkpoints/diffae-ffhq256/ffhq256_autoenc_ema.safetensors", device="cpu")
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
print("Loaded DiffAEModel + DiffAEScheduler")

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
batch_device = batch.to(device)
cond = model.encode(batch_device)
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
