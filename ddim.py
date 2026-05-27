# %%
# Imports
import numpy as np
import torch
import tqdm
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from PIL import Image

# %%
# Load model
model_id = "google/ddpm-celebahq-256"
# ddim = DDIMPipeline.from_pretrained(model_id)
model = UNet2DModel.from_pretrained(model_id)
model.to("cuda")
print("Loaded model")

# %%
# Initialize scheduler
scheduler = DDIMScheduler.from_config(model_id)
scheduler.set_timesteps(num_inference_steps=50)

# %%
# Generate noisy sample
torch.manual_seed(0)
noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
noisy_sample = noisy_sample.to("cuda")
print(noisy_sample.shape)

# %%
# Perform denoising
sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute previous image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    # if (i + 1) % 10 == 0:
    #     display_sample(sample, i + 1)

# %%
# Save image
image_processed = sample.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = Image.fromarray(image_processed[0])
image_pil.save("results/ddim_generated_image.png")

# %%
