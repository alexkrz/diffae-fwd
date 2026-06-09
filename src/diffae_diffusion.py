from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# Configs
@dataclass
class GaussianDiffusionBeatGansConfig:
    gen_type: str
    betas: Tuple[float]
    model_type: str
    model_mean_type: str
    model_var_type: str
    loss_type: str
    rescale_timesteps: bool
    fp16: bool
    train_pred_xstart_detach: bool = True

    def make_sampler(self):
        return GaussianDiffusionBeatGans(self)


@dataclass
class SpacedDiffusionBeatGansConfig(GaussianDiffusionBeatGansConfig):
    use_timesteps: Tuple[int] = None

    def make_sampler(self):
        return SpacedDiffusionBeatGans(self)


# Defintions
def _as_str(value):
    return value.value if hasattr(value, "value") else value


def _is_model_type_autoencoder(model_type):
    return _as_str(model_type) == "autoencoder"


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusionBeatGans:
    def __init__(self, conf: GaussianDiffusionBeatGansConfig):
        self.conf = conf
        self.model_mean_type = _as_str(conf.model_mean_type)
        self.model_var_type = _as_str(conf.model_var_type)
        self.loss_type = _as_str(conf.loss_type)
        self.rescale_timesteps = conf.rescale_timesteps

        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def sample(
        self,
        model,
        shape=None,
        noise=None,
        cond=None,
        x_start=None,
        clip_denoised=True,
        model_kwargs=None,
        progress=False,
    ):
        if model_kwargs is None:
            model_kwargs = {}
            if _is_model_type_autoencoder(self.conf.model_type):
                model_kwargs["x_start"] = x_start
                model_kwargs["cond"] = cond

        if _as_str(self.conf.gen_type) == "ddpm":
            return self.p_sample_loop(
                model,
                shape=shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                progress=progress,
            )
        if _as_str(self.conf.gen_type) == "ddim":
            return self.ddim_sample_loop(
                model,
                shape=shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                progress=progress,
            )
        raise NotImplementedError()

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, _C = x.shape[:2]
        assert t.shape == (B,)
        with torch.amp.autocast("cuda", enabled=self.conf.fp16):
            model_forward = model.forward(x=x, t=self._scale_timesteps(t), **model_kwargs)
        model_output = model_forward.pred

        if self.model_var_type in ["fixed_large", "fixed_small"]:
            model_variance, model_log_variance = {
                "fixed_large": (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                "fixed_small": (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        else:
            raise NotImplementedError(self.model_var_type)

        def process_xstart(x_start):
            if denoised_fn is not None:
                x_start = denoised_fn(x_start)
            if clip_denoised:
                return x_start.clamp(-1, 1)
            return x_start

        if self.model_mean_type == "eps":
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "model_forward": model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None:
            raise NotImplementedError("cond_fn path is not implemented in interpolate.py")
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            raise NotImplementedError("cond_fn path is not implemented in interpolate.py")

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = torch.randn_like(x)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model,
        x,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        all_t = []
        indices = list(range(self.num_timesteps))
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        sample = x
        for i in indices:
            t = torch.tensor([i] * len(sample), device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(
                    model,
                    sample,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                sample = out["sample"]
                sample_t.append(sample)
                xstart_t.append(out["pred_xstart"])
                all_t.append(t)

        return {"sample": sample, "sample_t": sample_t, "xstart_t": xstart_t, "T": all_t}

    def ddim_sample_loop(
        self,
        model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            if isinstance(model_kwargs, list):
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out["t"] = t
                yield out
                img = out["sample"]


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def forward(self, x, t, t_cond=None, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)

        def do(in_t):
            new_ts = map_tensor[in_t]
            if self.rescale_timesteps:
                new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            return new_ts

        if t_cond is not None:
            t_cond = do(t_cond)

        return self.model(x=x, t=do(t), t_cond=t_cond, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(name)


class SpacedDiffusionBeatGans(GaussianDiffusionBeatGans):
    def __init__(self, conf: SpacedDiffusionBeatGansConfig):
        self.conf = conf
        self.use_timesteps = set(conf.use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(conf.betas)

        base_diffusion = GaussianDiffusionBeatGans(conf)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        conf.betas = np.array(new_betas)
        super().__init__(conf)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


if __name__ == "__main__":
    # Example for ffhq256 dataset
    cfg = SpacedDiffusionBeatGansConfig(
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
    scheduler = SpacedDiffusionBeatGans(cfg)
