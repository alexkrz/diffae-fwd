import copy
import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _as_str(value):
    return value.value if hasattr(value, "value") else value


def _is_model_type_autoencoder(model_type):
    return _as_str(model_type) == "autoencoder"


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Create betas that discretize a continuous alpha_bar(t) curve."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Return a beta schedule compatible with the original diffusion configs."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    const_schedules = {
        "const0.01": 0.01,
        "const0.015": 0.015,
        "const0.008": 0.008,
        "const0.0065": 0.0065,
        "const0.0055": 0.0055,
        "const0.0045": 0.0045,
        "const0.0035": 0.0035,
        "const0.0025": 0.0025,
        "const0.0015": 0.0015,
    }
    if schedule_name in const_schedules:
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * const_schedules[schedule_name]] * num_diffusion_timesteps, dtype=np.float64)

    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def space_timesteps(num_timesteps, section_counts):
    """Select the spaced timestep subset used by DDPM/DDIM evaluation."""
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


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


@dataclass
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *args, preserve_rng_state=preserve_rng_state)
    return func(*args)


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig:
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    use_condition: bool = True
    use_conv: bool = False
    dims: int = 2
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    two_cond: bool = False
    cond_emb_channels: int = None
    has_lateral: bool = False
    lateral_channels: int = None
    use_zero_module: bool = True

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = nn.functional.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def apply_conditions(
    h, emb=None, cond=None, layers: nn.Sequential = None, scale_bias: float = 1, in_channels: int = 512
):
    two_cond = emb is not None and cond is not None
    if emb is not None:
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
    if two_cond:
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        scale_shifts = [emb, cond]
    else:
        scale_shifts = [emb]

    for i, each in enumerate(scale_shifts):
        if each is None:
            scale_shifts[i] = (None, None)
        elif each.shape[1] == in_channels * 2:
            scale_shifts[i] = torch.chunk(each, 2, dim=1)
        else:
            scale_shifts[i] = (each, None)

    biases = [scale_bias] * len(scale_shifts) if isinstance(scale_bias, Number) else scale_bias
    pre_layers, post_layers = layers[0], layers[1:]
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    h = pre_layers(h)
    for i, (scale, shift) in enumerate(scale_shifts):
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = mid_layers(h)
    h = post_layers(h)
    return h


class ResBlock(TimestepBlock):
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf
        self.in_layers = nn.Sequential(
            normalization(conf.channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.channels, conf.out_channels, 3, padding=1),
        )
        self.updown = conf.up or conf.down
        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if conf.use_condition:
            self.emb_layers = nn.Sequential(nn.SiLU(), linear(conf.emb_channels, 2 * conf.out_channels))
            if conf.two_cond:
                self.cond_emb_layers = nn.Sequential(nn.SiLU(), linear(conf.cond_emb_channels, conf.out_channels))
            conv = conv_nd(conf.dims, conf.out_channels, conf.out_channels, 3, padding=1)
            if conf.use_zero_module:
                conv = zero_module(conv)
            self.out_layers = nn.Sequential(
                normalization(conf.out_channels),
                nn.SiLU(),
                nn.Dropout(p=conf.dropout),
                conv,
            )

        if conf.out_channels == conf.channels:
            self.skip_connection = nn.Identity()
        else:
            kernel_size = 3 if conf.use_conv else 1
            padding = 1 if conf.use_conv else 0
            self.skip_connection = conv_nd(conf.dims, conf.channels, conf.out_channels, kernel_size, padding=padding)

    def forward(self, x, emb=None, cond=None, lateral=None):
        return torch_checkpoint(self._forward, (x, emb, cond, lateral), self.conf.use_checkpoint)

    def _forward(self, x, emb=None, cond=None, lateral=None):
        if self.conf.has_lateral:
            assert lateral is not None
            x = torch.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.conf.use_condition:
            emb_out = self.emb_layers(emb).type(h.dtype) if emb is not None else None
            if self.conf.two_cond:
                cond_out = self.cond_emb_layers(cond).type(h.dtype) if cond is not None else None
                if cond_out is not None:
                    while len(cond_out.shape) < len(h.shape):
                        cond_out = cond_out[..., None]
            else:
                cond_out = None
            h = apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                layers=self.out_layers,
                scale_bias=1,
                in_channels=self.conf.out_channels,
            )

        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
        self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = (
            QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        )
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x,), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Return(NamedTuple):
    pred: torch.Tensor


class UNet2DModelOutput(NamedTuple):
    sample: torch.Tensor


@dataclass
class BeatGANsUNetConfig:
    image_size: int = 64
    in_channels: int = 3
    model_channels: int = 64
    out_channels: int = 3
    num_res_blocks: int = 2
    num_input_res_blocks: int = None
    embed_channels: int = 512
    attention_resolutions: Tuple[int] = (16,)
    time_embed_channels: int = None
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    dims: int = 2
    num_classes: int = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    resblock_updown: bool = True
    use_new_attention_order: bool = False
    resnet_two_cond: bool = False
    resnet_cond_channels: int = None
    resnet_use_zero_module: bool = True
    attn_checkpoint: bool = False

    def make_model(self):
        return BeatGANsUNetModel(self)


class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf
        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads
        self.dtype = torch.float32
        self.time_emb_channels = conf.time_embed_channels or conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels),
        )

        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))]
        )
        kwargs = dict(
            use_condition=True,
            two_cond=conf.resnet_two_cond,
            use_zero_module=conf.resnet_use_zero_module,
            cond_emb_channels=conf.resnet_cond_channels,
        )

        self._feature_size = ch
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        resolution = conf.image_size
        for level, mult in enumerate(conf.input_channel_mult or conf.channel_mult):
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                            **kwargs,
                        ).make_model()
                        if conf.resblock_updown
                        else Downsample(ch, conf.conv_resample, dims=conf.dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch, conf.embed_channels, conf.dropout, dims=conf.dims, use_checkpoint=conf.use_checkpoint, **kwargs
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch, conf.embed_channels, conf.dropout, dims=conf.dims, use_checkpoint=conf.use_checkpoint, **kwargs
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    ich = 0
                layers = [
                    ResBlockConfig(
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.use_new_attention_order,
                        )
                    )
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model()
                        if conf.resblock_updown
                        else Upsample(ch, conf.conv_resample, dims=conf.dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        if conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1),
            )

    def forward(self, x, t, y=None, **kwargs):
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                hs[i].append(h)
                k += 1
        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1
        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


@dataclass
class BeatGANsEncoderConfig:
    image_size: int
    in_channels: int
    model_channels: int
    out_hid_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    dropout: float = 0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = "adaptivenonzero"

    def make_model(self):
        return BeatGANsEncoderModel(self)


class BeatGANsEncoderModel(nn.Module):
    def __init__(self, conf: BeatGANsEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = torch.float32
        self.model_channels = conf.model_channels

        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        resolution = conf.image_size
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,
                        use_checkpoint=conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                        ).make_model()
                        if conf.resblock_updown
                        else Downsample(ch, conf.conv_resample, dims=conf.dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch

        if conf.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False):
        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
        h = self.middle_block(h, emb=emb)
        h_2d = h.type(x.dtype)
        h = self.out(h_2d)
        return (h, h_2d) if return_2d_feature else h


class AutoencReturn(NamedTuple):
    pred: torch.Tensor
    cond: Optional[torch.Tensor] = None


class EmbedReturn(NamedTuple):
    emb: Optional[torch.Tensor] = None
    time_emb: Optional[torch.Tensor] = None
    style: Optional[torch.Tensor] = None


class TimeStyleSeperateEmbed(nn.Module):
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        time_emb = None if time_emb is None else self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = "depthconv"
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: object = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

    def encode(self, x):
        cond = self.encoder.forward(x)
        return {"cond": cond}

    def forward(self, x, t, y=None, x_start=None, cond=None, style=None, noise=None, t_cond=None, **kwargs):
        if t_cond is None:
            t_cond = t

        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f"{len(x)} != {len(x_start)}"
            cond = self.encode(x_start)["cond"]

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(time_emb=_t_emb, cond=cond, time_cond_emb=_t_cond_emb)
            emb = res.time_emb
            cond_emb = res.emb
        else:
            raise NotImplementedError()

        style = style or res.style
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h, emb=enc_time_emb, cond=enc_cond_emb)
                    hs[i].append(h)
                    k += 1
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None

                h = self.output_blocks[k](h, emb=dec_time_emb, cond=dec_cond_emb, lateral=lateral)
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=cond)


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


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


@dataclass
class TrainConfig:
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: str = "ddim"
    beatgans_loss_type: str = "mse"
    beatgans_model_mean_type: str = "eps"
    beatgans_model_var_type: str = "fixed_large"
    beatgans_rescale_timesteps: bool = False
    beta_scheduler: str = "linear"
    diffusion_type: str = "beatgans"
    dropout: float = 0.1
    eval_ema_every_samples: int = 200_000
    eval_every_samples: int = 200_000
    fp16: bool = False
    img_size: int = 64
    model_conf: object = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    name: str = ""
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    net_beatgans_embed_channels: int = 512
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_cond_channels: int = None
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_enc_num_res_blocks: int = 2
    net_enc_pool: str = "adaptivenonzero"
    net_latent_net_type: str = "none"
    net_num_input_res_blocks: int = None
    net_num_res_blocks: int = 2
    net_resblock_updown: bool = True
    sample_every_samples: int = 20_000
    sample_size: int = 64
    style_ch: int = 512
    T_eval: int = 1_000
    T: int = 1_000
    train_pred_xstart_detach: bool = True

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size

    @property
    def model_out_channels(self):
        return 3

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        scale = num_gpus * num_nodes
        self.eval_ema_every_samples *= scale
        self.eval_every_samples *= scale
        self.sample_every_samples *= scale
        self.batch_size *= scale
        self.batch_size_eval *= scale
        return self

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type != "beatgans":
            raise NotImplementedError(self.diffusion_type)

        if self.model_type is None:
            if _as_str(self.model_name) == "beatgans_autoenc":
                self.model_type = "autoencoder"
            elif _as_str(self.model_name) == "beatgans_ddpm":
                self.model_type = "ddpm"

        T = T if T is not None else self.T
        if _as_str(self.beatgans_gen_type) == "ddpm":
            section_counts = [T]
        elif _as_str(self.beatgans_gen_type) == "ddim":
            section_counts = f"ddim{T}"
        else:
            raise NotImplementedError(self.beatgans_gen_type)

        return SpacedDiffusionBeatGansConfig(
            gen_type=self.beatgans_gen_type,
            model_type=self.model_type,
            betas=get_named_beta_schedule(self.beta_scheduler, self.T),
            model_mean_type=self.beatgans_model_mean_type,
            model_var_type=self.beatgans_model_var_type,
            loss_type=self.beatgans_loss_type,
            rescale_timesteps=self.beatgans_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T, section_counts=section_counts),
            fp16=self.fp16,
            train_pred_xstart_detach=self.train_pred_xstart_detach,
        )

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_model_conf(self):
        if _as_str(self.model_name) == "beatgans_ddpm":
            self.model_type = "ddpm"
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
            )
            return self.model_conf

        if _as_str(self.model_name) == "beatgans_autoenc":
            self.model_type = "autoencoder"
            if _as_str(self.net_latent_net_type) != "none":
                raise NotImplementedError("Only LatentNetType.none is supported in interpolate.py")

            self.model_conf = BeatGANsAutoencConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
                latent_net_conf=None,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
            return self.model_conf

        raise NotImplementedError(self.model_name)


class DiffAEScheduler:
    """Standalone inference scheduler (diffusers-style model/scheduler split)."""

    def __init__(self, conf: TrainConfig, num_inference_steps: Optional[int] = None):
        self.conf = conf
        self.num_inference_steps = num_inference_steps
        self._eval_sampler = self._build_sampler(num_inference_steps)

    def _build_sampler(self, num_inference_steps: Optional[int]) -> SpacedDiffusionBeatGans:
        if num_inference_steps is None:
            diff_conf = self.conf.make_eval_diffusion_conf()
        else:
            diff_conf = self.conf._make_diffusion_conf(num_inference_steps)

        if not isinstance(diff_conf, SpacedDiffusionBeatGansConfig):
            raise TypeError(f"Expected SpacedDiffusionBeatGansConfig, got {type(diff_conf).__name__}")

        sampler = diff_conf.make_sampler()
        if not isinstance(sampler, SpacedDiffusionBeatGans):
            raise TypeError(f"Expected SpacedDiffusionBeatGans, got {type(sampler).__name__}")
        return sampler

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        self._eval_sampler = self._build_sampler(num_inference_steps)

    def _resolve_model(self, model: nn.Module) -> nn.Module:
        # Allow passing either the wrapper Unet2DModel or the underlying network.
        return model.unet if hasattr(model, "unet") else model

    def get_sampler(self, T: Optional[int] = None) -> SpacedDiffusionBeatGans:
        if T is None or T == self.num_inference_steps:
            return self._eval_sampler
        return self._build_sampler(T)

    @torch.no_grad()
    def reverse_sample_loop(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        cond: Optional[torch.Tensor],
        T: Optional[int],
        progress: bool = False,
    ):
        sampler = self.get_sampler(T)
        model_kwargs = {"cond": cond} if cond is not None else {}
        out = sampler.ddim_reverse_sample_loop(
            self._resolve_model(model), sample, model_kwargs=model_kwargs, progress=progress
        )
        return out["sample"]

    @torch.no_grad()
    def sample_loop(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        cond: Optional[torch.Tensor],
        T: Optional[int],
        progress: bool = False,
    ):
        sampler = self.get_sampler(T)
        model = self._resolve_model(model)
        if cond is None:
            return sampler.sample(model=model, noise=noise, progress=progress)
        return sampler.sample(model=model, noise=noise, model_kwargs={"cond": cond}, progress=progress)


class DiffAEModel(nn.Module):
    """Inference UNet wrapper with diffusers-style forward API."""

    def __init__(self, conf: TrainConfig):
        super().__init__()
        self.conf = conf

        model_conf = conf.make_model_conf()
        if not isinstance(model_conf, BeatGANsAutoencConfig):
            raise TypeError(f"Expected BeatGANsAutoencConfig, got {type(model_conf).__name__}")

        model = model_conf.make_model()
        if not isinstance(model, BeatGANsAutoencModel):
            raise TypeError(f"Expected BeatGANsAutoencModel, got {type(model).__name__}")

        # Keep the actual network on .unet so schedulers can consume it directly.
        self.unet: BeatGANsAutoencModel = model
        self.unet.requires_grad_(False)
        self.unet.eval()

    def forward(
        self,
        sample: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # Accept both diffusers-style args and internal diffusion call kwargs (x/t).
        if sample is None:
            sample = kwargs.pop("x", None)
        if timestep is None:
            timestep = kwargs.pop("t", None)
        if sample is None or timestep is None:
            raise ValueError("Unet2DModel.forward expects sample/x and timestep/t")

        cond = kwargs.pop("cond", None)
        if cond is None:
            cond = encoder_hidden_states

        model_out = self.unet(x=sample, t=timestep, cond=cond, **kwargs)
        if return_dict:
            return UNet2DModelOutput(sample=model_out.pred)
        return (model_out.pred,)

    def load_ema_state_dict(self, state_dict, strict=True):
        """Load EMA-only weights.

        Accepts either plain model keys ("encoder.*", "time_embed.*", ...)
        or full-checkpoint keys prefixed with "ema_model.".
        """
        model_keys = set(self.unet.state_dict().keys())

        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        if all(isinstance(k, str) for k in state_dict.keys()):
            if any(k.startswith("ema_model.") for k in state_dict.keys()):
                state_dict = {k[len("ema_model.") :]: v for k, v in state_dict.items() if k.startswith("ema_model.")}

        # If a full model state dict is passed in (with model/ema_model keys),
        # keep only keys that map to the single inference model.
        if not set(state_dict.keys()).issubset(model_keys):
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

        return self.unet.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def encode(self, x):
        return self.unet.encoder.forward(x)


def ffhq256_autoenc():
    """Standalone FFHQ-256 autoencoder config for interpolation notebooks/scripts."""
    conf = TrainConfig()

    # Base autoencoder defaults.
    conf.batch_size = 32
    conf.beatgans_gen_type = "ddim"
    conf.beta_scheduler = "linear"
    conf.data_name = "ffhq"
    conf.diffusion_type = "beatgans"
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = "beatgans_autoenc"
    conf.model_type = "autoencoder"
    conf.net_attn = (16,)
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = "adaptivenonzero"
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000

    # FFHQ-128 autoencoder base tweaks.
    conf.data_name = "ffhqlmdb256"
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000

    # FFHQ-256 final tweaks.
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.name = "ffhq256_autoenc"
    return conf


@dataclass
class ClsConfig:
    name: str
    style_ch: int = 512
    num_classes: int = 40
    manipulate_znormalize: bool = True


class ClsModel(nn.Module):
    """Inference-only classifier used by manipulate.ipynb."""

    def __init__(self, conf: ClsConfig):
        super().__init__()
        self.conf = conf
        self.classifier = nn.Linear(conf.style_ch, conf.num_classes)
        self.ema_classifier = copy.deepcopy(self.classifier)
        self.register_buffer("conds_mean", None)
        self.register_buffer("conds_std", None)

    def load_state_dict(self, state_dict, strict=False):
        # Classifier checkpoints may include extra keys from training modules.
        return super().load_state_dict(state_dict, strict=strict)

    def set_latent_stats(self, conds_mean: torch.Tensor, conds_std: torch.Tensor):
        self.conds_mean = conds_mean.reshape(1, -1).float()
        self.conds_std = conds_std.reshape(1, -1).float()

    def normalize(self, cond):
        if self.conds_mean is None or self.conds_std is None:
            return cond
        return (cond - self.conds_mean.to(cond.device)) / self.conds_std.to(cond.device)

    def denormalize(self, cond):
        if self.conds_mean is None or self.conds_std is None:
            return cond
        return (cond * self.conds_std.to(cond.device)) + self.conds_mean.to(cond.device)


def ffhq256_autoenc_cls():
    """Standalone classifier config for FFHQ256 autoencoder latents."""
    base_conf = ffhq256_autoenc()
    return ClsConfig(
        name="ffhq256_autoenc_cls",
        style_ch=base_conf.style_ch,
        num_classes=40,
        manipulate_znormalize=True,
    )
