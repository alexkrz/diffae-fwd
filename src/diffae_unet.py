import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from .nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


# Configs
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


# Definitions
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


class AutoencReturn(NamedTuple):
    pred: torch.Tensor
    cond: Optional[torch.Tensor] = None


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


if __name__ == "__main__":
    # Test construction of Encoder
    encoder_cfg = BeatGANsEncoderConfig(
        image_size=256,
        in_channels=3,
        model_channels=128,
        out_hid_channels=512,
        out_channels=512,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        channel_mult=(1, 1, 2, 2, 4, 4, 4),
        use_time_condition=False,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        resblock_updown=True,
        use_new_attention_order=False,
        pool="adaptivenonzero",
    )

    diffae_encoder = BeatGANsEncoderModel(encoder_cfg)
