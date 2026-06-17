# Extended version of unet.py from https://github.com/openai/guided-diffusion
import copy
import math
from abc import abstractmethod
from numbers import Number
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *args, preserve_rng_state=preserve_rng_state)
    return func(*args)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

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
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

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
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_condition=True,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        two_cond=False,
        cond_emb_channels=None,
        has_lateral=False,
        lateral_channels=None,
        use_zero_module=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_condition = use_condition
        self.use_conv = use_conv
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.up = up
        self.down = down
        self.two_cond = two_cond
        self.cond_emb_channels = cond_emb_channels or emb_channels
        self.has_lateral = has_lateral
        self.lateral_channels = lateral_channels
        self.use_zero_module = use_zero_module

        self.in_layers = nn.Sequential(
            normalization(self.channels),
            nn.SiLU(),
            conv_nd(self.dims, self.channels, self.out_channels, 3, padding=1),
        )
        self.updown = self.up or self.down
        if self.up:
            self.h_upd = Upsample(self.channels, False, self.dims)
            self.x_upd = Upsample(self.channels, False, self.dims)
        elif self.down:
            self.h_upd = Downsample(self.channels, False, self.dims)
            self.x_upd = Downsample(self.channels, False, self.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.use_condition:
            self.emb_layers = nn.Sequential(nn.SiLU(), linear(self.emb_channels, 2 * self.out_channels))
            if self.two_cond:
                self.cond_emb_layers = nn.Sequential(nn.SiLU(), linear(self.cond_emb_channels, self.out_channels))
            conv = conv_nd(self.dims, self.out_channels, self.out_channels, 3, padding=1)
            if self.use_zero_module:
                conv = zero_module(conv)
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=self.dropout),
                conv,
            )

        if self.out_channels == self.channels:
            self.skip_connection = nn.Identity()
        else:
            kernel_size = 3 if self.use_conv else 1
            padding = 1 if self.use_conv else 0
            self.skip_connection = conv_nd(self.dims, self.channels, self.out_channels, kernel_size, padding=padding)

    def forward(self, x, emb=None, cond=None, lateral=None):
        return torch_checkpoint(self._forward, (x, emb, cond, lateral), self.use_checkpoint)

    def _forward(self, x, emb=None, cond=None, lateral=None):
        if self.has_lateral:
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

        if self.use_condition:
            emb_out = self.emb_layers(emb).type(h.dtype) if emb is not None else None
            if self.two_cond:
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
                in_channels=self.out_channels,
            )

        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
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
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
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
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

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
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size=64,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        num_input_res_blocks=None,
        embed_channels=512,
        attention_resolutions=(16,),
        time_embed_channels=None,
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        input_channel_mult=None,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=True,
        use_new_attention_order=False,
        resnet_two_cond=False,
        resnet_cond_channels=None,
        resnet_use_zero_module=True,
        attn_checkpoint=False,
    ):
        super().__init__()

        self.channel_mult = channel_mult
        self.model_channels = model_channels
        self.resnet_two_cond = resnet_two_cond
        if num_heads_upsample == -1:
            self.num_heads_upsample = num_heads
        else:
            self.num_heads_upsample = num_heads_upsample
        self.dtype = torch.float32

        self.time_emb_channels = time_embed_channels or model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, embed_channels),
            nn.SiLU(),
            linear(embed_channels, embed_channels),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        kwargs = dict(
            use_condition=True,
            two_cond=resnet_two_cond,
            use_zero_module=resnet_use_zero_module,
            cond_emb_channels=resnet_cond_channels,
        )

        self._feature_size = ch
        input_block_chans = [[] for _ in range(len(channel_mult))]
        input_block_chans[0].append(ch)
        self.input_num_blocks = [0 for _ in range(len(channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(channel_mult))]

        resolution = image_size
        for level, mult in enumerate(input_channel_mult or channel_mult):
            for _ in range(num_input_res_blocks or num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_channels,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        **kwargs,
                    )
                ]
                ch = int(mult * model_channels)
                if resolution in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint or attn_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
            if level != len(channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_channels,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            down=True,
                            **kwargs,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, embed_channels, dropout, dims=dims, use_checkpoint=use_checkpoint, **kwargs),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint or attn_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(ch, embed_channels, dropout, dims=dims, use_checkpoint=use_checkpoint, **kwargs),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    ich = 0
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=embed_channels,
                        dropout=dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    )
                ]
                ch = int(model_channels * mult)
                if resolution in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint or attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            embed_channels,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            up=True,
                            **kwargs,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        if resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(dims, input_ch, out_channels, 3, padding=1),
            )

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = [[] for _ in range(len(self.channel_mult))]
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
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_hid_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        use_time_condition=True,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptivenonzero",
    ):
        super().__init__()
        self.dtype = torch.float32
        self.model_channels = model_channels
        self.use_time_condition = use_time_condition

        if use_time_condition:
            time_embed_dim = model_channels * 4
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        resolution = image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_condition=use_time_condition,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = int(mult * model_channels)
                if resolution in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
            if level != len(channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_condition=use_time_condition,
                            use_checkpoint=use_checkpoint,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_condition=use_time_condition,
                use_checkpoint=use_checkpoint,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_condition=use_time_condition,
                use_checkpoint=use_checkpoint,
            ),
        )
        self._feature_size += ch

        if pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(dims, ch, out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False):
        if self.use_time_condition:
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
    def __init__(
        self,
        image_size=64,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        num_input_res_blocks=None,
        embed_channels=512,
        attention_resolutions=(16,),
        time_embed_channels=None,
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        input_channel_mult=None,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=True,
        use_new_attention_order=False,
        resnet_two_cond=False,
        resnet_cond_channels=None,
        resnet_use_zero_module=True,
        attn_checkpoint=False,
        enc_out_channels=512,
        enc_attn_resolutions=None,
        enc_pool="depthconv",
        enc_num_res_block=2,
        enc_channel_mult=None,
        enc_grad_checkpoint=False,
        latent_net_conf=None,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            num_input_res_blocks=num_input_res_blocks,
            embed_channels=embed_channels,
            attention_resolutions=attention_resolutions,
            time_embed_channels=time_embed_channels,
            dropout=dropout,
            channel_mult=channel_mult,
            input_channel_mult=input_channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            resnet_two_cond=resnet_two_cond,
            resnet_cond_channels=resnet_cond_channels,
            resnet_use_zero_module=resnet_use_zero_module,
            attn_checkpoint=attn_checkpoint,
        )
        self.model_channels = model_channels
        self.resnet_two_cond = resnet_two_cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=model_channels,
            time_out_channels=embed_channels,
        )

        self.encoder = BeatGANsEncoderModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_hid_channels=enc_out_channels,
            out_channels=enc_out_channels,
            num_res_blocks=enc_num_res_block,
            attention_resolutions=(enc_attn_resolutions or attention_resolutions),
            dropout=dropout,
            channel_mult=enc_channel_mult or channel_mult,
            use_time_condition=False,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint or enc_grad_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            pool=enc_pool,
        )

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
            _t_emb = timestep_embedding(t, self.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.model_channels)
        else:
            _t_emb = None
            _t_cond_emb = None

        if self.resnet_two_cond:
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

        hs = [[] for _ in range(len(self.channel_mult))]

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
            hs = [[] for _ in range(len(self.channel_mult))]

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


class ClsModel(nn.Module):
    """Inference-only classifier used by manipulate.py."""

    def __init__(
        self,
        *,
        style_ch: int = 512,
        num_classes: int = 40,
        manipulate_znormalize: bool = True,
    ):
        super().__init__()
        self.style_ch = style_ch
        self.num_classes = num_classes
        self.manipulate_znormalize = manipulate_znormalize
        self.classifier = nn.Linear(style_ch, num_classes)
        self.ema_classifier = copy.deepcopy(self.classifier)
        self.register_buffer("conds_mean", None)
        self.register_buffer("conds_std", None)

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


if __name__ == "__main__":
    # Example for ffhq256 dataset
    diffae_auotoencoder = BeatGANsAutoencModel(
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
