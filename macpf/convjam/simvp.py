"""SimVP congestion forecasters.

Two variants are provided:

* ``CongestionSimVP``: the earlier compact project-local SimVP-style model.
* ``OpenSTLCongestionSimVP``: an OpenSTL/SimVPv2-inspired implementation with
  ConvSC spatial encoder/decoder and a gSTA MidMetaNet translator.

Both expose the same forecasting interface used by ConvLSTM:
    enc_in : (B, T_in, 5, H, W)
    dec_in : (B, T_out, 5, H, W)
    output : (B, T_out, 1, H, W)
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

__all__ = [
    "GatedConvBlock",
    "CongestionSimVP",
    "OpenSTLCongestionSimVP",
]


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, expansion: int = 2):
        super().__init__()
        padding = kernel_size // 2
        hidden = int(channels) * int(expansion)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, kernel_size=kernel_size, padding=padding),
            nn.GLU(dim=1),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class CongestionSimVP(nn.Module):
    """Compact SimVP-style model kept for backward-compatible checkpoints."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 48,
        translator_layers: int = 4,
        kernel_size: int = 3,
        t_in: int = 60,
        t_out: int = 10,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.translator_layers = int(translator_layers)
        self.kernel_size = int(kernel_size)
        self.t_in = int(t_in)
        self.t_out = int(t_out)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        translator_channels = (self.t_in + self.t_out) * self.hidden_dim
        blocks: List[nn.Module] = [
            nn.Conv2d(translator_channels, self.t_out * self.hidden_dim, kernel_size=1),
            nn.GELU(),
        ]
        for _ in range(self.translator_layers):
            blocks.append(GatedConvBlock(self.t_out * self.hidden_dim, self.kernel_size))
        self.translator = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )

    def _encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        z = self.encoder(frames.reshape(b * t, c, h, w))
        return z.reshape(b, t, self.hidden_dim, h, w)

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        if enc_in.dim() != 5 or dec_in.dim() != 5:
            raise ValueError("enc_in/dec_in must be (B, T, C, H, W) tensors.")
        if enc_in.shape[1] != self.t_in or dec_in.shape[1] != self.t_out:
            raise ValueError(
                f"Expected t_in={self.t_in}, t_out={self.t_out}; "
                f"got {enc_in.shape[1]}, {dec_in.shape[1]}"
            )
        frames = torch.cat([enc_in, dec_in], dim=1)
        z = self._encode_sequence(frames)
        b, t, c, h, w = z.shape
        z = z.reshape(b, t * c, h, w)
        fut = self.translator(z).reshape(b, self.t_out, self.hidden_dim, h, w)
        out = self.decoder(fut.reshape(b * self.t_out, self.hidden_dim, h, w))
        return out.reshape(b, self.t_out, 1, h, w)


class DropPath(nn.Module):
    """Stochastic depth, matching timm's behavior without adding a dependency."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        upsampling: bool = False,
        act_norm: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        self.act_norm = bool(act_norm)
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size, stride=1, padding=padding),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """OpenSTL ConvSC spatial block: optional downsample/upsample + norm/activation."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
        act_norm: bool = True,
        act_inplace: bool = True,
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsampling=upsampling,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def sampling_generator(n: int, reverse: bool = False) -> list[bool]:
    samplings = [False, True] * (int(n) // 2 + 1)
    samplings = samplings[: int(n)]
    return list(reversed(samplings)) if reverse else samplings


class Encoder(nn.Module):
    """OpenSTL SimVP spatial encoder."""

    def __init__(self, c_in: int, c_hid: int, n_s: int, spatio_kernel: int, act_inplace: bool = True):
        super().__init__()
        samplings = sampling_generator(n_s)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace),
            *[
                ConvSC(c_hid, c_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace)
                for s in samplings[1:]
            ],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc[0](x)
        latent = enc1
        for layer in self.enc[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    """OpenSTL SimVP spatial decoder with first-layer skip connection."""

    def __init__(self, c_hid: int, c_out: int, n_s: int, spatio_kernel: int, act_inplace: bool = True):
        super().__init__()
        samplings = sampling_generator(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvSC(c_hid, c_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace)
                for s in samplings[:-1]
            ],
            ConvSC(c_hid, c_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace),
        )
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1)

    def forward(self, hid: torch.Tensor, enc1: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.dec[:-1]:
            hid = layer(hid)
        if enc1 is not None and hid.shape == enc1.shape:
            hid = hid + enc1
        y = self.dec[-1](hid)
        return self.readout(y)


class MixMlp(nn.Module):
    """Convolutional MLP used by OpenSTL gSTA blocks."""

    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class AttentionModule(nn.Module):
    """Large-kernel spatial gating attention from OpenSTL's GASubBlock."""

    def __init__(self, dim: int, kernel_size: int = 21, dilation: int = 3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_x, g_x = torch.chunk(self.conv1(attn), 2, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 21, attn_shortcut: bool = True):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim, kernel_size=kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_shortcut = bool(attn_shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut if self.attn_shortcut else x


class GASubBlock(nn.Module):
    """OpenSTL gSTA block: spatial gating attention + convolutional MLP."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        init_value: float = 1e-2,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MixMlp(dim, int(dim * mlp_ratio), drop=drop)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale1 = self.layer_scale_1.view(1, -1, 1, 1)
        scale2 = self.layer_scale_2.view(1, -1, 1, 1)
        x = x + self.drop_path(scale1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(scale2 * self.mlp(self.norm2(x)))
        return x


class MetaBlock(nn.Module):
    """Minimal OpenSTL MetaBlock supporting the default gSTA model type."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_type: str = "gsta",
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        model_type = str(model_type).lower()
        if model_type not in {"gsta", "ga"}:
            raise ValueError("This lightweight OpenSTL port currently supports model_type='gsta' only.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.block = GASubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.reduction = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(self.block(x))


class MidMetaNet(nn.Module):
    """OpenSTL-style hidden translator.

    Original SimVP maps T*hidden channels back to T*hidden channels. For AMR
    forecasting we adapt it to map (T_in + T_out_known)*hidden to T_out*hidden.
    """

    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        channel_out: int,
        n_t: int,
        model_type: str = "gsta",
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if int(n_t) < 2:
            raise ValueError("n_t must be >= 2 for MidMetaNet.")
        n_t = int(n_t)
        dpr = torch.linspace(1e-2, float(drop_path), n_t).tolist()
        layers: list[nn.Module] = [
            MetaBlock(channel_in, channel_hid, model_type, mlp_ratio, drop, dpr[0])
        ]
        for i in range(1, n_t - 1):
            layers.append(MetaBlock(channel_hid, channel_hid, model_type, mlp_ratio, drop, dpr[i]))
        layers.append(MetaBlock(channel_hid, channel_out, model_type, mlp_ratio, drop, float(drop_path)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b, t * c, h, w)
        return self.net(x)


class OpenSTLCongestionSimVP(nn.Module):
    """OpenSTL SimVP/gSTA adapted to AMR congestion forecasting."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        hid_s: int = 16,
        hid_t: int = 256,
        n_s: int = 4,
        n_t: int = 4,
        model_type: str = "gsta",
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        t_in: int = 60,
        t_out: int = 10,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hid_s = int(hid_s)
        self.hid_t = int(hid_t)
        self.n_s = int(n_s)
        self.n_t = int(n_t)
        self.model_type = str(model_type)
        self.mlp_ratio = float(mlp_ratio)
        self.drop = float(drop)
        self.drop_path = float(drop_path)
        self.spatio_kernel_enc = int(spatio_kernel_enc)
        self.spatio_kernel_dec = int(spatio_kernel_dec)
        self.t_in = int(t_in)
        self.t_out = int(t_out)

        self.enc = Encoder(self.in_channels, self.hid_s, self.n_s, self.spatio_kernel_enc, act_inplace=False)
        self.hid = MidMetaNet(
            channel_in=(self.t_in + self.t_out) * self.hid_s,
            channel_hid=self.hid_t,
            channel_out=self.t_out * self.hid_s,
            n_t=self.n_t,
            model_type=self.model_type,
            mlp_ratio=self.mlp_ratio,
            drop=self.drop,
            drop_path=self.drop_path,
        )
        self.dec = Decoder(self.hid_s, self.out_channels, self.n_s, self.spatio_kernel_dec, act_inplace=False)

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        if enc_in.dim() != 5 or dec_in.dim() != 5:
            raise ValueError("enc_in/dec_in must be (B, T, C, H, W) tensors.")
        if enc_in.shape[1] != self.t_in or dec_in.shape[1] != self.t_out:
            raise ValueError(
                f"Expected t_in={self.t_in}, t_out={self.t_out}; "
                f"got {enc_in.shape[1]}, {dec_in.shape[1]}"
            )
        frames = torch.cat([enc_in, dec_in], dim=1)
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        embed, skip = self.enc(x)
        _, c_hid, h_hid, w_hid = embed.shape
        z = embed.reshape(b, t, c_hid, h_hid, w_hid)
        hid = self.hid(z).reshape(b * self.t_out, c_hid, h_hid, w_hid)

        # OpenSTL decoder uses the first encoder block as a skip. For the adapted
        # horizon we provide the most recent/future-known skips aligned to t_out.
        skip_seq = skip.reshape(b, t, c_hid, skip.shape[-2], skip.shape[-1])
        skip_future = skip_seq[:, -self.t_out :, ...].reshape(b * self.t_out, c_hid, skip.shape[-2], skip.shape[-1])
        y = self.dec(hid, skip_future)
        return y.reshape(b, self.t_out, self.out_channels, h, w)
