from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from ..._shared.timm import DropPath, trunc_normal_

from .attention import (
    in_migraphx_export,
    pad_heads,
    window_attention,
)
from .inference import (
    Derived as _Derived,
    add_into as _sum,
    border_window_mask,
    img2windows,
    shifted_window_attention,
    shifted_window_mask,
    windows2img,
    padded_rows as _padded_rows,
    traced as _traced
)
from ..._shared.padded_norm import KernelLayerNorm


def _unit_length(x):
    """x scaled to unit length over its tokens, dimension 2: the length of many tokens in
    FP32, where FP16's sum of squares overflows, and in FP16 only the division."""
    if x.dtype != torch.float16:
        return F.normalize(x, dim=2)
    length = torch.linalg.vector_norm(x, dim=2, keepdim=True, dtype=torch.float32)
    # FP16's smallest normal number: a head this small is all zeros in FP16 anyway
    return x / length.clamp_min(2.0**-14).half()


def _row_gram(q, k, H, W):
    """q's channels against k's over every token, (batch, heads, d, d), for q and k of
    unit length over the tokens: a product per image row, the rows summed in FP32. As one
    product over every token, cuBLAS picks a kernel of a few blocks: 2.1 ms a block for
    DAT2 at 512x512 in FP16. A row's sums stay within [-1, 1]."""
    B, heads, _, d = q.shape
    rows = q.reshape(B, heads, H, W, d).transpose(-2, -1) @ k.reshape(B, heads, H, W, d)
    wide = torch.promote_types(q.dtype, torch.float32)
    return rows.sum(2, dtype=wide).to(q.dtype)


class SpatialGate(_Derived):
    """Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = KernelLayerNorm(dim)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, _N, C = x.shape
        if not _traced():
            # channels last: the convolution reads and writes the tokens where they are
            x2 = self.norm(x2).reshape(B, H, W, C // 2).permute(0, 3, 1, 2)
            return x1 * self.run_conv("conv", self.conv, x2).permute(0, 2, 3, 1).flatten(1, 2)
        x2 = (
            self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )

        return x1 * x2


class SGFN(nn.Module):
    """Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Spatial_Attention(_Derived):
    """Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(
        self,
        dim,
        idx,
        split_size=[8, 8],
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
        position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            raise ValueError(f"ERROR MODE: {idx}")
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def cached_bias(self):
        """relative_position_bias, computed once: its input is constant, so in inference it
        changes only with the weights, the device or the type."""
        if _traced():
            return self.relative_position_bias()
        return self.keep("bias", self.relative_position_bias)

    def relative_position_bias(self):
        pos = self.pos(self.rpe_biases)
        # select position bias
        relative_position_bias = pos[self.relative_position_index.view(-1)].view(
            self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (3, B, H, W, C), H, W, mask: (nW, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        # partition the q,k,v, image to window, channels last throughout
        windows = img2windows(qkv, self.H_sp, self.W_sp, self.num_heads)
        q, k, v = windows[0], windows[1], windows[2]

        # B head N C @ B head C N --> B head N N, plus the drpe and, for shifted
        # windows, the mask; softmax; @ B head N C
        x = window_attention(
            q,
            k,
            v,
            self.cached_bias() if self.position_bias else None,
            mask,
            self.scale,
            self.attn_drop.p if self.training else 0.0,
        )

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x

    def forward_shifted(self, qkv, H, W, shift_h, shift_w, border_mask):
        """shifted_window_attention on these windows."""
        return shifted_window_attention(
            qkv,
            H,
            W,
            self.H_sp,
            self.W_sp,
            shift_h,
            shift_w,
            self.num_heads,
            self.cached_bias() if self.position_bias else None,
            border_mask,
            self.scale,
            self.attn_drop.p if self.training else 0.0,
        )


def _dwconv(attention, x):
    """attention.dwconv on channels-last x."""
    conv, norm, activation = attention.dwconv
    return activation(attention.run_conv("dwconv", conv, x, norm))


def _channel_map(attention, x):
    """attention.channel_interaction on channels-last x."""
    _, conv_0, norm, activation, conv_1 = attention.channel_interaction
    x = attention.run_conv("channel 0", conv_0, x.mean((2, 3), keepdim=True), norm)
    return attention.run_conv("channel 1", conv_1, activation(x))


def _spatial_map(attention, x):
    """attention.spatial_interaction on channels-last x."""
    conv_0, norm, activation, conv_1 = attention.spatial_interaction
    x = attention.run_conv("spatial 0", conv_0, x, norm)
    return attention.run_conv("spatial 1", conv_1, activation(x))


class Adaptive_Spatial_Attention(_Derived):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """

    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert (
            0 <= self.shift_size[0] < self.split_size[0]
        ), "shift_size must in 0-split_size0"
        assert (
            0 <= self.shift_size[1] < self.split_size[1]
        ), "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList(
            [
                Spatial_Attention(
                    dim // 2,
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]
        )
        # The masks of the wrapping windows by size, one set for all the model's blocks
        self.border_masks = {}

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            attn_mask = self.calculate_mask(
                self.patches_resolution, self.patches_resolution
            )
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def calculate_mask(self, H, W, dtype=None, device=None):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        attn_mask_0 = shifted_window_mask(
            H,
            W,
            self.split_size[0],
            self.split_size[1],
            self.shift_size[0],
            self.shift_size[1],
            dtype,
            device,
        )
        attn_mask_1 = shifted_window_mask(
            H,
            W,
            self.split_size[1],
            self.split_size[0],
            self.shift_size[1],
            self.shift_size[0],
            dtype,
            device,
        )
        return attn_mask_0, attn_mask_1

    def _apply(self, fn, recurse=True):
        self.border_masks.clear()
        return super()._apply(fn, recurse)

    def border_window_masks(self, H, W, dtype, device):
        """border_window_mask of both branches, kept for the last few sizes."""
        key = (H, W, dtype, device)
        masks = self.border_masks.get(key)
        if masks is None:
            split, shift = self.split_size, self.shift_size
            masks = (
                border_window_mask(H, W, split[0], split[1], shift[0], shift[1], dtype, device),
                border_window_mask(H, W, split[1], split[0], shift[1], shift[0], dtype, device),
            )
            if not (self.training or torch.is_grad_enabled()):
                if len(self.border_masks) >= 4:
                    self.border_masks.clear()
                self.border_masks[key] = masks
        return masks

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        if _traced():
            return self.forward_traced(x, H, W)
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # heads padded to a multiple of 8 for the fused attention kernels
        d = C // self.num_heads
        width = d + -d % 8
        # an export's copy has them padded in qkv already
        given = self.qkv.weight.shape[0] // (3 * self.num_heads)
        if given != d or width == d or self.training or torch.is_grad_enabled():
            qkv = pad_heads(self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1))
        else:
            weight, bias = self.keep(
                "qkv", lambda: _padded_rows(self.qkv, self.num_heads, d, width)
            )
            qkv = F.linear(x, weight, bias).reshape(B, H, W, 3, self.num_heads, width)
        # V without partition, channels last
        v = qkv[:, :, :, 2, :, :d].flatten(3).permute(0, 3, 1, 2)
        padded = self.num_heads * width
        qkv = qkv.reshape(B, H, W, 3, padded).permute(3, 0, 1, 2, 4)  # 3, B, H, W, C

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size
        if pad_r or pad_b:
            qkv = F.pad(qkv, (0, 0, 0, pad_r, 0, pad_b))  # 3, B, _H, _W, C
        _H = pad_b + H
        _W = pad_r + W

        half = padded // 2
        shifted = (
            self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0
        ) or (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0)
        if shifted and _H > max_split_size and _W > max_split_size:
            split, shift = self.split_size, self.shift_size
            mask_0, mask_1 = self.border_window_masks(_H, _W, x.dtype, x.device)
            x1 = self.attns[0].forward_shifted(
                qkv[..., :half], _H, _W, shift[0], shift[1], mask_0
            )
            x2 = self.attns[1].forward_shifted(
                qkv[..., half:], _H, _W, shift[1], shift[0], mask_1
            )
        elif shifted:
            mask_0, mask_1 = self.calculate_mask(_H, _W, dtype=x.dtype, device=x.device)
            x1 = torch.roll(
                self.attns[0](
                    torch.roll(
                        qkv[..., :half],
                        shifts=(-self.shift_size[0], -self.shift_size[1]),
                        dims=(2, 3),
                    ),
                    _H,
                    _W,
                    mask=mask_0,
                ),
                shifts=(self.shift_size[0], self.shift_size[1]),
                dims=(1, 2),
            )
            x2 = torch.roll(
                self.attns[1](
                    torch.roll(
                        qkv[..., half:],
                        shifts=(-self.shift_size[1], -self.shift_size[0]),
                        dims=(2, 3),
                    ),
                    _H,
                    _W,
                    mask=mask_1,
                ),
                shifts=(self.shift_size[1], self.shift_size[0]),
                dims=(1, 2),
            )
        else:
            x1 = self.attns[0](qkv[..., :half], _H, _W)
            x2 = self.attns[1](qkv[..., half:], _H, _W)
        # each branch's own channels, one pass: B, H, W, C
        heads = self.num_heads // 2
        attened_x = torch.cat(
            [
                x1[:, :H, :W].reshape(B, H, W, heads, -1)[..., :d].flatten(3),
                x2[:, :H, :W].reshape(B, H, W, heads, -1)[..., :d].flatten(3),
            ],
            dim=3,
        )

        # convolution output, channels last
        conv_x = _dwconv(self, v)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = _channel_map(self, conv_x).permute(0, 2, 3, 1)
        # S-Map (before sigmoid)
        spatial_map = _spatial_map(self, attened_x.permute(0, 3, 1, 2))

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = (torch.sigmoid(spatial_map) * conv_x).permute(0, 2, 3, 1)

        x = _sum(conv_x, attened_x).flatten(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward_traced(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # Heads as wide as qkv makes them (an export may pad them); each branch has
        # half of them
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1)
        d = C // self.num_heads
        # V without partition: its C channels
        v = qkv[:, :, :, 2, :, :d].permute(0, 3, 4, 1, 2).reshape(B, C, H, W)
        # heads padded to a multiple of 8 for the fused attention kernels
        qkv = pad_heads(qkv)
        width = qkv.shape[-1]
        C = self.num_heads * width
        qkv = qkv.reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)  # 3, B, H, W, C

        # image padding, channels last (no round trip through B, C, H, W)
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = F.pad(qkv, (0, 0, 0, pad_r, 0, pad_b))  # 3, B, _H, _W, C
        _H = pad_b + H
        _W = pad_r + W

        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            qkv_0 = torch.roll(
                qkv[:, :, :, :, : C // 2],
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),
            )
            qkv_1 = torch.roll(
                qkv[:, :, :, :, C // 2 :],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),
            )

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W, dtype=x.dtype, device=x.device)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0])
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1])
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(
                x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
            x2 = torch.roll(
                x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)
            )
            # attention output
            attened_x = torch.cat([x1[:, :H, :W, :], x2[:, :H, :W, :]], dim=3)

        else:
            x1 = self.attns[0](qkv[..., : C // 2], _H, _W)[:, :H, :W, :]
            x2 = self.attns[1](qkv[..., C // 2 :], _H, _W)[:, :H, :W, :]
            # attention output
            attened_x = torch.cat([x1, x2], dim=3)

        # the heads' own channels: B, L, C
        attened_x = attened_x.reshape(B, H, W, self.num_heads, width)
        C = self.num_heads * d
        attened_x = attened_x[..., :d].reshape(B, L, C)

        # convolution output
        conv_x = self.dwconv(v)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = (
            self.channel_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, 1, C)
        )
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Adaptive_Channel_Attention(_Derived):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        if _traced():
            return self.forward_traced(x, H, W)
        B, N, C = x.shape
        tokens = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = tokens.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = _row_gram(_unit_length(q), _unit_length(k), H, W) * self.temperature
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (attn @ v^T)^T in token order: as a view of the channel-major product, the adds
        # below would read it transposed, at a cost that grows faster than the image
        attened_x = (v @ attn.transpose(-2, -1)).transpose(1, 2).reshape(B, N, C)

        # channels last from here: the convolutions read and write the tokens where they are
        conv_x = _dwconv(self, tokens[:, :, 2].reshape(B, H, W, C).permute(0, 3, 1, 2))
        channel_map = _channel_map(self, attened_x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        spatial_map = _spatial_map(self, conv_x).permute(0, 2, 3, 1).flatten(1, 2)

        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = (conv_x * torch.sigmoid(channel_map)).permute(0, 2, 3, 1).flatten(1, 2)

        x = self.proj(_sum(conv_x, attened_x))
        x = self.proj_drop(x)

        return x

    def forward_traced(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        # Each norm sums the squares of every token (65,536 at 256x256), which fp16
        # overflows when an exported graph computes it in fp16; like PyTorch's own
        # kernels, normalize in fp32, then cast back
        half = q.dtype == torch.float16
        if half:
            q, k = q.float(), k.float()
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        if half:
            q, k = q.half(), k.half()

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        if in_migraphx_export():
            # transposed, which MIGraphX does not fuse into one attention kernel
            attened_x = v.transpose(-2, -1) @ attn.transpose(-2, -1)
            attened_x = attened_x.permute(0, 2, 1, 3).reshape(B, N, C)
        else:
            attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = (
            self.spatial_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, N, 1)
        )

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DATB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[2, 4],
        shift_size=[1, 2],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        if b_idx % 2 == 0:
            # DSTB
            self.attn = Adaptive_Spatial_Attention(
                dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                rg_idx=rg_idx,
                b_idx=b_idx,
            )
        else:
            # DCTB
            self.attn = Adaptive_Channel_Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        if _traced():
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
            return x
        x = _sum(x, self.drop_path(self.attn(self.norm1(x), H, W)))
        return _sum(x, self.drop_path(self.ffn(self.norm2(x), H, W)))


class ResidualGroup(nn.Module):
    """ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=[2, 4],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_paths=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=2,
        use_chk=False,
        resi_connection="1conv",
        rg_idx=0,
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList(
            [
                DATB(
                    dim=dim,
                    num_heads=num_heads,
                    reso=reso,
                    split_size=split_size,
                    shift_size=[split_size[0] // 2, split_size[1] // 2],
                    expansion_factor=expansion_factor,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_paths[i],  # type: ignore
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    rg_idx=rg_idx,
                    b_idx=i,
                )
                for i in range(depth)
            ]
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if _traced():
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
            x = self.conv(x)
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            # channels last: the convolution reads and writes the tokens where they are
            B, _, C = x.shape
            x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
            return _sum(res, x.permute(0, 2, 3, 1).flatten(1, 2))
        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super().__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class DAT(nn.Module):
    """Dual Aggregation Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DATB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        use_chk (bool): Whether to use checkpointing to save memory.
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """
    def __init__(
        self,
        *,
        img_size=64,
        in_chans=3,
        embed_dim=180,
        split_size=[2, 4],
        depth=[2, 2, 2, 2],
        num_heads=[2, 2, 2, 2],
        expansion_factor=4.0,
        qkv_bias=True,
        qk_scale: float | None = None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_chk=False,
        upscale=2,
        img_range=1.0,
        resi_connection="1conv",
        upsampler="pixelshuffle",
    ):
        super().__init__()
        # spandrel: the trunk's LayerNorms run spandrel's kernel in eager CUDA inference
        # (KernelLayerNorm: nn.LayerNorm's parameters, keys and exports)
        if norm_layer is nn.LayerNorm:
            norm_layer = KernelLayerNorm

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rg_idx=i,
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, Reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, (img_size, img_size)
            )

        self.apply(self._init_weights)

        # One store of the wrapping windows' masks for all the model's blocks
        border_masks = {}
        for module in self.modules():
            if isinstance(module, Adaptive_Spatial_Attention):
                module.border_masks = border_masks

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:  # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for image SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        x = x / self.img_range + self.mean
        return x
