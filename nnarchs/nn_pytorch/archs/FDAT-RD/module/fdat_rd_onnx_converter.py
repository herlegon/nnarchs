#!/usr/bin/env python3
"""
FDAT-RD Model Converter
========================
Standalone converter for FDAT-RD (fdat_rd): FDAT + rectangular alternating
windows + token-dictionary cross-attention, dense spatial bias. The embedded
arch uses manual softmax attention (training arch uses SDPA) - identical
weights, cleanest export graph.

Converts: PyTorch (.pth/.pt), SafeTensors, ONNX (static / dynamic).

Usage:
    # 2x unshuffle, 720x540 input -> 1080p, static ONNX (H W = 540 720), verified:
    python fdat_rd_converter.py model.safetensors -f onnx-static --input-size 540 720 --split-size 10 30

    # tiny aligned verify first, then big export on GPU, no verify:
    python fdat_rd_converter.py model.safetensors -f onnx-static --input-size 60 60
    python fdat_rd_converter.py model.safetensors -f onnx-static --input-size 540 720 --device cuda --no-verify

    python fdat_rd_converter.py model.pth --info
    python fdat_rd_converter.py model.pth -f all

Notes:
    * --split-size carries the rectangular factorization (default 10 30); the
      spatial bias only stores the window product (s0*s1), validated.
    * scale is inferred for unshuffle models (in_ch 12 -> 2x, 48 -> 1x).
"""

import argparse
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fdat_rd import (
    SimplifiedResidualGroup,
    UniUpsampleV3,
)

# ============================================================================
# Standalone upsampler stack (verbatim; keys must match training)
# ============================================================================


# ============================================================================
# FDAT-RD architecture (MANUAL attention for clean export; keys match training)
# ============================================================================



class FDATRD(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, embed_dim=128, num_groups=4,
                 depth_per_group=2, num_heads=4, split_size=(10, 30), num_dict_tokens=128,
                 ffn_expansion_ratio=2.0, aim_reduction_ratio=8, group_block_pattern=None,
                 drop_path_rate=0.1, mid_dim=64, upsampler_type="transpose+conv",
                 img_range=1.0, unshuffle_mod=False, **kw):
        super().__init__()
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel", "dictionary"]
        self.upscale = scale
        align = math.lcm(int(split_size[0]), int(split_size[1]))
        self.pad = align
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            scale = 4
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(num_in_ch * unshuffle**2, embed_dim, 3, 1, 1, bias=True))
            self.pad = unshuffle * align
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, bias=True)
        ad = depth_per_group * len(group_block_pattern)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_groups * ad)]
        spatials_per_group = sum(
            1 for j in range(ad) if group_block_pattern[j % len(group_block_pattern)] == "spatial")
        self.groups = nn.Sequential(*[
            SimplifiedResidualGroup(
                embed_dim, ad, num_heads, split_size, ffn_expansion_ratio, aim_reduction_ratio,
                group_block_pattern, dpr[i * ad:(i + 1) * ad], num_dict_tokens,
                start_parity=(i * spatials_per_group) % 2)
            for i in range(num_groups)])
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(upsampler_type, scale, embed_dim, num_out_ch, mid_dim, 4)

    def check_img_size(self, x, h, w):
        if self.pad == 0:
            return x
        mph = (self.pad - h % self.pad) % self.pad
        mpw = (self.pad - w % self.pad) % self.pad
        mode = "replicate" if (mph >= h or mpw >= w) else "reflect"
        return F.pad(x, (0, mpw, 0, mph), mode)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.check_img_size(x, h, w)
        x_shallow = self.conv_first(x)
        x_deep = self.conv_after(self.groups(x_shallow))
        return self.upsampler(x_deep + x_shallow)[:, :, :h * self.upscale, :w * self.upscale]


# ============================================================================
# Detection / loading
# ============================================================================

def _block_type(sd, gi, bi):
    base = f"groups.{gi}.blocks.{bi}.attn."
    if base + "dictionary" in sd:
        return "dictionary"
    if base + "temp" in sd:
        return "channel"
    if base + "rel_pos_bias" in sd:
        return "spatial"
    return None


def detect_model_params(sd, split_size):
    p = dict(num_in_ch=3, num_out_ch=3, scale=4, embed_dim=128, num_groups=4, depth_per_group=2,
             num_heads=4, split_size=tuple(split_size), num_dict_tokens=128,
             ffn_expansion_ratio=2.0, aim_reduction_ratio=8,
             group_block_pattern=["spatial", "channel", "dictionary"], drop_path_rate=0.1,
             mid_dim=64, upsampler_type="transpose+conv", unshuffle_mod=False)

    if "conv_first.1.weight" in sd:
        p["embed_dim"] = sd["conv_first.1.weight"].shape[0]
        p["unshuffle_mod"] = True
        in_ch = sd["conv_first.1.weight"].shape[1]
        if in_ch == 12:
            p["scale"] = 2
        elif in_ch == 48:
            p["scale"] = 1
    elif "conv_first.weight" in sd:
        p["embed_dim"] = sd["conv_first.weight"].shape[0]
        p["num_in_ch"] = sd["conv_first.weight"].shape[1]

    gc = 0
    while f"groups.{gc}.conv.weight" in sd:
        gc += 1
    if gc:
        p["num_groups"] = gc

    types, bi = [], 0
    while f"groups.0.blocks.{bi}.n1.weight" in sd:
        types.append(_block_type(sd, 0, bi))
        bi += 1
    if types:
        n = len(types)
        period = n
        for c in range(1, n + 1):
            if n % c == 0 and all(types[i] == types[i % c] for i in range(n)):
                period = c
                break
        p["group_block_pattern"] = types[:period]
        p["depth_per_group"] = n // period

    s0, s1 = split_size
    for bi in range(len(types)):
        b = f"groups.0.blocks.{bi}.attn."
        if b + "temp" in sd:
            p["num_heads"] = sd[b + "temp"].shape[0]
            break
        if b + "rel_pos_bias" in sd:
            p["num_heads"] = sd[b + "rel_pos_bias"].shape[0]
            break
    for bi in range(len(types)):
        b = f"groups.0.blocks.{bi}.attn.rel_pos_bias"
        if b in sd:
            prod = sd[b].shape[1]
            if prod != s0 * s1:
                print(f"  WARNING: --split-size {split_size} product {s0*s1} "
                      f"!= detected window product {prod}. Fix --split-size.")
            break
    for bi in range(len(types)):
        b = f"groups.0.blocks.{bi}.attn.dictionary"
        if b in sd:
            p["num_dict_tokens"] = sd[b].shape[0]
            break

    if "groups.0.blocks.0.ffn.fc1.weight" in sd:
        w = sd["groups.0.blocks.0.ffn.fc1.weight"]
        r = w.shape[0] / w.shape[1]
        for c in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0):
            if abs(r - c) < 1e-3:
                r = c
                break
        p["ffn_expansion_ratio"] = r
    for bi in range(len(types)):
        k = f"groups.0.blocks.{bi}.inter.cg.1.weight"
        if k in sd:
            cg = sd[k]
            p["aim_reduction_ratio"] = max(1, cg.shape[1] // cg.shape[0])
            break

    if "upsampler.MetaUpsample" in sd:
        meta = sd["upsampler.MetaUpsample"]
        up = ["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv",
              "dysample", "transpose+conv", "lda", "pa_up"]
        p["upsampler_type"] = up[int(meta[1])]
        p["mid_dim"] = int(meta[5])
        if not p["unshuffle_mod"]:
            p["scale"] = int(meta[2])

    last = None
    for k in sd:
        if k.startswith("upsampler.") and k.endswith(".weight") and "MetaUpsample" not in k:
            last = k
    if last:
        p["num_out_ch"] = sd[last].shape[0]
    return p


def clean_state_dict(sd):
    out = OrderedDict()
    for k, v in sd.items():
        for pre in ["module.", "model.", "net.", "network.", "state_dict."]:
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def load_model(path, device, split_size, override=None):
    print(f"Loading checkpoint: {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ["params_ema", "params", "state_dict", "model", "net"]:
                if key in ckpt:
                    sd = ckpt[key]
                    break
            else:
                sd = ckpt
        else:
            sd = ckpt
    sd = clean_state_dict(sd)
    params = detect_model_params(sd, split_size)
    if override:
        params.update(override)
    print(f"  Detected: scale={params['scale']} embed_dim={params['embed_dim']} "
          f"groups={params['num_groups']} depth={params['depth_per_group']} "
          f"heads={params['num_heads']} split={params['split_size']} "
          f"dict_tokens={params['num_dict_tokens']} pattern={params['group_block_pattern']} "
          f"ffn={params['ffn_expansion_ratio']} up={params['upsampler_type']} "
          f"unshuffle={params['unshuffle_mod']}")
    model = FDATRD(**params)
    if hasattr(model.upsampler, "MetaUpsample"):
        sd["upsampler.MetaUpsample"] = model.upsampler.MetaUpsample
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, params


# ============================================================================
# Export
# ============================================================================



def print_model_info(model, params):
    total = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 60 + "\nMODEL INFORMATION  (FDAT-RD)\n" + "=" * 60)
    for k in ("scale", "num_in_ch", "num_out_ch", "embed_dim", "num_groups", "depth_per_group",
              "num_heads", "split_size", "num_dict_tokens", "group_block_pattern",
              "ffn_expansion_ratio", "aim_reduction_ratio", "upsampler_type", "unshuffle_mod"):
        print(f"  {k}: {params[k]}")
    print(f"  Total parameters: {total:,} ({total/1e6:.2f}M)\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description="FDAT-RD Model Converter",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("-f", "--formats", nargs="+", default=["fp32"],
                    choices=["fp32", "fp16", "safetensors", "onnx-static", "onnx-dynamic", "all"])
    ap.add_argument("--scale", type=int)
    ap.add_argument("--split-size", type=int, nargs=2, default=[10, 30])
    ap.add_argument("--num-dict-tokens", type=int)
    ap.add_argument("--upsampler")
    ap.add_argument("--unshuffle-mod", action="store_true")
    ap.add_argument("--input-size", type=int, nargs=2, default=[540, 720],
                    help="Static ONNX input H W; default 540 720 (720x540 input -> 1080p)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--name")
    args = ap.parse_args()

    override = {}
    if args.scale:
        override["scale"] = args.scale
    if args.split_size:
        override["split_size"] = tuple(args.split_size)
    if args.num_dict_tokens:
        override["num_dict_tokens"] = args.num_dict_tokens
    if args.upsampler:
        override["upsampler_type"] = args.upsampler
    if args.unshuffle_mod:
        override["unshuffle_mod"] = True

    model, params = load_model(args.input, args.device, args.split_size, override or None)
    model = model.to(args.device)
    print_model_info(model, params)
    if args.info:
        return


    print("\nExporting...")
    for fmt in args.formats:
        if fmt == "fp32":
            export_pytorch(model, str(out_dir / f"{base}_fp32.pth"))
        elif fmt == "fp16":
            export_pytorch(model, str(out_dir / f"{base}_fp16.pth"), fp16=True)
        elif fmt == "safetensors":
            export_safetensors(model, str(out_dir / f"{base}.safetensors"))
        elif fmt in ("onnx-static", "onnx-dynamic"):
            h, w = args.input_size
            isize = (1, params["num_in_ch"], h, w)
            dyn = fmt == "onnx-dynamic"
            path = str(out_dir / (f"{base}_dynamic.onnx" if dyn else f"{base}_static_{h}x{w}.onnx"))
            export_onnx(model, path, dynamic=dyn, input_size=isize, opset=args.opset)
            if not args.no_verify:
                verify_onnx(model, path, isize)
    print("\nDone!")


if __name__ == "__main__":
    main()
