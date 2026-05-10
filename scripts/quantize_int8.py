#!/usr/bin/env python3
"""
quantize_int8.py — Convert axiom-format fp32 safetensors to int8 + fp16 scales.

Scheme B: block-wise symmetric quantization, block_size=32 along K (last dim).
For each weight matrix W with shape [N, K]:
  scale[n, kb] = max(abs(W[n, kb*32 : (kb+1)*32])) / 127.0     (fp16, per block)
  W_int8[n, k] = round(W[n, k] / scale[n, k // 32]).clip(-128, 127)

Output safetensors layout per matmul weight named "<X>.weight":
  <X>_quantized   INT8    [N, K]
  <X>_scale       FLOAT16 [N, K // 32]

Other tensors (LayerNorm gains, biases, conv weights, etc.) are
copied through unchanged.

Usage:
  python3 scripts/quantize_int8.py \\
    --in encoder-axiom.safetensors \\
    --out encoder-axiom-int8.safetensors \\
    --block-size 32
"""

import argparse
from pathlib import Path

import numpy as np
import safetensors.numpy
from safetensors import safe_open


# Weight-tensor name patterns to quantize. Encoder-only matmul weights.
# LayerNorm + biases + conv subsampling stack stay fp32/fp16.
QUANTIZABLE_PATTERNS = (
    ".mha_.q_proj.weight",
    ".mha_.k_proj.weight",
    ".mha_.v_proj.weight",
    ".mha_.out_proj.weight",
    ".ffn1_.fc1_.weight",   # FFN expand
    ".ffn1_.fc2_.weight",   # FFN contract
    ".ffn2_.fc1_.weight",   # FFN expand
    ".ffn2_.fc2_.weight",   # FFN contract
)


def is_quantizable(name: str) -> bool:
    return any(p in name for p in QUANTIZABLE_PATTERNS)


def quantize_block_sym(W: np.ndarray, block_size: int = 32):
    """W: [N, K] fp32. Returns (W_int8 [N, K], scales [N, K // block_size] fp16)."""
    assert W.dtype == np.float32, f"expected fp32, got {W.dtype}"
    assert W.ndim == 2, f"expected 2D, got {W.shape}"
    N, K = W.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    K_blocks = K // block_size

    W_blocks = W.reshape(N, K_blocks, block_size)
    maxabs = np.maximum(np.abs(W_blocks).max(axis=-1), 1.0e-10)  # avoid /0
    scale = maxabs / 127.0  # fp32 scale per (n, kb)

    inv = 1.0 / scale
    W_int8 = np.round(W_blocks * inv[..., None]).clip(-128, 127).astype(np.int8)
    W_int8 = W_int8.reshape(N, K)

    scales_fp16 = scale.astype(np.float16)
    return W_int8, scales_fp16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    out_tensors = {}
    quant_count = 0
    passthrough_count = 0
    total_bytes_in = 0
    total_bytes_out = 0

    with safe_open(args.in_path, framework="numpy") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            total_bytes_in += t.nbytes

            if is_quantizable(name) and t.ndim == 2 and t.shape[1] % args.block_size == 0:
                t_fp32 = t.astype(np.float32)
                W_int8, scales_fp16 = quantize_block_sym(t_fp32, args.block_size)
                base = name[:-len(".weight")]  # strip ".weight"
                out_tensors[f"{base}_quantized"] = W_int8
                out_tensors[f"{base}_scale"] = scales_fp16
                total_bytes_out += W_int8.nbytes + scales_fp16.nbytes
                quant_count += 1
                print(f"  Q  {name}  {t.shape}  fp32 -> int8  "
                      f"({W_int8.nbytes / 1e6:.1f}MB+"
                      f"{scales_fp16.nbytes / 1e6:.1f}MB scales)")
            else:
                out_tensors[name] = t
                total_bytes_out += t.nbytes
                passthrough_count += 1

    print(f"\nQuantized: {quant_count} matmul weights")
    print(f"Pass-through: {passthrough_count} other tensors")
    if total_bytes_in > 0:
        print(f"Total: {total_bytes_in / 1e6:.0f} MB -> {total_bytes_out / 1e6:.0f} MB "
              f"({100 * (1 - total_bytes_out / total_bytes_in):.0f}% reduction)")

    safetensors.numpy.save_file(out_tensors, args.out_path)
    print(f"\nWrote {args.out_path}")


if __name__ == "__main__":
    main()
