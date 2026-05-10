import subprocess
import sys
from pathlib import Path

import numpy as np
import safetensors.numpy

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quantize_int8 import quantize_block_sym

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "quantize_int8.py"


def test_round_trip_dequant_within_tolerance():
    """quantize(W) -> dequant(int8 * scale) ~= W within scheme-B tolerance."""
    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.1, size=(64, 256)).astype(np.float32)

    W_int8, scales = quantize_block_sym(W, block_size=32)

    # Dequantize.
    K_blocks = 256 // 32
    W_int8_blocks = W_int8.reshape(64, K_blocks, 32).astype(np.float32)
    scales_fp32 = scales.astype(np.float32)
    W_dequant = (W_int8_blocks * scales_fp32[:, :, None]).reshape(64, 256)

    err = np.abs(W - W_dequant)
    # Block-32 sym quant: max err per element should be <= scale/2 per block.
    # Mean err on Gaussian inputs typically ~0.001-0.003.
    assert err.mean() < 5.0e-3, f"mean abs err {err.mean()} too high"
    assert err.max() < 5.0e-2, f"max abs err {err.max()} too high"


def test_quantize_handles_zero_block():
    """All-zeros block must not divide-by-zero or produce NaN."""
    W = np.zeros((4, 32), dtype=np.float32)
    W_int8, scales = quantize_block_sym(W, block_size=32)
    assert np.all(W_int8 == 0)
    assert np.all(np.isfinite(scales))


def test_int8_range_clipped():
    """Outliers in W must clip cleanly to [-128, 127]."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, size=(4, 32)).astype(np.float32)
    W[0, 0] = 1.0e10  # absurd outlier
    W_int8, _ = quantize_block_sym(W, block_size=32)
    assert W_int8.min() >= -128 and W_int8.max() <= 127


def _write_wrong_naming_checkpoint(path: Path) -> None:
    """Write a safetensors file whose key naming does NOT match QUANTIZABLE_PATTERNS.

    Uses PyTorch-export-style keys (e.g. `attention.q.weight`) instead of the
    axiom-style keys (`mha_.q_proj.weight`) that the script targets. Result:
    the quantizable predicate misses every matmul weight and the script
    completes with quant_count == 0.
    """
    tensors = {
        # Wrong naming convention — none of these match QUANTIZABLE_PATTERNS.
        "attention.q.weight": np.zeros((64, 64), dtype=np.float32),
        "attention.k.weight": np.zeros((64, 64), dtype=np.float32),
        "attention.v.weight": np.zeros((64, 64), dtype=np.float32),
        "attention.out.weight": np.zeros((64, 64), dtype=np.float32),
        "ffn.fc1.weight": np.zeros((256, 64), dtype=np.float32),
        "ffn.fc2.weight": np.zeros((64, 256), dtype=np.float32),
    }
    safetensors.numpy.save_file(tensors, str(path))


def test_strict_flag_errors_on_zero_quantized(tmp_path):
    """--strict must exit non-zero if no matmul weights were quantized."""
    in_path = tmp_path / "wrong-naming.safetensors"
    out_path = tmp_path / "wrong-naming-int8.safetensors"
    _write_wrong_naming_checkpoint(in_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--strict",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"expected non-zero exit under --strict with 0 quantized; "
        f"got rc={result.returncode}\nstdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
    assert "0 weights" in result.stderr or "quantized" in result.stderr, (
        f"expected diagnostic stderr; got: {result.stderr}"
    )
    # Output file must NOT have been written when --strict errors out.
    assert not out_path.exists(), "output file should not be written on --strict error"


def test_strict_flag_default_off_still_passes_through(tmp_path):
    """Without --strict, a zero-quant pass-through still writes the output."""
    in_path = tmp_path / "wrong-naming.safetensors"
    out_path = tmp_path / "wrong-naming-passthrough.safetensors"
    _write_wrong_naming_checkpoint(in_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--in",
            str(in_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"default mode must NOT fail on zero-quant; got rc={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert out_path.exists(), "output file should be written without --strict"
