import numpy as np
import sys
from pathlib import Path

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from quantize_int8 import quantize_block_sym


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
