// tests/test_linear_int8_mpsmm_parity.cpp
#include <gtest/gtest.h>

#include <axiom/axiom.hpp>
#include <axiom/nn/linear.hpp>
#include <axiom/operations.hpp>
#include <axiom/system.hpp>
#include <axiom/tensor.hpp>

#include <cmath>
#include <cstdlib>
#include <map>
#include <string>

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::Tensor;

// L2 integration test for WAS-27. Verifies nn::Linear::forward() with int8
// weights produces matching output across the new dequant+MPSMM fast path
// and the legacy lazy path by flipping AXIOM_FORCE_LAZY_LINEAR between calls.
//
// Builds on the axiom L1 parity test (test_int8_matmul_mpsmm_parity) by
// exercising dispatch through the public nn::Linear API rather than via direct
// ops::int8_matmul calls. Exercises the FFN-expand shape (1900x1024x4096),
// the QKVO shape (1900x1024x1024), and the FFN-contract shape (1900x4096x1024)
// from Parakeet TDT-0.6b-v3.

// RAII env var guard — ensures AXIOM_FORCE_LAZY_LINEAR is cleared on scope
// exit even if forward() throws. Without this, a regression that crashes
// during the lazy-path forward() would leak the env var into subsequent tests
// in the same binary.
struct ScopedLazyEnv {
    explicit ScopedLazyEnv(bool on) {
        if (on) setenv("AXIOM_FORCE_LAZY_LINEAR", "1", 1);
        else    unsetenv("AXIOM_FORCE_LAZY_LINEAR");
    }
    ~ScopedLazyEnv() { unsetenv("AXIOM_FORCE_LAZY_LINEAR"); }
};

// Block-symmetric int8 quantization, block=32 along K dim (Phase 1 scheme).
// Returns {weight_int8 [N,K] int8, scale_fp16 [N, K/32] fp16}.
//
// NOTE: duplicated from axiom's L1 test helper. Update both copies in tandem
// if the Phase 1 quant scheme (block size, clamp range) ever changes.
// TODO(M4): hoist to axiom test utils when the post-WAS-27 cleanup PR lands.
struct QuantPair {
    Tensor weight_int8;
    Tensor scale_fp16;
};

QuantPair quantize_block_symmetric_k32(const Tensor &w_fp32) {
    // w_fp32: [N, K], block=32 along K
    // For each row n and each block b: scale = max(|w[n, b*32 : (b+1)*32]|) / 127
    //                                  w_int8[n, k] = round(w_fp32[n, k] / scale[n, b])
    auto w_cpu = w_fp32.to(Device::CPU).astype(DType::Float32);
    const auto *w_data = w_cpu.typed_data<float>();
    size_t N = w_cpu.shape()[0];
    size_t K = w_cpu.shape()[1];
    constexpr size_t kBlock = 32;
    size_t num_blocks = K / kBlock;

    auto w_int8_cpu = Tensor::zeros({N, K}, DType::Int8, Device::CPU);
    auto scale_cpu = Tensor::zeros({N, num_blocks}, DType::Float32, Device::CPU);
    auto *w_int8_data = w_int8_cpu.typed_data<int8_t>();
    auto *scale_data = scale_cpu.typed_data<float>();

    for (size_t n = 0; n < N; ++n) {
        for (size_t b = 0; b < num_blocks; ++b) {
            float max_abs = 0.0f;
            for (size_t k = 0; k < kBlock; ++k) {
                float v = std::fabs(w_data[n * K + b * kBlock + k]);
                if (v > max_abs)
                    max_abs = v;
            }
            float s = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            scale_data[n * num_blocks + b] = s;
            for (size_t k = 0; k < kBlock; ++k) {
                float v = w_data[n * K + b * kBlock + k] / s;
                int q = static_cast<int>(std::round(v));
                if (q < -128)
                    q = -128;
                if (q > 127)
                    q = 127;
                w_int8_data[n * K + b * kBlock + k] = static_cast<int8_t>(q);
            }
        }
    }

    QuantPair out;
    out.weight_int8 = w_int8_cpu.to(Device::GPU);
    out.scale_fp16 = scale_cpu.astype(DType::Float16).to(Device::GPU);
    return out;
}

float max_relative_error_fp16(const Tensor &a, const Tensor &b) {
    auto a_cpu = a.to(Device::CPU).astype(DType::Float32);
    auto b_cpu = b.to(Device::CPU).astype(DType::Float32);
    const auto *a_data = a_cpu.typed_data<float>();
    const auto *b_data = b_cpu.typed_data<float>();
    size_t n = a_cpu.numel();
    float max_rel = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float denom = std::max(std::abs(b_data[i]), 1e-3f);
        float rel = std::abs(a_data[i] - b_data[i]) / denom;
        if (rel > max_rel)
            max_rel = rel;
    }
    return max_rel;
}

// ─── Parameterized fixture ────────────────────────────────────────────────────

struct LinearParityParams {
    size_t M;
    size_t K;
    size_t N;
    const char *shape_name; // used in INSTANTIATE_TEST_SUITE_P naming
};

class LinearInt8MpsmmParityP
    : public ::testing::TestWithParam<LinearParityParams> {
  protected:
    void SetUp() override {
        if (!axiom::system::should_run_gpu_tests()) {
            GTEST_SKIP() << "Requires Metal GPU";
        }
    }
};

TEST_P(LinearInt8MpsmmParityP, ParityAndBiasSanity) {
    const auto &p = GetParam();
    const size_t M = p.M;
    const size_t K = p.K;
    const size_t N = p.N;

    auto input = Tensor::randn({M, K}, DType::Float32, Device::CPU)
                     .astype(DType::Float16)
                     .to(Device::GPU);

    auto w_fp32 = Tensor::randn({N, K}, DType::Float32, Device::CPU);
    auto qp = quantize_block_symmetric_k32(w_fp32);

    auto bias = Tensor::randn({N}, DType::Float32, Device::CPU)
                    .astype(DType::Float16)
                    .to(Device::GPU);

    axiom::nn::Linear linear(/*bias=*/true);
    linear.load_int8_weights(qp.weight_int8, qp.scale_fp16);
    std::map<std::string, Tensor> state;
    state["bias"] = bias;
    linear.load_state_dict(state, "", /*strict=*/false);

    Tensor out_fast, out_lazy;
    {
        ScopedLazyEnv guard(false);
        out_fast = linear.forward(input);
    }
    {
        ScopedLazyEnv guard(true);
        out_lazy = linear.forward(input);
    }

    EXPECT_EQ(out_fast.shape(), out_lazy.shape());
    float rel_err = max_relative_error_fp16(out_fast, out_lazy);
    EXPECT_LE(rel_err, 1e-3f)
        << p.shape_name << " parity failed: max relative error " << rel_err;

    // Sanity check: bias was actually applied (catches a future regression
    // that drops the bias-add from Linear::forward — would still pass parity
    // since both fast and lazy paths would be equally bias-less).
    axiom::nn::Linear linear_nobias(/*bias=*/false);
    linear_nobias.load_int8_weights(qp.weight_int8, qp.scale_fp16);
    Tensor out_nobias;
    {
        ScopedLazyEnv guard(false);
        out_nobias = linear_nobias.forward(input);
    }
    float diff = max_relative_error_fp16(out_fast, out_nobias);
    EXPECT_GT(diff, 1e-3f) << "bias appears to have been dropped: out_fast and "
                               "out_nobias indistinguishable within 1e-3";
}

INSTANTIATE_TEST_SUITE_P(
    LinearInt8MpsmParity,
    LinearInt8MpsmmParityP,
    ::testing::Values(
        // QKVO shape from Parakeet TDT-0.6b-v3 (hidden=1024).
        LinearParityParams{1900, 1024, 1024, "QKVO"},
        // FFN expand shape from Parakeet TDT-0.6b-v3 (hidden=1024, ffn=4096).
        LinearParityParams{1900, 1024, 4096, "FFNExpand"},
        // FFN contract shape from Parakeet TDT-0.6b-v3 (hidden=1024, ffn=4096).
        LinearParityParams{1900, 4096, 1024, "FFNContract"}),
    [](const ::testing::TestParamInfo<LinearParityParams> &info) {
        return info.param.shape_name;
    });

} // namespace
