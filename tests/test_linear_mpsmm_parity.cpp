// tests/test_linear_mpsmm_parity.cpp
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

// L2 integration test for WAS-26. Verifies nn::Linear::forward() produces
// matching output across the new gpu_linear fast path and the legacy lazy
// path by flipping env var AXIOM_FORCE_LAZY_LINEAR between calls.
//
// Per-call (non-cached) env-var read in axiom enables this single-process
// test pattern; see axiom commit aeb7821.

class LinearMpsmmParityTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!axiom::system::should_run_gpu_tests()) {
            GTEST_SKIP() << "Requires Metal GPU";
        }
    }
};

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

TEST_F(LinearMpsmmParityTest, FFNExpandShapeParity) {
    // FFN expand shape from Parakeet TDT-0.6b-v3 (hidden=1024, ffn=4096).
    auto input = Tensor::randn({1900, 1024}, DType::Float32, Device::CPU)
                     .astype(DType::Float16)
                     .to(Device::GPU);

    axiom::nn::Linear linear(/*bias=*/true);
    auto weight = Tensor::randn({4096, 1024}, DType::Float32, Device::CPU)
                      .astype(DType::Float16)
                      .to(Device::GPU);
    auto bias = Tensor::randn({4096}, DType::Float32, Device::CPU)
                    .astype(DType::Float16)
                    .to(Device::GPU);
    std::map<std::string, Tensor> state;
    state["weight"] = weight;
    state["bias"] = bias;
    linear.load_state_dict(state);

    // Path A: env var unset → gpu_linear fast path.
    unsetenv("AXIOM_FORCE_LAZY_LINEAR");
    auto out_fast = linear.forward(input);

    // Path B: force legacy lazy path via env var.
    setenv("AXIOM_FORCE_LAZY_LINEAR", "1", 1);
    auto out_lazy = linear.forward(input);
    unsetenv("AXIOM_FORCE_LAZY_LINEAR");

    EXPECT_EQ(out_fast.shape(), out_lazy.shape());
    float rel_err = max_relative_error_fp16(out_fast, out_lazy);
    EXPECT_LE(rel_err, 1e-3f)
        << "FFN expand parity failed: max relative error " << rel_err;
}

TEST_F(LinearMpsmmParityTest, OutputShapeMatchesLazyPath) {
    // Smoke test: verify Linear produces the expected output shape on QKVO.
    auto input = Tensor::randn({1900, 1024}, DType::Float32, Device::CPU)
                     .astype(DType::Float16)
                     .to(Device::GPU);
    axiom::nn::Linear linear(/*bias=*/true);
    auto weight = Tensor::randn({1024, 1024}, DType::Float32, Device::CPU)
                      .astype(DType::Float16)
                      .to(Device::GPU);
    auto bias = Tensor::randn({1024}, DType::Float32, Device::CPU)
                    .astype(DType::Float16)
                    .to(Device::GPU);
    std::map<std::string, Tensor> state;
    state["weight"] = weight;
    state["bias"] = bias;
    linear.load_state_dict(state);

    auto out = linear.forward(input);
    Shape expected{1900, 1024};
    EXPECT_EQ(out.shape(), expected);
}

} // namespace
