#include <gtest/gtest.h>

#include <axiom/system.hpp>
#include <axiom/tensor.hpp>

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "parakeet/models/encoder.hpp"

namespace {

using axiom::DType;
using axiom::Device;
using axiom::Tensor;
using parakeet::models::ConformerAttention;

class ScopedEnvironmentVariable {
  public:
    ScopedEnvironmentVariable(const char *name, std::optional<std::string> value)
        : name_(name) {
        if (const char *old_value = std::getenv(name_.c_str())) {
            old_value_ = old_value;
        }
        if (value.has_value()) {
            setenv(name_.c_str(), value->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ~ScopedEnvironmentVariable() {
        if (old_value_.has_value()) {
            setenv(name_.c_str(), old_value_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) = delete;

  private:
    std::string name_;
    std::optional<std::string> old_value_;
};

constexpr int kNumHeads = 8;
constexpr size_t kHidden = 64;
constexpr size_t kTime = 8;

struct QuantizedProjection {
    Tensor weight;
    Tensor scale;
};

QuantizedProjection quantize_projection(const Tensor &weight) {
    constexpr size_t kBlock = 32;
    auto weight_cpu = weight.to(Device::CPU).astype(DType::Float32);
    const auto *source = weight_cpu.typed_data<float>();
    const size_t rows = weight_cpu.shape()[0];
    const size_t columns = weight_cpu.shape()[1];
    const size_t blocks = columns / kBlock;

    auto quantized = Tensor::zeros({rows, columns}, DType::Int8, Device::CPU);
    auto scales = Tensor::zeros({rows, blocks}, DType::Float32, Device::CPU);
    auto *quantized_data = quantized.typed_data<int8_t>();
    auto *scale_data = scales.typed_data<float>();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            float max_abs = 0.0f;
            for (size_t offset = 0; offset < kBlock; ++offset) {
                const size_t index = row * columns + block * kBlock + offset;
                max_abs = std::max(max_abs, std::abs(source[index]));
            }
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            scale_data[row * blocks + block] = scale;
            for (size_t offset = 0; offset < kBlock; ++offset) {
                const size_t index = row * columns + block * kBlock + offset;
                const int value = std::clamp(
                    static_cast<int>(std::round(source[index] / scale)),
                    -128, 127);
                quantized_data[index] = static_cast<int8_t>(value);
            }
        }
    }

    return {quantized, scales.astype(DType::Float16)};
}

void configure_attention(ConformerAttention &attention) {
    const Tensor q_weight =
        Tensor::randn({kHidden, kHidden}, DType::Float32) * 0.05f;
    const Tensor k_weight =
        Tensor::randn({kHidden, kHidden}, DType::Float32) * 0.05f;
    const Tensor v_weight =
        Tensor::randn({kHidden, kHidden}, DType::Float32) * 0.05f;
    const Tensor out_weight =
        Tensor::randn({kHidden, kHidden}, DType::Float32) * 0.05f;
    std::map<std::string, Tensor> state;
    state["norm_.weight"] = Tensor::ones({kHidden}, DType::Float32);
    state["norm_.bias"] = Tensor::zeros({kHidden}, DType::Float32);
    state["mha_.q_proj.weight"] = q_weight;
    state["mha_.k_proj.weight"] = k_weight;
    state["mha_.v_proj.weight"] = v_weight;
    state["mha_.out_proj.weight"] = out_weight;
    state["pos_proj_.weight"] = Tensor::zeros({kHidden, kHidden}, DType::Float32);
    state["pos_bias_u_"] =
        Tensor::zeros({kNumHeads, kHidden / kNumHeads}, DType::Float32);
    state["pos_bias_v_"] =
        Tensor::zeros({kNumHeads, kHidden / kNumHeads}, DType::Float32);
    attention.load_state_dict(state, "", /*strict=*/false);

    const auto q = quantize_projection(q_weight);
    const auto k = quantize_projection(k_weight);
    const auto v = quantize_projection(v_weight);
    const auto out = quantize_projection(out_weight);
    attention.load_int8_weights(q.weight, q.scale, k.weight, k.scale, v.weight,
                                v.scale, out.weight, out.scale);
    attention.to(DType::Float16);
    attention.to(Device::GPU);
}

float max_abs_error(const Tensor &actual, const Tensor &expected) {
    auto actual_cpu = actual.to(Device::CPU).astype(DType::Float32);
    auto expected_cpu = expected.to(Device::CPU).astype(DType::Float32);
    const auto *actual_data = actual_cpu.typed_data<float>();
    const auto *expected_data = expected_cpu.typed_data<float>();
    float result = 0.0f;
    for (size_t i = 0; i < actual_cpu.numel(); ++i) {
        result = std::max(result, std::abs(actual_data[i] - expected_data[i]));
    }
    return result;
}

struct AttentionInputs {
    Tensor input;
    Tensor pos_emb;
};

AttentionInputs make_inputs() {
    return {
        Tensor::randn({1, kTime, kHidden}, DType::Float32, Device::CPU)
            .astype(DType::Float16)
            .to(Device::GPU),
        Tensor::randn({2 * kTime - 1, kHidden}, DType::Float32, Device::CPU)
            .astype(DType::Float16)
            .to(Device::GPU),
    };
}

void expect_direct_route_parity(const std::optional<std::string> &head_layout) {
    ASSERT_TRUE(axiom::system::should_run_gpu_tests());
    ConformerAttention attention(kNumHeads, /*dropout=*/0.0f);
    configure_attention(attention);
    auto inputs = make_inputs();

    ScopedEnvironmentVariable generic_qkv("PARAKEET_DIRECT_INT8_QKV",
                                          std::string("0"));
    Tensor generic = attention.forward(inputs.input, inputs.pos_emb);
    EXPECT_GT(max_abs_error(generic, inputs.input), 1.0e-3f);

    {
        ScopedEnvironmentVariable direct_qkv("PARAKEET_DIRECT_INT8_QKV",
                                              std::nullopt);
        ScopedEnvironmentVariable direct_head_layout(
            "PARAKEET_DIRECT_INT8_QKV_HEAD_LAYOUT", head_layout);
        Tensor direct = attention.forward(inputs.input, inputs.pos_emb);
        EXPECT_LE(max_abs_error(generic, direct), 2.0e-3f);
    }
}

TEST(DirectInt8QkvRoute, PublishesEncoderRouteStatsContract) {
    parakeet::models::FastConformerEncoder encoder;
    const parakeet::models::EncoderRouteStats stats = encoder.route_stats();
    EXPECT_EQ(stats.direct_int8_qkv, 0U);
    EXPECT_EQ(stats.direct_int8_qkv_rejected, 0U);
    EXPECT_EQ(stats.direct_int8_qkv_head_layout, 0U);
}

TEST(DirectInt8QkvRoute, ForcedGenericMatchesSyntheticAttention) {
    if (!axiom::system::should_run_gpu_tests()) {
        GTEST_SKIP() << "Requires Metal GPU";
    }
    ConformerAttention attention(kNumHeads, /*dropout=*/0.0f);
    configure_attention(attention);
    auto inputs = make_inputs();
    ScopedEnvironmentVariable generic_qkv("PARAKEET_DIRECT_INT8_QKV",
                                          std::string("0"));
    const Tensor output = attention.forward(inputs.input, inputs.pos_emb);
    EXPECT_EQ(output.shape(), inputs.input.shape());
    EXPECT_GT(max_abs_error(output, inputs.input), 1.0e-3f);
}

TEST(DirectInt8QkvRoute, DefaultHeadLayoutMatchesGenericAttention) {
    expect_direct_route_parity(std::nullopt);
}

TEST(DirectInt8QkvRoute, ExplicitFlatLayoutMatchesGenericAttention) {
    expect_direct_route_parity(std::string("0"));
}

TEST(DirectInt8QkvRoute, IneligibleNonContiguousInputRetainsGenericBehavior) {
    if (!axiom::system::should_run_gpu_tests()) {
        GTEST_SKIP() << "Requires Metal GPU";
    }
    ConformerAttention attention(kNumHeads, /*dropout=*/0.0f);
    configure_attention(attention);
    Tensor non_contiguous =
        Tensor::randn({2, kTime, kHidden}, DType::Float32, Device::CPU)
            .astype(DType::Float16)
            .to(Device::GPU)
            .transpose({1, 0, 2});
    Tensor pos_emb =
        Tensor::randn({3, kHidden}, DType::Float32, Device::CPU)
            .astype(DType::Float16)
            .to(Device::GPU);
    ASSERT_FALSE(non_contiguous.is_contiguous());

    ScopedEnvironmentVariable generic_qkv("PARAKEET_DIRECT_INT8_QKV",
                                          std::string("0"));
    Tensor generic = attention.forward(non_contiguous, pos_emb);
    EXPECT_GT(max_abs_error(generic, non_contiguous), 1.0e-3f);
    {
        ScopedEnvironmentVariable direct_qkv("PARAKEET_DIRECT_INT8_QKV",
                                              std::nullopt);
        Tensor actual = attention.forward(non_contiguous, pos_emb);
        EXPECT_LE(max_abs_error(generic, actual), 2.0e-3f);
    }
}

} // namespace
