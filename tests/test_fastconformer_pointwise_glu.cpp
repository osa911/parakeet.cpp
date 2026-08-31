#include <gtest/gtest.h>

#include <axiom/axiom.hpp>
#include <axiom/operations.hpp>
#include <axiom/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::Tensor;

constexpr size_t kBatch = 1;
constexpr size_t kChannels = 1024;
constexpr size_t kTime = 138;

Tensor deterministic_float_tensor(const Shape &shape, float amplitude) {
    Tensor tensor = Tensor::zeros(shape, DType::Float32, Device::CPU);
    auto *data = tensor.typed_data<float>();
    for (size_t index = 0; index < tensor.numel(); ++index) {
        const float phase = static_cast<float>((index * 17 + 11) % 257);
        data[index] = amplitude * std::sin(phase * 0.071f);
    }
    return tensor;
}

float max_abs_error(const Tensor &actual, const Tensor &expected) {
    const Tensor actual_cpu = actual.to(Device::CPU).astype(DType::Float32);
    const Tensor expected_cpu = expected.to(Device::CPU).astype(DType::Float32);
    const auto *actual_data = actual_cpu.typed_data<float>();
    const auto *expected_data = expected_cpu.typed_data<float>();
    float error = 0.0f;
    for (size_t index = 0; index < actual_cpu.numel(); ++index) {
        error = std::max(error,
                         std::abs(actual_data[index] - expected_data[index]));
    }
    return error;
}

TEST(FastConformerPointwiseGlu, MatchesGenericInt8ProjectionWithAndWithoutBias) {
    const Tensor input =
        deterministic_float_tensor({kBatch, kTime, kChannels}, 0.75f)
            .astype(DType::Float16)
            .to(Device::GPU);

    Tensor weight =
        Tensor::zeros({kChannels * 2, kChannels}, DType::Int8, Device::CPU);
    auto *weight_data = weight.typed_data<int8_t>();
    for (size_t index = 0; index < weight.numel(); ++index) {
        weight_data[index] = static_cast<int8_t>((index * 13 + 7) % 15 - 7);
    }
    weight = weight.to(Device::GPU);
    const Tensor scale =
        deterministic_float_tensor({kChannels * 2, kChannels / 32}, 0.003f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor bias = deterministic_float_tensor({kChannels * 2}, 0.1f)
                            .astype(DType::Float16)
                            .to(Device::GPU);

    const auto expect_matches_generic = [&](const Tensor &optional_bias) {
        const Tensor projected = optional_bias.storage()
                                     ? axiom::ops::int8_matmul_bias(
                                           input, weight, scale, optional_bias)
                                     : axiom::ops::int8_matmul(input, weight,
                                                                scale);
        const Tensor expected =
            axiom::ops::glu(projected.permute({0, 2, 1}), /*dim=*/1)
                .ascontiguousarray();
        const Tensor actual = axiom::ops::fastconformer_int8_pointwise_glu_f16(
            input, weight, scale, optional_bias);

        EXPECT_EQ(actual.shape(), Shape({kBatch, kChannels, kTime}));
        EXPECT_TRUE(actual.is_contiguous());
        EXPECT_LE(max_abs_error(actual, expected), 0.003f);
    };

    expect_matches_generic(bias);
    expect_matches_generic(Tensor());
}

TEST(FastConformerPointwiseGlu, DirectGluMatchesGenericF16Glu) {
    const Tensor time_major =
        deterministic_float_tensor({kBatch, kTime, kChannels * 2}, 0.75f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor input = time_major.permute({0, 2, 1});
    const Tensor expected = axiom::ops::glu(input, /*dim=*/1).ascontiguousarray();
    const Tensor actual = axiom::ops::fastconformer_glu_f16(input);

    EXPECT_EQ(actual.shape(), Shape({kBatch, kChannels, kTime}));
    EXPECT_TRUE(actual.is_contiguous());
    EXPECT_LE(max_abs_error(actual, expected), 0.0f);
}

TEST(FastConformerPointwiseGlu, DirectChannelsFirstGluMatchesGenericF16Glu) {
    const Tensor input =
        deterministic_float_tensor({kBatch, kChannels * 2, kTime}, 0.75f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor expected = axiom::ops::glu(input, /*dim=*/1).ascontiguousarray();
    const Tensor actual =
        axiom::ops::fastconformer_glu_channels_first_f16(input);

    EXPECT_EQ(actual.shape(), Shape({kBatch, kChannels, kTime}));
    EXPECT_TRUE(actual.is_contiguous());
    EXPECT_LE(max_abs_error(actual, expected), 0.0f);
}

TEST(FastConformerPointwiseGlu,
     DirectChannelsFirstGluPrecisionVariantsStayWithinObservedEnvelope) {
    // The low-amplitude route test above is exact. Real pointwise projections
    // can be much larger, so keep a separate diagnostic for the default Metal
    // exp implementation before it can be considered as an MPSGraph match.
    const Tensor input =
        deterministic_float_tensor({kBatch, kChannels * 2, kTime}, 32.0f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor expected = axiom::ops::glu(input, /*dim=*/1).ascontiguousarray();
    const Tensor precise =
        axiom::ops::fastconformer_glu_channels_first_f16(input);
    const Tensor default_exp =
        axiom::ops::fastconformer_glu_channels_first_f16_default_exp(input);
    const Tensor rounded_sigmoid =
        axiom::ops::fastconformer_glu_channels_first_f16_rounded_sigmoid(input);

    const float precise_error = max_abs_error(precise, expected);
    const float default_exp_error = max_abs_error(default_exp, expected);
    const float rounded_sigmoid_error = max_abs_error(rounded_sigmoid, expected);
    std::cout << "direct_channels_first_glu model-magnitude errors precise="
              << precise_error << " default_exp=" << default_exp_error
              << " rounded_sigmoid=" << rounded_sigmoid_error << '\n';
    EXPECT_LE(precise_error, 0.0078125f);
    EXPECT_LE(default_exp_error, 0.0078125f);
    EXPECT_LE(rounded_sigmoid_error, 0.015625f);
}

TEST(FastConformerPointwiseGlu, DirectFp16ProjectionMatchesGenericBeforeGlu) {
    const Tensor input =
        deterministic_float_tensor({kBatch, kTime, kChannels}, 0.75f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor weight =
        deterministic_float_tensor({kChannels * 2, kChannels}, 0.04f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor bias = deterministic_float_tensor({kChannels * 2}, 0.1f)
                            .astype(DType::Float16)
                            .to(Device::GPU);

    const Tensor expected =
        axiom::ops::add(axiom::ops::matmul(input, weight, false, true), bias);
    const Tensor actual = axiom::ops::fastconformer_f16_pointwise_f16(
        input, weight, bias);

    EXPECT_EQ(actual.shape(), Shape({kBatch, kTime, kChannels * 2}));
    EXPECT_TRUE(actual.is_contiguous());
    // This test exposes the projection output before GLU so the encoder
    // experiment can attribute any later transcript drift to the correct
    // operation boundary rather than guessing from the fused result.
    EXPECT_LE(max_abs_error(actual, expected), 0.01f);
}

TEST(FastConformerPointwiseGlu, MatchesGenericFp16ProjectionWithAndWithoutBias) {
    const Tensor input =
        deterministic_float_tensor({kBatch, kTime, kChannels}, 0.75f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor weight =
        deterministic_float_tensor({kChannels * 2, kChannels}, 0.04f)
            .astype(DType::Float16)
            .to(Device::GPU);
    const Tensor bias = deterministic_float_tensor({kChannels * 2}, 0.1f)
                            .astype(DType::Float16)
                            .to(Device::GPU);

    const auto expect_matches_generic = [&](const Tensor &optional_bias) {
        Tensor projected = axiom::ops::matmul(input, weight, false, true);
        if (optional_bias.storage()) {
            projected = axiom::ops::add(projected, optional_bias);
        }
        const Tensor expected =
            axiom::ops::glu(projected.permute({0, 2, 1}), /*dim=*/1)
                .ascontiguousarray();
        const Tensor actual = axiom::ops::fastconformer_f16_pointwise_glu_f16(
            input, weight, optional_bias);

        EXPECT_EQ(actual.shape(), Shape({kBatch, kChannels, kTime}));
        EXPECT_TRUE(actual.is_contiguous());
        // simdgroup matrix accumulation has a different reduction order from
        // the generic MPSGraph convolution. The observed Float16 rounding
        // delta is below four FP16 ULPs; this bound still catches a layout or
        // indexing error by several orders of magnitude.
        EXPECT_LE(max_abs_error(actual, expected), 0.01f);
    };

    expect_matches_generic(bias);
    expect_matches_generic(Tensor());
}

} // namespace
