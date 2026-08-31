#include <gtest/gtest.h>

#include <axiom/axiom.hpp>
#include <axiom/error.hpp>
#include <axiom/nn/conv.hpp>
#include <axiom/operations.hpp>
#include <axiom/system.hpp>
#include <axiom/tensor.hpp>

#include "backends/metal/metal_common.hpp"
#include "backends/metal/metal_operations.hpp"
#include "backends/metal/metal_workspace_cache.hpp"
#include "parakeet/models/encoder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace axiom::ops {

std::array<Tensor, 3> int8_qkv_matmul_bias_head_layout(
    const Tensor &input, const Tensor &q_weight, const Tensor &q_scale,
    const Tensor &q_bias, const Tensor &k_weight, const Tensor &k_scale,
    const Tensor &k_bias, const Tensor &v_weight, const Tensor &v_scale,
    const Tensor &v_bias, size_t num_heads);

// Experimental internal primitive: emits one (mean, inverse-standard-
// deviation) pair per last-axis row. The production declaration is added only
// after this test demonstrates the missing behavior.
Tensor layer_norm_row_stats_f16(const Tensor &input, float eps);

} // namespace axiom::ops

namespace axiom::graph {
size_t gpu_graph_cache_max_size();
} // namespace axiom::graph

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::ShapeError;
using axiom::Tensor;

constexpr size_t kBatch = 1;
constexpr size_t kTime = 16;
constexpr size_t kHidden = 64;
constexpr size_t kIntermediate = 128;
constexpr size_t kBlock = 32;

struct QuantPair {
    Tensor weight;
    Tensor scale;
};

Tensor deterministic_tensor(const Shape &shape, float amplitude) {
    Tensor tensor = Tensor::zeros(shape, DType::Float32, Device::CPU);
    auto *data = tensor.typed_data<float>();
    for (size_t index = 0; index < tensor.numel(); ++index) {
        const float phase = static_cast<float>((index * 17 + 11) % 257);
        data[index] = amplitude * std::sin(phase * 0.071f);
    }
    return tensor;
}

QuantPair quantize_block_symmetric_k32(const Tensor &weight_fp32) {
    const auto weight = weight_fp32.to(Device::CPU).astype(DType::Float32);
    const size_t rows = weight.shape()[0];
    const size_t columns = weight.shape()[1];
    const size_t blocks = columns / kBlock;
    const auto *source = weight.typed_data<float>();

    Tensor quantized = Tensor::zeros({rows, columns}, DType::Int8, Device::CPU);
    Tensor scales = Tensor::zeros({rows, blocks}, DType::Float32, Device::CPU);
    auto *quantized_data = quantized.typed_data<int8_t>();
    auto *scale_data = scales.typed_data<float>();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            float max_abs = 0.0f;
            for (size_t column = 0; column < kBlock; ++column) {
                max_abs = std::max(
                    max_abs,
                    std::abs(source[row * columns + block * kBlock + column]));
            }
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            scale_data[row * blocks + block] = scale;
            for (size_t column = 0; column < kBlock; ++column) {
                const float value = source[row * columns + block * kBlock + column] / scale;
                const int rounded = std::clamp(
                    static_cast<int>(std::round(value)), -128, 127);
                quantized_data[row * columns + block * kBlock + column] =
                    static_cast<int8_t>(rounded);
            }
        }
    }

    return {quantized.to(Device::GPU), scales.astype(DType::Float16).to(Device::GPU)};
}

float max_abs_error(const Tensor &actual, const Tensor &expected) {
    const auto actual_cpu = actual.to(Device::CPU).astype(DType::Float32);
    const auto expected_cpu = expected.to(Device::CPU).astype(DType::Float32);
    const auto *actual_data = actual_cpu.typed_data<float>();
    const auto *expected_data = expected_cpu.typed_data<float>();
    float error = 0.0f;
    for (size_t index = 0; index < actual_cpu.numel(); ++index) {
        error = std::max(error, std::abs(actual_data[index] - expected_data[index]));
    }
    return error;
}

template <typename Invoke>
double median_gpu_milliseconds(Invoke invoke) {
    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    Tensor output;
    for (int iteration = 0; iteration < 7; ++iteration) {
        output = invoke();
        output.sync();
        stream.synchronize();
    }
    std::array<double, 31> samples{};
    for (double &sample : samples) {
        stream.synchronize();
        const auto started = std::chrono::steady_clock::now();
        output = invoke();
        output.sync();
        stream.synchronize();
        const auto finished = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::milli>(finished - started)
                     .count();
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// This is deliberately a plain CPU implementation rather than a composition
// of Axiom attention operations. It catches a wrong relative-position index,
// a dropped u/v bias, or a mask-broadcast regression in the fused GPU path.
Tensor relative_position_attention_reference(const Tensor &query,
                                             const Tensor &key,
                                             const Tensor &value,
                                             const Tensor &position,
                                             const Tensor &bias_u,
                                             const Tensor &bias_v,
                                             const Tensor &mask,
                                             float scale) {
    const Tensor q = query.to(Device::CPU).astype(DType::Float32);
    const Tensor k = key.to(Device::CPU).astype(DType::Float32);
    const Tensor v = value.to(Device::CPU).astype(DType::Float32);
    const Tensor p = position.to(Device::CPU).astype(DType::Float32);
    const Tensor u = bias_u.to(Device::CPU).astype(DType::Float32);
    const Tensor b = bias_v.to(Device::CPU).astype(DType::Float32);
    const Tensor m = mask.to(Device::CPU);

    const size_t batch = q.shape()[0];
    const size_t heads = q.shape()[1];
    const size_t time = q.shape()[2];
    const size_t head_dim = q.shape()[3];
    Tensor output = Tensor::zeros({batch, heads, time, head_dim},
                                  DType::Float32, Device::CPU);
    const auto *q_data = q.typed_data<float>();
    const auto *k_data = k.typed_data<float>();
    const auto *v_data = v.typed_data<float>();
    const auto *p_data = p.typed_data<float>();
    const auto *u_data = u.typed_data<float>();
    const auto *b_data = b.typed_data<float>();
    const auto *m_data = m.typed_data<uint8_t>();
    auto *out = output.typed_data<float>();

    const auto element = [heads, time, head_dim](size_t batch_index,
                                                 size_t head_index,
                                                 size_t time_index,
                                                 size_t dim) {
        return ((batch_index * heads + head_index) * time + time_index) *
                   head_dim +
               dim;
    };
    const auto position_element = [time, head_dim](size_t head_index,
                                                   size_t position_index,
                                                   size_t dim) {
        return ((head_index * (2 * time - 1)) + position_index) * head_dim +
               dim;
    };

    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (size_t head_index = 0; head_index < heads; ++head_index) {
            for (size_t query_index = 0; query_index < time; ++query_index) {
                std::vector<float> scores(time);
                float row_max = -INFINITY;
                for (size_t key_index = 0; key_index < time; ++key_index) {
                    if (m_data[(batch_index * time + query_index) * time +
                               key_index]) {
                        scores[key_index] = -INFINITY;
                        continue;
                    }
                    float content = 0.0f;
                    float relative = 0.0f;
                    const size_t position_index =
                        time - 1 - query_index + key_index;
                    for (size_t dim = 0; dim < head_dim; ++dim) {
                        const float q_value =
                            q_data[element(batch_index, head_index, query_index,
                                           dim)];
                        content += (q_value + u_data[head_index * head_dim + dim]) *
                                   k_data[element(batch_index, head_index,
                                                  key_index, dim)];
                        relative +=
                            (q_value + b_data[head_index * head_dim + dim]) *
                            p_data[position_element(head_index, position_index,
                                                    dim)];
                    }
                    scores[key_index] = (content + relative) * scale;
                    row_max = std::max(row_max, scores[key_index]);
                }

                float row_sum = 0.0f;
                for (float &score : scores) {
                    score = std::exp(score - row_max);
                    row_sum += score;
                }
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    float weighted = 0.0f;
                    for (size_t key_index = 0; key_index < time; ++key_index) {
                        weighted += scores[key_index] *
                                    v_data[element(batch_index, head_index,
                                                   key_index, dim)];
                    }
                    out[element(batch_index, head_index, query_index, dim)] =
                        weighted / row_sum;
                }
            }
        }
    }
    return output;
}

// Matches ConformerAttention's established MPSGraph composition. Keeping this
// beside the independent CPU reference separates a fused-kernel indexing bug
// from ordinary FP16/MPSGraph numerical differences.
Tensor relative_position_attention_mpsgraph(const Tensor &query,
                                            const Tensor &key,
                                            const Tensor &value,
                                            const Tensor &position,
                                            const Tensor &bias_u,
                                            const Tensor &bias_v,
                                            const Tensor &mask, float scale) {
    const size_t batch = query.shape()[0];
    const size_t heads = query.shape()[1];
    const size_t time = query.shape()[2];
    const size_t head_dim = query.shape()[3];
    const size_t position_length = position.shape()[2];
    const Tensor u = bias_u.reshape({1, heads, 1, head_dim});
    const Tensor b = bias_v.reshape({1, heads, 1, head_dim});
    const Tensor content = axiom::ops::matmul(query + u, key, false, true);
    Tensor relative = axiom::ops::matmul(query + b, position, false, true);
    relative = axiom::ops::pad(relative, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
    relative = relative.reshape({batch, heads, position_length + 1, time});
    relative = relative.slice(
        {axiom::Slice(), axiom::Slice(), axiom::Slice(1), axiom::Slice()});
    relative = relative.reshape({batch, heads, time, position_length});
    relative = relative.slice(
        {axiom::Slice(), axiom::Slice(), axiom::Slice(),
         axiom::Slice(0, static_cast<int64_t>(time))});
    Tensor scores = (content + relative) * scale;
    if (mask.storage()) {
        scores = axiom::ops::masked_fill(scores, mask, -1e9f);
    }
    return axiom::ops::matmul(axiom::ops::softmax(scores, -1), value);
}

Tensor relative_position_attention_scores_mpsgraph(
    const Tensor &query, const Tensor &key, const Tensor &position,
    const Tensor &bias_u, const Tensor &bias_v, const Tensor &mask,
    float scale) {
    const size_t batch = query.shape()[0];
    const size_t heads = query.shape()[1];
    const size_t time = query.shape()[2];
    const size_t head_dim = query.shape()[3];
    const size_t position_length = position.shape()[2];
    const Tensor u = bias_u.reshape({1, heads, 1, head_dim});
    const Tensor v = bias_v.reshape({1, heads, 1, head_dim});
    const Tensor content = axiom::ops::matmul(query + u, key, false, true);
    Tensor relative = axiom::ops::matmul(query + v, position, false, true);
    relative = axiom::ops::pad(relative, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
    relative = relative.reshape({batch, heads, position_length + 1, time});
    relative = relative.slice(
        {axiom::Slice(), axiom::Slice(), axiom::Slice(1), axiom::Slice()});
    relative = relative.reshape({batch, heads, time, position_length});
    relative = relative.slice(
        {axiom::Slice(), axiom::Slice(), axiom::Slice(),
         axiom::Slice(0, static_cast<int64_t>(time))});
    Tensor scores = (content + relative) * scale;
    return mask.storage() ? axiom::ops::masked_fill(scores, mask, -1e9f)
                          : scores;
}

// The position-only half of FastConformer attention. This deliberately stays
// in the test so the additive-bias attention API is checked against an
// independently composed MPSGraph result, not a second call through itself.
Tensor relative_position_bias_mpsgraph(const Tensor &query,
                                       const Tensor &position,
                                       const Tensor &bias_v) {
    const size_t batch = query.shape()[0];
    const size_t heads = query.shape()[1];
    const size_t time = query.shape()[2];
    const size_t head_dim = query.shape()[3];
    const size_t position_length = position.shape()[2];
    const Tensor v = bias_v.reshape({1, heads, 1, head_dim});
    Tensor relative = axiom::ops::matmul(query + v, position, false, true);
    relative = axiom::ops::pad(relative, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
    relative = relative.reshape({batch, heads, position_length + 1, time});
    relative = relative.slice(
        {axiom::Slice(), axiom::Slice(), axiom::Slice(1), axiom::Slice()});
    relative = relative.reshape({batch, heads, time, position_length});
    return relative.slice(
        {axiom::Slice(), axiom::Slice(), axiom::Slice(),
         axiom::Slice(0, static_cast<int64_t>(time))});
}

class Int8FfnGpuTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!axiom::system::should_run_gpu_tests()) {
            GTEST_SKIP() << "Requires a Metal GPU";
        }

        normalized_ = deterministic_tensor({kBatch, kTime, kHidden}, 0.25f)
                          .astype(DType::Float16)
                          .to(Device::GPU);
        residual_ = deterministic_tensor({kBatch, kTime, kHidden}, 0.5f)
                        .astype(DType::Float16)
                        .to(Device::GPU);
        const QuantPair fc1 = quantize_block_symmetric_k32(
            deterministic_tensor({kIntermediate, kHidden}, 0.125f));
        const QuantPair fc2 = quantize_block_symmetric_k32(
            deterministic_tensor({kHidden, kIntermediate}, 0.125f));
        fc1_weight_ = fc1.weight;
        fc1_scale_ = fc1.scale;
        fc2_weight_ = fc2.weight;
        fc2_scale_ = fc2.scale;
        fc1_bias_ = deterministic_tensor({kIntermediate}, 0.1f)
                        .astype(DType::Float16)
                        .to(Device::GPU);
        fc2_bias_ = deterministic_tensor({kHidden}, 0.1f)
                        .astype(DType::Float16)
                        .to(Device::GPU);
    }

    Tensor normalized_;
    Tensor residual_;
    Tensor fc1_weight_;
    Tensor fc1_scale_;
    Tensor fc1_bias_;
    Tensor fc2_weight_;
    Tensor fc2_scale_;
    Tensor fc2_bias_;
};

TEST_F(Int8FfnGpuTest, DirectSequenceMatchesGenericInt8Reference) {
    const Tensor direct = axiom::ops::int8_ffn_silu_residual(
        normalized_, residual_, fc1_weight_, fc1_scale_, fc1_bias_,
        fc2_weight_, fc2_scale_, fc2_bias_);

    Tensor reference = axiom::ops::int8_matmul_bias(
        normalized_, fc1_weight_, fc1_scale_, fc1_bias_);
    reference = axiom::ops::silu(reference);
    reference = axiom::ops::int8_matmul_bias(
        reference, fc2_weight_, fc2_scale_, fc2_bias_);
    reference = residual_ + reference * 0.5f;

    EXPECT_EQ(direct.shape(), (Shape{kBatch, kTime, kHidden}));
    EXPECT_LE(max_abs_error(direct, reference), 2.0e-3f);
}

TEST(Int8GpuGraphCachePolicyTest, UsesAnExplicitBoundedProcessLimit) {
    ASSERT_EQ(setenv("AXIOM_GPU_GRAPH_CACHE_LIMIT", "4", /*overwrite=*/1), 0);
    EXPECT_EQ(axiom::graph::gpu_graph_cache_max_size(), 4u);
    unsetenv("AXIOM_GPU_GRAPH_CACHE_LIMIT");
    EXPECT_EQ(axiom::graph::gpu_graph_cache_max_size(), 128u);
}

TEST_F(Int8FfnGpuTest, DirectSequenceSupportsBiasFreeModelLayout) {
    const Tensor direct = axiom::ops::int8_ffn_silu_residual(
        normalized_, residual_, fc1_weight_, fc1_scale_, Tensor(),
        fc2_weight_, fc2_scale_, Tensor());

    Tensor reference =
        axiom::ops::int8_matmul(normalized_, fc1_weight_, fc1_scale_);
    reference = axiom::ops::silu(reference);
    reference = axiom::ops::int8_matmul(reference, fc2_weight_, fc2_scale_);
    reference = residual_ + reference * 0.5f;

    EXPECT_EQ(direct.shape(), (Shape{kBatch, kTime, kHidden}));
    EXPECT_LE(max_abs_error(direct, reference), 2.0e-3f);
}

TEST_F(Int8FfnGpuTest, DirectFullResidualMatchesMpsGraphReference) {
    const Tensor expected = normalized_ + residual_;
    Tensor actual = normalized_.clone();
    axiom::ops::int8_add_residual_inplace(actual, residual_);

    EXPECT_EQ(actual.shape(), (Shape{kBatch, kTime, kHidden}));
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);
}

TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionMatchesIndependentReferenceWithBroadcastMask) {
    constexpr size_t kHeads = 2;
    constexpr size_t kAttentionTime = 17;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16);
    const Tensor value = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.25f)
                             .astype(DType::Float16);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16);
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16);
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 13; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }

    const Tensor expected = relative_position_attention_reference(
        query, key, value, position, bias_u, bias_v, mask, attention_scale);
    const Tensor query_gpu = query.to(Device::GPU);
    const Tensor key_gpu = key.to(Device::GPU);
    const Tensor value_gpu = value.to(Device::GPU);
    const Tensor position_gpu = position.to(Device::GPU);
    const Tensor bias_u_gpu = bias_u.to(Device::GPU);
    const Tensor bias_v_gpu = bias_v.to(Device::GPU);
    const Tensor mask_gpu = mask.to(Device::GPU);
    const Tensor generic = relative_position_attention_mpsgraph(
        query_gpu, key_gpu, value_gpu, position_gpu, bias_u_gpu, bias_v_gpu,
        mask_gpu, attention_scale);
    const Tensor actual = axiom::ops::relative_position_attention(
        query_gpu, key_gpu, value_gpu, position_gpu, bias_u_gpu, bias_v_gpu,
        mask_gpu, attention_scale);

    EXPECT_EQ(actual.shape(),
              (Shape{kBatch, kHeads, kAttentionTime, kAttentionHeadDim}));
    EXPECT_LE(max_abs_error(generic, expected), 3.0e-3f);
    EXPECT_LE(max_abs_error(actual, generic), 3.0e-3f);
    EXPECT_LE(max_abs_error(actual, expected), 3.0e-3f);
}

TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionMatchesMpsGraphAtFastConformerLength) {
    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 240;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor value = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 220; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);

    const Tensor expected = relative_position_attention_mpsgraph(
        query, key, value, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor actual = axiom::ops::relative_position_attention(
        query, key, value, position, bias_u, bias_v, mask_gpu, attention_scale);

    EXPECT_EQ(actual.shape(),
              (Shape{kBatch, kHeads, kAttentionTime, kAttentionHeadDim}));
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);
}

// Laboratory gate for a future complete Metal replacement. The TDT decoder
// has already shown that a small maximum tensor error can change text, so an
// opt-in candidate must match the established MPSGraph output byte-for-byte
// before it is allowed into an end-to-end timing screen. It is skipped in the
// normal suite because the current experimental kernel is intentionally known
// to fail this contract.
TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionStrictOptInRequiresByteExactOutputAtProductionShortLength) {
    const char *strict = std::getenv("AXIOM_STRICT_RELATIVE_POSITION_ATTENTION");
    if (strict == nullptr || strict[0] != '1' || strict[1] != '\0') {
        GTEST_SKIP() << "strict relative-attention laboratory gate is disabled";
    }

    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 44;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor value = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU)
                              .reshape({1, kHeads, 1, kAttentionHeadDim});
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU)
                              .reshape({1, kHeads, 1, kAttentionHeadDim});
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 40; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);

    // The native route is intentionally limited to materialized contiguous
    // Metal buffers. Force that real encoder boundary before building the
    // MPSGraph reference; otherwise a lazy input silently exercises the
    // fallback and turns this exactness gate into a false pass.
    query.sync();
    key.sync();
    value.sync();
    position.sync();
    bias_u.sync();
    bias_v.sync();
    mask_gpu.sync();
    ASSERT_TRUE(
        axiom::backends::metal::gpu_relative_position_attention_pipeline_available());

    const Tensor expected = relative_position_attention_mpsgraph(
        query, key, value, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor actual = axiom::ops::relative_position_attention(
        query, key, value, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor expected_cpu = expected.to(Device::CPU).astype(DType::Float16);
    const Tensor actual_cpu = actual.to(Device::CPU).astype(DType::Float16);

    ASSERT_EQ(actual_cpu.shape(), expected_cpu.shape());
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.data(), expected_cpu.data(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0)
        << "max_abs_error=" << max_abs_error(actual, expected);

}

// This companion gate localizes the strict-output failure above. The existing
// score-only route keeps MPSGraph softmax and value aggregation, so byte drift
// here proves that the candidate is already numerically different before
// softmax rather than blaming a downstream decoder-sensitive reduction.
TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionScoreStrictOptInRequiresByteExactMpsGraphScores) {
    const char *strict = std::getenv("AXIOM_STRICT_RELATIVE_POSITION_ATTENTION");
    if (strict == nullptr || strict[0] != '1' || strict[1] != '\0') {
        GTEST_SKIP() << "strict relative-attention laboratory gate is disabled";
    }

    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 44;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU)
                              .reshape({1, kHeads, 1, kAttentionHeadDim});
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU)
                              .reshape({1, kHeads, 1, kAttentionHeadDim});
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 40; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);
    query.sync();
    key.sync();
    position.sync();
    bias_u.sync();
    bias_v.sync();
    mask_gpu.sync();

    const Tensor expected = relative_position_attention_scores_mpsgraph(
        query, key, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor actual =
        axiom::backends::metal::gpu_relative_position_attention_scores_tiled(
            query, key, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor expected_cpu = expected.to(Device::CPU).astype(DType::Float16);
    const Tensor actual_cpu = actual.to(Device::CPU).astype(DType::Float16);

    ASSERT_EQ(actual_cpu.shape(), expected_cpu.shape());
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.data(), expected_cpu.data(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0)
        << "max_abs_error=" << max_abs_error(actual, expected);

    // Nulling one branch at a time preserves the same direct Metal score
    // operation and mask while exposing whether the first mismatch belongs to
    // content QK multiplication or the relative-position score/shift path.
    const Tensor zero_key = Tensor::zeros(key.shape(), DType::Float16, Device::GPU);
    const Tensor zero_position =
        Tensor::zeros(position.shape(), DType::Float16, Device::GPU);
    const Tensor zero_bias = Tensor::zeros(bias_u.shape(), DType::Float16,
                                            Device::GPU);
    zero_key.sync();
    zero_position.sync();
    zero_bias.sync();

    const auto expect_exact_scores = [&](const char *label, const Tensor &actual_scores,
                                         const Tensor &expected_scores) {
        const Tensor expected_stage_cpu =
            expected_scores.to(Device::CPU).astype(DType::Float16);
        const Tensor actual_stage_cpu =
            actual_scores.to(Device::CPU).astype(DType::Float16);
        ASSERT_EQ(actual_stage_cpu.shape(), expected_stage_cpu.shape()) << label;
        ASSERT_EQ(actual_stage_cpu.numel(), expected_stage_cpu.numel()) << label;
        EXPECT_EQ(std::memcmp(actual_stage_cpu.data(), expected_stage_cpu.data(),
                              actual_stage_cpu.numel() * sizeof(uint16_t)),
                  0)
            << label << " max_abs_error="
            << max_abs_error(actual_scores, expected_scores);
    };

    const Tensor expected_content = relative_position_attention_scores_mpsgraph(
        query, key, zero_position, bias_u, zero_bias, mask_gpu, attention_scale);
    const Tensor actual_content =
        axiom::backends::metal::gpu_relative_position_attention_scores_tiled(
            query, key, zero_position, bias_u, zero_bias, mask_gpu,
            attention_scale);
    expect_exact_scores("content-score branch", actual_content, expected_content);

    const Tensor expected_relative = relative_position_attention_scores_mpsgraph(
        query, zero_key, position, zero_bias, bias_v, mask_gpu, attention_scale);
    const Tensor actual_relative =
        axiom::backends::metal::gpu_relative_position_attention_scores_tiled(
            query, zero_key, position, zero_bias, bias_v, mask_gpu,
            attention_scale);
    expect_exact_scores("relative-score-and-shift branch", actual_relative,
                        expected_relative);
}

// Keep the experiment honest: the opt-in must select a distinct SIMD-group
// kernel, rather than quietly falling back to the established scalar kernel
// or the MPSGraph decomposition.  The workspace trace observes the actual
// Metal operation while the output check retains the numerical contract.
TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionTiledScoresOptInSchedulesNamedMetalOperation) {
    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 120;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor value = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU)
                              .reshape({1, kHeads, 1, kAttentionHeadDim});
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU)
                              .reshape({1, kHeads, 1, kAttentionHeadDim});
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 112; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);
    const Tensor expected = relative_position_attention_mpsgraph(
        query, key, value, position, bias_u, bias_v, mask_gpu,
        attention_scale);
    expected.sync();

    const auto trace_path = std::filesystem::temp_directory_path() /
                            "axiom-relative-attention-simd-workspace-trace.json";
    std::filesystem::remove(trace_path);
    ASSERT_EQ(setenv("PARAKEET_RELATIVE_POSITION_ATTENTION_SCORE_TILED", "1",
                     /*overwrite=*/1),
              0);
    ASSERT_EQ(setenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH",
                     trace_path.c_str(), /*overwrite=*/1),
              0);

    float actual_error = 0.0f;
    try {
        axiom::backends::metal::MetalWorkspaceCache cache(
            32ULL * 1024ULL * 1024ULL);
        axiom::backends::metal::ScopedMetalWorkspace workspace(cache);
        {
            const Tensor actual = axiom::ops::relative_position_attention(
                query, key, value, position, bias_u, bias_v, mask_gpu,
                attention_scale);
            actual_error = max_abs_error(actual, expected);
        }
        axiom::backends::metal::MetalExecutionStream::instance().synchronize();
        workspace.close();
    } catch (...) {
        unsetenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH");
        unsetenv("PARAKEET_RELATIVE_POSITION_ATTENTION_SCORE_TILED");
        std::filesystem::remove(trace_path);
        throw;
    }

    unsetenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH");
    unsetenv("PARAKEET_RELATIVE_POSITION_ATTENTION_SCORE_TILED");
    ASSERT_TRUE(std::filesystem::exists(trace_path));
    std::ifstream trace_file(trace_path);
    const std::string trace_json((std::istreambuf_iterator<char>(trace_file)),
                                 std::istreambuf_iterator<char>());
    std::filesystem::remove(trace_path);

    EXPECT_LE(actual_error, 1.0e-3f);
    EXPECT_NE(trace_json.find("\"label\":\"relative_position_attention_scores_tiled\""),
              std::string::npos);
}

// Catches a future kernel that misinterprets the relative-position score as a
// content score, drops the per-head content bias, or changes mask broadcast.
// The expected tensor is the established full MPSGraph composition; the
// additive bias is composed independently in the test before it reaches the
// new operation boundary.
TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionFromBiasMatchesMpsGraphAtProductionShortLength) {
    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 120;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor value = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 112; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);

    const Tensor expected = relative_position_attention_mpsgraph(
        query, key, value, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor relative_bias = relative_position_bias_mpsgraph(
        query, position, bias_v);
    const Tensor actual = axiom::ops::relative_position_attention_from_bias(
        query, key, value, relative_bias, bias_u, mask_gpu, attention_scale);

    EXPECT_EQ(actual.shape(),
              (Shape{kBatch, kHeads, kAttentionTime, kAttentionHeadDim}));
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);

    if (const char *benchmark =
            std::getenv("AXIOM_BENCHMARK_RELATIVE_BIAS_ATTENTION");
        benchmark && *benchmark && benchmark[0] != '0') {
        const double mpsgraph_ms = median_gpu_milliseconds([&] {
            return axiom::ops::relative_position_attention_from_bias(
                query, key, value, relative_bias, bias_u, mask_gpu,
                attention_scale);
        });
        ASSERT_EQ(setenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS", "1",
                         /*overwrite=*/1),
                  0);
        const double custom_ms = median_gpu_milliseconds([&] {
            return axiom::ops::relative_position_attention_from_bias(
                query, key, value, relative_bias, bias_u, mask_gpu,
                attention_scale);
        });
        const double full_mpsgraph_ms = median_gpu_milliseconds([&] {
            return relative_position_attention_mpsgraph(
                query, key, value, position, bias_u, bias_v, mask_gpu,
                attention_scale);
        });
        const double split_candidate_ms = median_gpu_milliseconds([&] {
            const Tensor fresh_relative_bias = relative_position_bias_mpsgraph(
                query, position, bias_v);
            return axiom::ops::relative_position_attention_from_bias(
                query, key, value, fresh_relative_bias, bias_u, mask_gpu,
                attention_scale);
        });
        unsetenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS");
        std::cout << "relative-bias attention T=120: MPSGraph="
                  << mpsgraph_ms << " ms, native=" << custom_ms
                  << " ms; full MPSGraph=" << full_mpsgraph_ms
                  << " ms, split candidate=" << split_candidate_ms << " ms\n";
    }
}

// This protects the SIMD experimental execution boundary itself. Numerical
// parity alone is insufficient: a future opt-in that silently uses either the
// scalar baseline or MPSGraph would look correct while providing none of the
// intended scheduling or allocation benefit. The active workspace trace
// observes the operation actually submitted to Metal.
TEST_F(Int8FfnGpuTest,
       RelativePositionAttentionFromBiasSimdOptInSchedulesNamedMetalOperation) {
    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 120;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor value = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor relative_bias = relative_position_bias_mpsgraph(
        query, position, bias_v);
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 112; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);
    const Tensor expected = relative_position_attention_mpsgraph(
        query, key, value, position, bias_u, bias_v, mask_gpu,
        attention_scale);
    // Both tensors are intentionally built before the scoped experiment. Make
    // them concrete now: otherwise their lazy MPSGraph work would allocate
    // inside the workspace and remain live through the trace assertion.
    relative_bias.sync();
    expected.sync();

    const auto trace_path = std::filesystem::temp_directory_path() /
                            "axiom-relative-bias-attention-workspace-trace.json";
    std::filesystem::remove(trace_path);
    ASSERT_EQ(setenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS", "1",
                     /*overwrite=*/1),
              0);
    ASSERT_EQ(setenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS_SIMDGROUP", "1",
                     /*overwrite=*/1),
              0);
    ASSERT_EQ(setenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH",
                     trace_path.c_str(), /*overwrite=*/1),
              0);

    float actual_error = 0.0f;
    try {
        axiom::backends::metal::MetalWorkspaceCache cache(
            32ULL * 1024ULL * 1024ULL);
        axiom::backends::metal::ScopedMetalWorkspace workspace(cache);
        {
            const Tensor actual = axiom::ops::relative_position_attention_from_bias(
                query, key, value, relative_bias, bias_u, mask_gpu,
                attention_scale);
            actual_error = max_abs_error(actual, expected);
        }
        axiom::backends::metal::MetalExecutionStream::instance().synchronize();
        workspace.close();
    } catch (...) {
        unsetenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH");
        unsetenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS_SIMDGROUP");
        unsetenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS");
        std::filesystem::remove(trace_path);
        throw;
    }

    unsetenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH");
    unsetenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS_SIMDGROUP");
    unsetenv("PARAKEET_RELATIVE_POSITION_FLASH_BIAS");
    ASSERT_TRUE(std::filesystem::exists(trace_path));
    std::ifstream trace_file(trace_path);
    const std::string trace_json((std::istreambuf_iterator<char>(trace_file)),
                                 std::istreambuf_iterator<char>());
    std::filesystem::remove(trace_path);

    EXPECT_LE(actual_error, 1.0e-3f);
    EXPECT_NE(trace_json.find("\"label\":\"relative_position_attention_from_bias_simdgroup\""),
              std::string::npos);
}

TEST_F(Int8FfnGpuTest,
       TiledRelativePositionScoresMatchMpsGraphAtProductionShortLength) {
    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 120;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 112; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);

    const Tensor expected = relative_position_attention_scores_mpsgraph(
        query, key, position, bias_u, bias_v, mask_gpu, attention_scale);
    const Tensor actual = axiom::ops::relative_position_attention_scores_tiled(
        query, key, position, bias_u, bias_v, mask_gpu, attention_scale);

    EXPECT_EQ(actual.shape(),
              (Shape{kBatch, kHeads, kAttentionTime, kAttentionTime}));
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);

    if (const char *benchmark =
            std::getenv("AXIOM_BENCHMARK_RELATIVE_SCORES");
        benchmark && *benchmark && benchmark[0] != '0') {
        const double mpsgraph_ms = median_gpu_milliseconds([&] {
            return relative_position_attention_scores_mpsgraph(
                query, key, position, bias_u, bias_v, mask_gpu,
                attention_scale);
        });
        const double tiled_ms = median_gpu_milliseconds([&] {
            return axiom::ops::relative_position_attention_scores_tiled(
                query, key, position, bias_u, bias_v, mask_gpu,
                attention_scale);
        });
        const Tensor value = deterministic_tensor(
            {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.18f)
                                 .astype(DType::Float16)
                                 .to(Device::GPU);
        const double mpsgraph_attention_ms = median_gpu_milliseconds([&] {
            return relative_position_attention_mpsgraph(
                query, key, value, position, bias_u, bias_v, mask_gpu,
                attention_scale);
        });
        std::cout << "relative-position scores T=120: MPSGraph=" << mpsgraph_ms
                  << " ms, tiled=" << tiled_ms
                  << " ms; full MPSGraph attention=" << mpsgraph_attention_ms
                  << " ms\n";
    }
}

// The wider tile is an independent scheduling path. Retaining this trace
// assertion prevents the environment switch from silently exercising the old
// 8x16 feasibility kernel while a benchmark appears to validate 16x16 work.
TEST_F(Int8FfnGpuTest,
       TiledRelativePositionScores16x16OptInSchedulesNamedMetalOperation) {
    constexpr size_t kHeads = 8;
    constexpr size_t kAttentionTime = 120;
    constexpr size_t kAttentionHeadDim = 128;
    const float attention_scale =
        1.0f / std::sqrt(static_cast<float>(kAttentionHeadDim));

    const Tensor query = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.20f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor key = deterministic_tensor(
        {kBatch, kHeads, kAttentionTime, kAttentionHeadDim}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor position = deterministic_tensor(
        {kBatch, kHeads, 2 * kAttentionTime - 1, kAttentionHeadDim}, 0.10f)
                                .astype(DType::Float16)
                                .to(Device::GPU);
    const Tensor bias_u = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor bias_v = deterministic_tensor({kHeads, kAttentionHeadDim}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    Tensor mask = Tensor::zeros({kBatch, 1, kAttentionTime, kAttentionTime},
                                DType::Bool, Device::CPU);
    auto *mask_data = mask.typed_data<uint8_t>();
    for (size_t query_index = 0; query_index < kAttentionTime; ++query_index) {
        for (size_t key_index = 112; key_index < kAttentionTime; ++key_index) {
            mask_data[query_index * kAttentionTime + key_index] = 1;
        }
    }
    const Tensor mask_gpu = mask.to(Device::GPU);
    const Tensor expected = relative_position_attention_scores_mpsgraph(
        query, key, position, bias_u, bias_v, mask_gpu, attention_scale);
    expected.sync();

    const auto trace_path = std::filesystem::temp_directory_path() /
                            "axiom-relative-score-16x16-workspace-trace.json";
    std::filesystem::remove(trace_path);
    ASSERT_EQ(setenv("PARAKEET_RELATIVE_POSITION_SCORE_TILE_16", "1",
                     /*overwrite=*/1),
              0);
    ASSERT_EQ(setenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH",
                     trace_path.c_str(), /*overwrite=*/1),
              0);

    float actual_error = 0.0f;
    try {
        axiom::backends::metal::MetalWorkspaceCache cache(
            32ULL * 1024ULL * 1024ULL);
        axiom::backends::metal::ScopedMetalWorkspace workspace(cache);
        {
            const Tensor actual = axiom::ops::relative_position_attention_scores_tiled(
                query, key, position, bias_u, bias_v, mask_gpu,
                attention_scale);
            actual_error = max_abs_error(actual, expected);
        }
        axiom::backends::metal::MetalExecutionStream::instance().synchronize();
        workspace.close();
    } catch (...) {
        unsetenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH");
        unsetenv("PARAKEET_RELATIVE_POSITION_SCORE_TILE_16");
        std::filesystem::remove(trace_path);
        throw;
    }

    unsetenv("WASPER_PARAKEET_ENCODER_WORKSPACE_TRACE_PATH");
    unsetenv("PARAKEET_RELATIVE_POSITION_SCORE_TILE_16");
    ASSERT_TRUE(std::filesystem::exists(trace_path));
    std::ifstream trace_file(trace_path);
    const std::string trace_json((std::istreambuf_iterator<char>(trace_file)),
                                 std::istreambuf_iterator<char>());
    std::filesystem::remove(trace_path);

    EXPECT_LE(actual_error, 1.0e-3f);
    EXPECT_NE(trace_json.find(
                  "\"label\":\"relative_position_attention_scores_tiled_16x16\""),
              std::string::npos);
}

TEST_F(Int8FfnGpuTest,
       Int8PointwiseConvMatchesFp16ConvWithinQuantizationTolerance) {
    constexpr size_t kChannels = 64;
    constexpr size_t kOutputChannels = 128;
    constexpr size_t kConvTime = 31;
    const Tensor weight = deterministic_tensor(
        {kOutputChannels, kChannels, size_t{1}}, 0.10f);
    const QuantPair quantized =
        quantize_block_symmetric_k32(weight.reshape({kOutputChannels, kChannels}));
    axiom::nn::Conv1d conv(/*stride=*/1, /*padding=*/0, /*dilation=*/1,
                           /*groups=*/1, /*bias=*/false);
    conv.load_state_dict({{"weight", weight}}, "", /*strict=*/true);
    conv.to(DType::Float16).to(Device::GPU);
    const Tensor input = deterministic_tensor({kBatch, kChannels, kConvTime}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);

    const Tensor expected = conv(input);
    conv.load_int8_pointwise_weights(quantized.weight, quantized.scale);
    const Tensor actual = conv(input);

    EXPECT_EQ(actual.shape(), (Shape{kBatch, kOutputChannels, kConvTime}));
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-2f);
}

TEST_F(Int8FfnGpuTest, DirectSequenceKeepsSixKernelsOnOneMetalStream) {
    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    stream.synchronize();

    const Tensor output = axiom::ops::int8_ffn_silu_residual(
        normalized_, residual_, fc1_weight_, fc1_scale_, fc1_bias_,
        fc2_weight_, fc2_scale_, fc2_bias_);

    EXPECT_EQ(output.shape(), (Shape{kBatch, kTime, kHidden}));
    EXPECT_EQ(stream.current_batch_size(), 6u);
}

TEST_F(Int8FfnGpuTest,
       FusedHalfResidualFlagCombinesTheFinalFfnStagesWithoutChangingOutput) {
    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    stream.synchronize();

    ASSERT_EQ(setenv("PARAKEET_FUSED_INT8_FFN_HALF_RESIDUAL", "1",
                     /*overwrite=*/1),
              0);
    const Tensor fused = axiom::ops::int8_ffn_silu_residual(
        normalized_, residual_, fc1_weight_, fc1_scale_, Tensor(),
        fc2_weight_, fc2_scale_, Tensor());
    unsetenv("PARAKEET_FUSED_INT8_FFN_HALF_RESIDUAL");

    EXPECT_EQ(fused.shape(), (Shape{kBatch, kTime, kHidden}));
    // The Parakeet-TDT-v3 FFNs are bias-free: FC1, SiLU, then FC2/residual.
    EXPECT_EQ(stream.current_batch_size(), 3u);

    Tensor expected = axiom::ops::int8_matmul(
        normalized_, fc1_weight_, fc1_scale_);
    expected = axiom::ops::silu(expected);
    expected = axiom::ops::int8_matmul(expected, fc2_weight_, fc2_scale_);
    expected = residual_ + expected * 0.5f;

    EXPECT_LE(max_abs_error(fused, expected), 2.0e-3f);
}

TEST_F(Int8FfnGpuTest, MaterializedGpuGraphReshapeReusesStorageAsView) {
    // A preceding operation can materialize a copy of a lazy tensor. The
    // original handle still has its lazy node, but that shared node now owns a
    // concrete contiguous result. Reshaping the original must be metadata-only
    // rather than scheduling an otherwise-empty MPSGraph reshape.
    const Tensor zeros =
        Tensor::zeros({kBatch, kTime, kHidden}, DType::Float16, Device::GPU);
    Tensor graph_backed = residual_ + zeros;
    const Tensor materializing_copy = graph_backed;
    static_cast<void>(materializing_copy.storage());
    ASSERT_TRUE(graph_backed.is_lazy());

    const Tensor flattened = graph_backed.reshape({kBatch * kTime, kHidden});

    ASSERT_FALSE(flattened.is_lazy());
    EXPECT_EQ(flattened.shape(), (Shape{kBatch * kTime, kHidden}));
    EXPECT_TRUE(flattened.shares_storage(graph_backed));
}

TEST_F(Int8FfnGpuTest,
       BatchedQkvProjectionMatchesIndependentInt8ProjectionsForFullAndPartialTiles) {
    const QuantPair q = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.10f));
    const QuantPair k = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.15f));
    const QuantPair v = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.20f));
    const Tensor q_bias = deterministic_tensor({kHidden}, 0.03f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor k_bias = deterministic_tensor({kHidden}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor v_bias = deterministic_tensor({kHidden}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    for (const size_t qkv_time : {kTime, size_t{144}}) {
        const Tensor qkv_input =
            deterministic_tensor({kBatch, qkv_time, kHidden}, 0.25f)
                .astype(DType::Float16)
                .to(Device::GPU);
        const Tensor q_reference =
            axiom::ops::int8_matmul_bias(qkv_input, q.weight, q.scale, q_bias);
        const Tensor k_reference =
            axiom::ops::int8_matmul_bias(qkv_input, k.weight, k.scale, k_bias);
        const Tensor v_reference =
            axiom::ops::int8_matmul_bias(qkv_input, v.weight, v.scale, v_bias);

        stream.synchronize();
        const auto qkv = axiom::ops::int8_qkv_matmul_bias(
            qkv_input, q.weight, q.scale, q_bias, k.weight, k.scale, k_bias,
            v.weight, v.scale, v_bias);

        EXPECT_EQ(qkv[0].shape(), (Shape{kBatch, qkv_time, kHidden}));
        EXPECT_EQ(qkv[1].shape(), (Shape{kBatch, qkv_time, kHidden}));
        EXPECT_EQ(qkv[2].shape(), (Shape{kBatch, qkv_time, kHidden}));
        EXPECT_EQ(stream.current_batch_size(), 1u);
        EXPECT_LE(max_abs_error(qkv[0], q_reference), 2.0e-3f);
        EXPECT_LE(max_abs_error(qkv[1], k_reference), 2.0e-3f);
        EXPECT_LE(max_abs_error(qkv[2], v_reference), 2.0e-3f);
    }
}

TEST_F(Int8FfnGpuTest, BatchedQkvProjectionSupportsBiasFreeModelProjections) {
    const QuantPair q = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.10f));
    const QuantPair k = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.15f));
    const QuantPair v = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.20f));

    const Tensor q_reference = axiom::ops::int8_matmul(normalized_, q.weight, q.scale);
    const Tensor k_reference = axiom::ops::int8_matmul(normalized_, k.weight, k.scale);
    const Tensor v_reference = axiom::ops::int8_matmul(normalized_, v.weight, v.scale);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    stream.synchronize();
    const auto qkv = axiom::ops::int8_qkv_matmul_bias(
        normalized_, q.weight, q.scale, Tensor(), k.weight, k.scale, Tensor(),
        v.weight, v.scale, Tensor());

    EXPECT_EQ(stream.current_batch_size(), 1u);
    EXPECT_LE(max_abs_error(qkv[0], q_reference), 2.0e-3f);
    EXPECT_LE(max_abs_error(qkv[1], k_reference), 2.0e-3f);
    EXPECT_LE(max_abs_error(qkv[2], v_reference), 2.0e-3f);
}

TEST_F(Int8FfnGpuTest,
       BatchedQkvHeadLayoutMatchesIndependentTimeMajorProjections) {
    constexpr size_t kHeads = 4;
    constexpr size_t kHeadDim = kHidden / kHeads;
    constexpr size_t kQkvTime = 144;
    const QuantPair q = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.10f));
    const QuantPair k = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.15f));
    const QuantPair v = quantize_block_symmetric_k32(
        deterministic_tensor({kHidden, kHidden}, 0.20f));
    const Tensor q_bias = deterministic_tensor({kHidden}, 0.03f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor k_bias = deterministic_tensor({kHidden}, 0.05f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor v_bias = deterministic_tensor({kHidden}, 0.07f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor input = deterministic_tensor({kBatch, kQkvTime, kHidden}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);

    const Tensor q_reference =
        axiom::ops::int8_matmul_bias(input, q.weight, q.scale, q_bias);
    const Tensor k_reference =
        axiom::ops::int8_matmul_bias(input, k.weight, k.scale, k_bias);
    const Tensor v_reference =
        axiom::ops::int8_matmul_bias(input, v.weight, v.scale, v_bias);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    stream.synchronize();
    const auto qkv = axiom::ops::int8_qkv_matmul_bias_head_layout(
        input, q.weight, q.scale, q_bias, k.weight, k.scale, k_bias, v.weight,
        v.scale, v_bias, kHeads);

    ASSERT_EQ(qkv[0].shape(), (Shape{kBatch, kHeads, kQkvTime, kHeadDim}));
    ASSERT_EQ(qkv[1].shape(), (Shape{kBatch, kHeads, kQkvTime, kHeadDim}));
    ASSERT_EQ(qkv[2].shape(), (Shape{kBatch, kHeads, kQkvTime, kHeadDim}));
    EXPECT_EQ(stream.current_batch_size(), 1u);

    const std::array<Tensor, 3> references = {q_reference, k_reference,
                                                v_reference};
    for (size_t projection = 0; projection < qkv.size(); ++projection) {
        const Tensor actual_cpu = qkv[projection].to(Device::CPU).astype(DType::Float32);
        const Tensor expected_cpu =
            references[projection].to(Device::CPU).astype(DType::Float32);
        const auto *actual = actual_cpu.typed_data<float>();
        const auto *expected = expected_cpu.typed_data<float>();
        for (size_t time = 0; time < kQkvTime; ++time) {
            for (size_t head = 0; head < kHeads; ++head) {
                for (size_t dim = 0; dim < kHeadDim; ++dim) {
                    const size_t actual_index =
                        ((head * kQkvTime) + time) * kHeadDim + dim;
                    const size_t expected_index =
                        time * kHidden + head * kHeadDim + dim;
                    EXPECT_NEAR(actual[actual_index], expected[expected_index],
                                2.0e-3f);
                }
            }
        }
    }
}

TEST_F(Int8FfnGpuTest, RejectsWrongFc2BiasLengthBeforeEncoding) {
    Tensor wrong_bias = deterministic_tensor({kHidden - 1}, 0.1f)
                            .astype(DType::Float16)
                            .to(Device::GPU);

    try {
        static_cast<void>(axiom::ops::int8_ffn_silu_residual(
            normalized_, residual_, fc1_weight_, fc1_scale_, fc1_bias_,
            fc2_weight_, fc2_scale_, wrong_bias));
        FAIL() << "Expected ShapeError for fc2_bias";
    } catch (const ShapeError &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("fc2_bias"), std::string::npos);
        EXPECT_NE(message.find("[K=64]"), std::string::npos);
    }
}

TEST_F(Int8FfnGpuTest, ExactFlagRoutesFeedForwardToMaterializedDirectOutput) {
    parakeet::models::FeedForward feed_forward(/*dropout=*/0.0f,
                                               /*bias=*/true);
    std::map<std::string, Tensor> state_dict;
    state_dict["norm_.weight"] = Tensor::ones({kHidden}, DType::Float32);
    state_dict["norm_.bias"] = Tensor::zeros({kHidden}, DType::Float32);
    state_dict["fc1_.weight"] =
        Tensor::zeros({kIntermediate, kHidden}, DType::Float32);
    state_dict["fc1_.bias"] = fc1_bias_.cpu().astype(DType::Float32);
    state_dict["fc2_.weight"] =
        Tensor::zeros({kHidden, kIntermediate}, DType::Float32);
    state_dict["fc2_.bias"] = fc2_bias_.cpu().astype(DType::Float32);
    feed_forward.load_state_dict(state_dict, "", /*strict=*/false);
    feed_forward.load_int8_weights(fc1_weight_.cpu(), fc1_scale_.cpu(),
                                   fc2_weight_.cpu(), fc2_scale_.cpu());
    feed_forward.to(DType::Float16);
    feed_forward.to(Device::GPU);

    ASSERT_TRUE(feed_forward.is_int8());
    ASSERT_EQ(setenv("PARAKEET_DIRECT_INT8_FFN", "1", /*overwrite=*/1), 0);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    stream.synchronize();
    const Tensor output = feed_forward.forward(residual_);
    unsetenv("PARAKEET_DIRECT_INT8_FFN");

    EXPECT_FALSE(output.is_lazy());
    EXPECT_EQ(output.shape(), (Shape{kBatch, kTime, kHidden}));
    EXPECT_EQ(stream.current_batch_size(), 7u);
}

TEST_F(Int8FfnGpuTest, ExactFlagRoutesBiasFreeFeedForwardToDirectOutput) {
    parakeet::models::FeedForward feed_forward(/*dropout=*/0.0f,
                                               /*bias=*/false);
    std::map<std::string, Tensor> state_dict;
    state_dict["norm_.weight"] = Tensor::ones({kHidden}, DType::Float32);
    state_dict["norm_.bias"] = Tensor::zeros({kHidden}, DType::Float32);
    state_dict["fc1_.weight"] =
        Tensor::zeros({kIntermediate, kHidden}, DType::Float32);
    state_dict["fc2_.weight"] =
        Tensor::zeros({kHidden, kIntermediate}, DType::Float32);
    feed_forward.load_state_dict(state_dict, "", /*strict=*/false);
    feed_forward.load_int8_weights(fc1_weight_.cpu(), fc1_scale_.cpu(),
                                   fc2_weight_.cpu(), fc2_scale_.cpu());
    feed_forward.to(DType::Float16);
    feed_forward.to(Device::GPU);

    ASSERT_TRUE(feed_forward.is_int8());
    ASSERT_EQ(setenv("PARAKEET_DIRECT_INT8_FFN", "1", /*overwrite=*/1), 0);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    stream.synchronize();
    const Tensor output = feed_forward.forward(residual_);
    unsetenv("PARAKEET_DIRECT_INT8_FFN");

    EXPECT_FALSE(output.is_lazy());
    EXPECT_EQ(output.shape(), (Shape{kBatch, kTime, kHidden}));
    EXPECT_EQ(stream.current_batch_size(), 5u);
}

TEST_F(Int8FfnGpuTest, DirectF16LayerNormFlagUsesCustomKernelWithinTolerance) {
    constexpr size_t kNormTime = 31;
    constexpr size_t kNormHidden = 1024;
    const Tensor input = deterministic_tensor({1, kNormTime, kNormHidden}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor weight = deterministic_tensor({kNormHidden}, 0.5f);
    const Tensor bias = deterministic_tensor({kNormHidden}, 0.125f);

    axiom::nn::LayerNorm layer_norm;
    layer_norm.load_state_dict({{"weight", weight}, {"bias", bias}}, "",
                               /*strict=*/true);
    layer_norm.to(DType::Float16);
    layer_norm.to(Device::GPU);

    const Tensor expected = layer_norm.forward(input);
    ASSERT_TRUE(expected.is_lazy());
    static_cast<void>(expected.cpu());

    ASSERT_EQ(setenv("PARAKEET_DIRECT_F16_LAYERNORM", "1", /*overwrite=*/1), 0);
    const Tensor actual = layer_norm.forward(input);
    unsetenv("PARAKEET_DIRECT_F16_LAYERNORM");

    EXPECT_FALSE(actual.is_lazy());
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);
}

TEST_F(Int8FfnGpuTest, DirectF16LayerNormRowStatsMatchCurrentReduction) {
    constexpr size_t kNormTime = 31;
    constexpr size_t kNormHidden = 1024;
    constexpr float kEpsilon = 1.0e-5f;
    const Tensor input = deterministic_tensor({1, kNormTime, kNormHidden}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);

    const Tensor actual = axiom::ops::layer_norm_row_stats_f16(input, kEpsilon);
    ASSERT_FALSE(actual.is_lazy());
    ASSERT_EQ(actual.shape(), (Shape{kNormTime, 2}));

    const Tensor stats_cpu = actual.cpu().astype(DType::Float32);
    const Tensor input_cpu = input.cpu().astype(DType::Float32);
    const auto *stats = stats_cpu.typed_data<float>();
    const auto *values = input_cpu.typed_data<float>();
    for (size_t row = 0; row < kNormTime; ++row) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (size_t column = 0; column < kNormHidden; ++column) {
            const float value = values[row * kNormHidden + column];
            sum += value;
            sum_sq += value * value;
        }
        const float mean = sum / static_cast<float>(kNormHidden);
        float variance = sum_sq / static_cast<float>(kNormHidden) - mean * mean;
        variance = variance < 1.0e-6f ? 0.0f : variance;
        const float inv_std = 1.0f / std::sqrt(variance + kEpsilon);

        EXPECT_NEAR(stats[row * 2], mean, 1.0e-4f);
        EXPECT_NEAR(stats[row * 2 + 1], inv_std, 1.0e-4f);
    }
}

TEST_F(Int8FfnGpuTest, LoopedF16LayerNormMatchesMpsGraphWithinTolerance) {
    constexpr size_t kNormTime = 2;
    constexpr size_t kNormHidden = 4097;
    const Tensor input = deterministic_tensor({1, kNormTime, kNormHidden}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor weight = deterministic_tensor({kNormHidden}, 0.5f);
    const Tensor bias = deterministic_tensor({kNormHidden}, 0.125f);

    axiom::nn::LayerNorm layer_norm;
    layer_norm.load_state_dict({{"weight", weight}, {"bias", bias}}, "",
                               /*strict=*/true);
    layer_norm.to(DType::Float16);
    layer_norm.to(Device::GPU);

    const Tensor expected = layer_norm.forward(input);
    ASSERT_TRUE(expected.is_lazy());
    static_cast<void>(expected.cpu());

    ASSERT_EQ(setenv("PARAKEET_DIRECT_F16_LAYERNORM", "1", /*overwrite=*/1), 0);
    const Tensor actual = layer_norm.forward(input);
    unsetenv("PARAKEET_DIRECT_F16_LAYERNORM");

    EXPECT_FALSE(actual.is_lazy());
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);
}

TEST_F(Int8FfnGpuTest, FusedAddF16LayerNormMatchesMpsGraphWithinTolerance) {
    constexpr size_t kNormTime = 31;
    constexpr size_t kNormHidden = 1024;
    const Tensor lhs = deterministic_tensor({1, kNormTime, kNormHidden}, 0.25f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor rhs = deterministic_tensor({1, kNormTime, kNormHidden}, 0.15f)
                           .astype(DType::Float16)
                           .to(Device::GPU);
    const Tensor weight = deterministic_tensor({kNormHidden}, 0.5f)
                              .astype(DType::Float16)
                              .to(Device::GPU);
    const Tensor bias = deterministic_tensor({kNormHidden}, 0.125f)
                            .astype(DType::Float16)
                            .to(Device::GPU);

    const Tensor expected = axiom::ops::layer_norm(lhs + rhs, weight, bias,
                                                    /*axis=*/-1, /*eps=*/1.0e-5f);
    static_cast<void>(expected.cpu());

    const Tensor actual = axiom::ops::add_layer_norm(
        lhs, rhs, weight, bias, /*axis=*/-1, /*eps=*/1.0e-5f);

    EXPECT_FALSE(actual.is_lazy());
    EXPECT_EQ(actual.shape(), (Shape{1, kNormTime, kNormHidden}));
    EXPECT_LE(max_abs_error(actual, expected), 1.0e-3f);
}

TEST_F(Int8FfnGpuTest, FusedDepthwiseConvBatchNormSiluMatchesMpsGraph) {
    constexpr size_t kChannels = 64;
    constexpr size_t kConvTime = 31;
    const Tensor input = deterministic_tensor({1, kChannels, kConvTime}, 0.25f)
                             .astype(DType::Float16)
                             .to(Device::GPU);
    const Tensor depthwise_weight = deterministic_tensor({kChannels, 1, 9}, 0.125f)
                                        .astype(DType::Float16)
                                        .to(Device::GPU);
    const Tensor running_mean = deterministic_tensor({kChannels}, 0.05f)
                                    .astype(DType::Float16)
                                    .to(Device::GPU);
    const Tensor running_var =
        (deterministic_tensor({kChannels}, 0.05f) + 1.0f).astype(DType::Float16).to(Device::GPU);
    const Tensor affine_weight = deterministic_tensor({kChannels}, 0.25f)
                                     .astype(DType::Float16)
                                     .to(Device::GPU);
    const Tensor affine_bias = deterministic_tensor({kChannels}, 0.05f)
                                   .astype(DType::Float16)
                                   .to(Device::GPU);

    const Tensor convolved = axiom::ops::conv1d(
        input, depthwise_weight, Tensor(), /*stride=*/1, /*padding=*/4,
        /*dilation=*/1, /*groups=*/static_cast<int>(kChannels));
    axiom::nn::BatchNorm1d batch_norm;
    batch_norm.load_state_dict(
        {{"weight", affine_weight}, {"bias", affine_bias},
         {"running_mean", running_mean}, {"running_var", running_var}},
        "", /*strict=*/false);
    const Tensor expected = axiom::ops::silu(batch_norm.forward(convolved));
    static_cast<void>(expected.cpu());

    const Tensor actual = axiom::ops::depthwise_conv1d_batch_norm_silu(
        input, depthwise_weight, running_mean, running_var, affine_weight,
        affine_bias, /*eps=*/1.0e-5f);

    EXPECT_FALSE(actual.is_lazy());
    EXPECT_EQ(actual.shape(), (Shape{1, kChannels, kConvTime}));
    // The fused depthwise kernel stores [B,T,C] physically and returns a
    // [B,C,T] view.  Conv1d's following pointwise projection permutes it back
    // to [B,T,C], which is then already contiguous and needs no gather.
    EXPECT_FALSE(actual.is_contiguous());
    EXPECT_TRUE(actual.permute({0, 2, 1}).is_contiguous());
    EXPECT_LE(max_abs_error(actual, expected), 2.0e-3f);
}

} // namespace
