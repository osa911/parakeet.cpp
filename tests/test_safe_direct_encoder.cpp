#include <gtest/gtest.h>

#include <axiom/io/safetensors.hpp>
#include <axiom/system.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "parakeet/models/config.hpp"
// This test exercises the exact private attention routing boundary with
// separately allocated Q/K/V tensors. Keep that seam test-only rather than
// adding a production switch that could change inference behavior.
#define private public
#include "parakeet/models/encoder.hpp"
#undef private

#ifndef PARAKEET_SAFE_DIRECT_MODEL_FIXTURE
#define PARAKEET_SAFE_DIRECT_MODEL_FIXTURE ""
#endif

extern char **environ;

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::Tensor;
using parakeet::models::FastConformerEncoder;

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.starts_with(prefix);
}

bool is_encoder_experiment(std::string_view name) {
    return starts_with(name, "PARAKEET_DIRECT_") ||
           starts_with(name, "PARAKEET_FUSED_") ||
           starts_with(name, "PARAKEET_CACHE_POSITION_") ||
           starts_with(name, "PARAKEET_INT8_POSITION_") ||
           starts_with(name, "PARAKEET_FASTCONFORMER_") ||
           starts_with(name, "PARAKEET_RELATIVE_POSITION_") ||
           starts_with(name, "WASPER_PARAKEET_SAFE_DIRECT_") ||
           (name.find("WORKSPACE") != std::string_view::npos &&
            (starts_with(name, "PARAKEET_") || starts_with(name, "AXIOM_") ||
             starts_with(name, "WASPER_")));
}

void unset_encoder_experiments() {
    std::vector<std::string> names;
    for (char **entry = environ; entry != nullptr && *entry != nullptr;
         ++entry) {
        const std::string_view assignment(*entry);
        const size_t separator = assignment.find('=');
        const std::string_view name = assignment.substr(0, separator);
        if (is_encoder_experiment(name)) {
            names.emplace_back(name);
        }
    }
    for (const std::string &name : names) {
        unsetenv(name.c_str());
    }
}

bool is_float_dtype(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16 ||
           dtype == DType::Float32 || dtype == DType::Float64;
}

std::string encoder_prefix(const std::map<std::string, Tensor> &weights) {
    for (const auto &[name, _] : weights) {
        if (starts_with(name, "encoder_.layers_.0.")) {
            return "encoder_.";
        }
    }
    for (const auto &[name, _] : weights) {
        if (starts_with(name, "layers_.0.")) {
            return "";
        }
    }
    throw std::runtime_error(
        "Safe Direct fixture has no FastConformer encoder weights");
}

bool belongs_to_encoder(std::string_view name, std::string_view prefix) {
    if (!prefix.empty()) {
        return starts_with(name, prefix);
    }
    return starts_with(name, "layers_.") || starts_with(name, "subsampling_.");
}

std::map<std::string, Tensor>
prepare_encoder_weights(const std::map<std::string, Tensor> &weights,
                        std::string_view prefix) {
    std::map<std::string, Tensor> prepared;
    for (const auto &[name, source] : weights) {
        if (!belongs_to_encoder(name, prefix)) {
            continue;
        }
        Tensor value = source;
        if (is_float_dtype(value.dtype()) && value.dtype() != DType::Float16) {
            value = value.astype(DType::Float16);
        }
        value = value.to(Device::GPU);
        prepared.emplace(name, std::move(value));
    }
    return prepared;
}

float max_abs_delta(const Tensor &actual, const Tensor &expected) {
    if (actual.shape() != expected.shape()) {
        return std::numeric_limits<float>::infinity();
    }
    const auto *actual_values = actual.typed_data<float>();
    const auto *expected_values = expected.typed_data<float>();
    float maximum = 0.0f;
    for (size_t index = 0; index < actual.numel(); ++index) {
        if (!std::isfinite(actual_values[index]) ||
            !std::isfinite(expected_values[index])) {
            return std::numeric_limits<float>::infinity();
        }
        maximum = std::max(
            maximum, std::abs(actual_values[index] - expected_values[index]));
    }
    return maximum;
}

TEST(SafeDirectEncoder, DirectQkvRequiresIdenticalLogicalViews) {
    const Tensor storage = Tensor::zeros({1, 5, 8}, DType::Float16);
    const Tensor first =
        storage.slice({axiom::Slice(), axiom::Slice(0, 4), axiom::Slice()});
    const Tensor shifted =
        storage.slice({axiom::Slice(), axiom::Slice(1, 5), axiom::Slice()});

    ASSERT_EQ(first.shape(), shifted.shape());
    ASSERT_TRUE(first.shares_storage(shifted));
    ASSERT_NE(first.offset(), shifted.offset());
    EXPECT_FALSE(::parakeet::models::detail::shares_direct_qkv_input(
        first, shifted, first));

    const Tensor square = Tensor::zeros({1, 4, 4}, DType::Float16);
    const Tensor transposed = square.transpose({0, 2, 1});

    ASSERT_EQ(square.shape(), transposed.shape());
    ASSERT_TRUE(square.shares_storage(transposed));
    ASSERT_NE(square.strides(), transposed.strides());
    EXPECT_FALSE(::parakeet::models::detail::shares_direct_qkv_input(
        square, transposed, square));

    EXPECT_TRUE(::parakeet::models::detail::shares_direct_qkv_input(
        first, first, first));
}

TEST(SafeDirectEncoder, IneligibleDirectQkvRouteKeepsLazyInputDeferred) {
#ifndef AXIOM_METAL_SUPPORT
    GTEST_SKIP() << "Metal/GPU not available";
#else
    const Tensor source =
        Tensor::zeros({1, 2, 32}, DType::Float16, Device::GPU);
    const Tensor lazy_input = source + source;
    ASSERT_TRUE(lazy_input.is_lazy());

    // Empty projections make the non-materializing Axiom capability check
    // reject the Direct-QKV route. Route selection must not inspect storage
    // identity first: shares_direct_qkv_input() synchronizes lazy tensors.
    const axiom::ops::Int8QkvProjections projections{};
    EXPECT_FALSE(::parakeet::models::detail::can_use_direct_qkv_head_layout(
        lazy_input, lazy_input, lazy_input, projections, 8));
    EXPECT_TRUE(lazy_input.is_lazy());
#endif
}

TEST(SafeDirectEncoder, UnsupportedHeadDimensionUsesGenericInt8Projections) {
#ifndef AXIOM_METAL_SUPPORT
    GTEST_SKIP() << "Metal/GPU not available";
#else
    // The Direct QKV Metal kernel requires head dimensions divisible by four.
    // This one-layer encoder has a valid Int8 projection shape but head_dim ==
    // 5. Its route statistics must show the established generic projections,
    // rather than the Direct QKV kernel.
    constexpr size_t hidden = 160;
    constexpr size_t num_heads = 32;
    constexpr size_t ffn_intermediate = 320;
    constexpr size_t subsampling_channels = 32;

    parakeet::models::EncoderConfig config;
    config.mel_bins = 80;
    config.subsampling_channels = static_cast<int>(subsampling_channels);
    config.hidden_size = static_cast<int>(hidden);
    config.num_layers = 1;
    config.num_heads = static_cast<int>(num_heads);
    config.ffn_intermediate = static_cast<int>(ffn_intermediate);
    config.dropout = 0.0f;
    FastConformerEncoder encoder(config);

    std::map<std::string, Tensor> state;

    const auto zeros = [](std::initializer_list<size_t> shape) {
        return Tensor::zeros(Shape(std::vector<size_t>(shape)), DType::Float32);
    };
    const auto ones = [](std::initializer_list<size_t> shape) {
        return Tensor::ones(Shape(std::vector<size_t>(shape)), DType::Float32);
    };
    const auto add_conv = [&](std::string_view name,
                              std::initializer_list<size_t> shape) {
        state.emplace(std::string(name) + ".weight", zeros(shape));
        state.emplace(std::string(name) + ".bias", zeros({shape.begin()[0]}));
    };

    add_conv("subsampling_.conv1_", {subsampling_channels, 1, 3, 3});
    add_conv("subsampling_.dw1_", {subsampling_channels, 1, 3, 3});
    add_conv("subsampling_.conv2_",
             {subsampling_channels, subsampling_channels, 1, 1});
    add_conv("subsampling_.dw2_", {subsampling_channels, 1, 3, 3});
    add_conv("subsampling_.conv3_",
             {subsampling_channels, subsampling_channels, 1, 1});
    state["subsampling_.proj_.weight"] =
        zeros({hidden, subsampling_channels * 10});
    state["subsampling_.proj_.bias"] = zeros({hidden});

    constexpr std::string_view layer = "layers_.0.";
    const auto add_norm = [&](std::string_view name) {
        state.emplace(std::string(layer) + std::string(name) + ".weight",
                      ones({hidden}));
        state.emplace(std::string(layer) + std::string(name) + ".bias",
                      zeros({hidden}));
    };
    const auto add_linear = [&](std::string_view name, size_t output,
                                size_t input) {
        state.emplace(std::string(layer) + std::string(name) + ".weight",
                      zeros({output, input}));
        state.emplace(std::string(layer) + std::string(name) + ".bias",
                      zeros({output}));
    };
    const auto add_int8_linear = [&](std::string_view name, size_t output,
                                     size_t input) {
        state.emplace(std::string(layer) + std::string(name) + "_quantized",
                      Tensor::zeros({output, input}, DType::Int8));
        state.emplace(std::string(layer) + std::string(name) + "_scale",
                      Tensor::ones({output, input / 32}, DType::Float16));
    };

    add_norm("ffn1_.norm_");
    add_linear("ffn1_.fc1_", ffn_intermediate, hidden);
    add_linear("ffn1_.fc2_", hidden, ffn_intermediate);
    add_int8_linear("ffn1_.fc1_", ffn_intermediate, hidden);
    add_int8_linear("ffn1_.fc2_", hidden, ffn_intermediate);

    add_norm("attn_.norm_");
    add_linear("attn_.mha_.q_proj", hidden, hidden);
    add_linear("attn_.mha_.k_proj", hidden, hidden);
    add_linear("attn_.mha_.v_proj", hidden, hidden);
    add_linear("attn_.mha_.out_proj", hidden, hidden);
    add_linear("attn_.pos_proj_", hidden, hidden);
    state[std::string(layer) + "attn_.pos_bias_u_"] =
        zeros({num_heads, hidden / num_heads});
    state[std::string(layer) + "attn_.pos_bias_v_"] =
        zeros({num_heads, hidden / num_heads});
    add_int8_linear("attn_.mha_.q_proj", hidden, hidden);
    add_int8_linear("attn_.mha_.k_proj", hidden, hidden);
    add_int8_linear("attn_.mha_.v_proj", hidden, hidden);
    add_int8_linear("attn_.mha_.out_proj", hidden, hidden);

    add_norm("conv_.norm_");
    add_conv("layers_.0.conv_.pointwise_conv1_", {2 * hidden, hidden, 1});
    add_conv("layers_.0.conv_.depthwise_conv_", {hidden, 1, 9});
    state[std::string(layer) + "conv_.batch_norm_.weight"] = ones({hidden});
    state[std::string(layer) + "conv_.batch_norm_.bias"] = zeros({hidden});
    state[std::string(layer) + "conv_.batch_norm_.running_mean"] =
        zeros({hidden});
    state[std::string(layer) + "conv_.batch_norm_.running_var"] =
        ones({hidden});
    state[std::string(layer) + "conv_.batch_norm_.num_batches_tracked"] =
        zeros({1});
    add_conv("layers_.0.conv_.pointwise_conv2_", {hidden, hidden, 1});

    add_norm("ffn2_.norm_");
    add_linear("ffn2_.fc1_", ffn_intermediate, hidden);
    add_linear("ffn2_.fc2_", hidden, ffn_intermediate);
    add_int8_linear("ffn2_.fc1_", ffn_intermediate, hidden);
    add_int8_linear("ffn2_.fc2_", hidden, ffn_intermediate);
    add_norm("final_norm_");

    encoder.load_state_dict(state, "", /*strict=*/false);
    encoder.to(DType::Float16);
    encoder.to(Device::GPU);

    const Tensor input = Tensor::ones({1, 32, 80}, DType::Float16, Device::GPU);

    Tensor output;
    EXPECT_NO_THROW(output = encoder.forward(input));
    ASSERT_EQ(output.shape(), Shape({1, 4, hidden}));
    EXPECT_EQ(output.to(Device::CPU).device(), Device::CPU);
    EXPECT_FALSE(encoder.route_stats().direct_qkv_head_layout_used);
#endif
}

TEST(SafeDirectEncoder, ProductionInt8DirectAndGenericAttentionMatch) {
    unset_encoder_experiments();

    const std::filesystem::path fixture(PARAKEET_SAFE_DIRECT_MODEL_FIXTURE);
    if (fixture.empty() || !std::filesystem::is_regular_file(fixture)) {
        GTEST_SKIP() << "Production INT8 FastConformer fixture unavailable: "
                     << fixture;
    }

    ASSERT_TRUE(axiom::system::should_run_gpu_tests())
        << "The configured production fixture test requires Metal";
    const auto weights = axiom::io::safetensors::load(fixture.string());
    const std::string prefix = encoder_prefix(weights);
    const auto config = parakeet::models::make_tdt_600m_config().encoder;
    FastConformerEncoder encoder(config);
    auto prepared = prepare_encoder_weights(weights, prefix);
    encoder.load_state_dict(prepared, prefix, /*strict=*/false);

    constexpr size_t seq_len = 120;
    const size_t hidden = static_cast<size_t>(config.hidden_size);
    const Tensor query =
        Tensor::zeros({1, seq_len, hidden}, DType::Float16, Device::GPU);
    const Tensor position =
        encoder.pos_emb(static_cast<int>(seq_len), config.hidden_size,
                        DType::Float16, Device::GPU);
    const auto &first_block =
        static_cast<const parakeet::models::ConformerBlock &>(
            encoder.layers_[0]);
    const auto &attention = first_block.attn_;
    const auto projection = [](const axiom::nn::Linear &linear) {
        return axiom::ops::Int8Projection{linear.weight(), linear.scale(),
                                          linear.has_bias() ? linear.bias()
                                                            : Tensor()};
    };
    const axiom::ops::Int8QkvProjections projections{
        projection(attention.mha_.q_proj()),
        projection(attention.mha_.k_proj()),
        projection(attention.mha_.v_proj())};
    EXPECT_TRUE(::parakeet::models::detail::can_use_direct_qkv_head_layout(
        query, query, query, projections,
        static_cast<size_t>(attention.mha_.num_heads())));
    EXPECT_FALSE(::parakeet::models::detail::can_use_direct_qkv_head_layout(
        query, query.clone(), query.clone(), projections,
        static_cast<size_t>(attention.mha_.num_heads())));

    // Identical Q/K/V views take Direct-QKV; clones force the established
    // generic projection route. All weights, scales, activations, and biases
    // remain valid GPU tensors in both legs.
    const Tensor direct =
        attention
            .rel_position_attention(query, query, query, position, Tensor())
            .to(Device::CPU)
            .astype(DType::Float32);
    const Tensor generic =
        attention
            .rel_position_attention(query, query.clone(), query.clone(),
                                    position, Tensor())
            .to(Device::CPU)
            .astype(DType::Float32);

    EXPECT_LE(max_abs_delta(direct, generic), 5e-4f);
}

TEST(SafeDirectEncoder, AttentionReloadClearsCachedPositionProjection) {
    using parakeet::models::ConformerAttention;

    ConformerAttention attention(/*num_heads=*/2, /*dropout=*/0.0f);
    const Tensor position = Tensor::ones({3, 4}, DType::Float32);
    std::map<std::string, Tensor> state;
    state["pos_proj_.weight"] = Tensor::ones({4, 4}, DType::Float32);
    attention.load_state_dict(state, "", /*strict=*/false);

    const Tensor first = attention.projected_position_head_layout(position, 2);
    const Tensor cached = attention.projected_position_head_layout(position, 2);
    ASSERT_EQ(attention.position_projection_cache_.size(), 1u);
    EXPECT_FLOAT_EQ(first.item<float>({0, 0, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(cached.item<float>({0, 0, 0, 0}), 4.0f);

    state["pos_proj_.weight"] = Tensor::zeros({4, 4}, DType::Float32);
    attention.load_state_dict(state, "", /*strict=*/false);

    EXPECT_TRUE(attention.position_projection_cache_.empty());
    const Tensor reloaded =
        attention.projected_position_head_layout(position, 2);
    EXPECT_FLOAT_EQ(reloaded.item<float>({0, 0, 0, 0}), 0.0f);
}

TEST(SafeDirectEncoder, RetainsPositionHeadLayoutsAcrossActiveBuckets) {
    unset_encoder_experiments();

    const std::filesystem::path fixture(PARAKEET_SAFE_DIRECT_MODEL_FIXTURE);
    if (fixture.empty() || !std::filesystem::is_regular_file(fixture)) {
        GTEST_SKIP() << "Production INT8 FastConformer fixture unavailable: "
                     << fixture;
    }

    ASSERT_TRUE(axiom::system::should_run_gpu_tests())
        << "The configured production fixture test requires Metal";
    const auto weights = axiom::io::safetensors::load(fixture.string());
    const std::string prefix = encoder_prefix(weights);
    const auto config = parakeet::models::make_tdt_600m_config().encoder;
    FastConformerEncoder encoder(config);
    auto prepared = prepare_encoder_weights(weights, prefix);
    encoder.load_state_dict(prepared, prefix, /*strict=*/false);

    const auto forward = [&](size_t frames) {
        return encoder
            .forward(
                Tensor::zeros({1, frames, 128}, DType::Float16, Device::GPU))
            .to(Device::CPU)
            .astype(DType::Float32);
    };

    static_cast<void>(forward(960));
    static_cast<void>(forward(1152));
    static_cast<void>(forward(960));

    EXPECT_TRUE(encoder.route_stats().cached_position_head_layout_used);
    EXPECT_TRUE(encoder.route_stats().cached_position_head_layout_cache_hit)
        << "returning to a previously seen padded bucket must reuse its "
           "position head-layout projection";
}

} // namespace
