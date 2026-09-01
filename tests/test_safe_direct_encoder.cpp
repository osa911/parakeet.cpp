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
#include "parakeet/models/encoder.hpp"

#ifndef PARAKEET_SAFE_DIRECT_MODEL_FIXTURE
#define PARAKEET_SAFE_DIRECT_MODEL_FIXTURE ""
#endif

extern char **environ;

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Tensor;
using parakeet::models::EncoderExecutionConfig;
using parakeet::models::EncoderRouteStats;
using parakeet::models::EncoderWorkspaceMode;
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

bool is_generic_control_scale(std::string_view name) {
    return name.ends_with("attn_.mha_.q_proj_scale") ||
           name.ends_with("ffn1_.fc1__scale") ||
           name.ends_with("ffn2_.fc1__scale");
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
                        std::string_view prefix, bool generic_control) {
    std::map<std::string, Tensor> prepared;
    for (const auto &[name, source] : weights) {
        if (!belongs_to_encoder(name, prefix)) {
            continue;
        }
        Tensor value = source;
        if (is_float_dtype(value.dtype()) && value.dtype() != DType::Float16) {
            value = value.astype(DType::Float16);
        }
        if (!(generic_control && is_generic_control_scale(name))) {
            value = value.to(Device::GPU);
        }
        prepared.emplace(name, std::move(value));
    }
    return prepared;
}

struct EncoderRun {
    Tensor output;
    EncoderRouteStats stats;

    const Tensor &encoder_output() const { return output; }
    const EncoderRouteStats &route_stats() const { return stats; }
};

EncoderRun run_encoder(const std::map<std::string, Tensor> &weights,
                       std::string_view prefix,
                       EncoderWorkspaceMode workspace_mode,
                       bool generic_control, bool repeat_forward = false) {
    const std::string run_name = generic_control ? "generic control"
                                 : workspace_mode == EncoderWorkspaceMode::Boost
                                     ? "Boost"
                                     : "LowerMemory";
    SCOPED_TRACE(run_name);
    const auto config = parakeet::models::make_tdt_600m_config().encoder;
    const EncoderExecutionConfig execution{.workspace_mode = workspace_mode};
    FastConformerEncoder encoder(config, execution);
    auto prepared = prepare_encoder_weights(weights, prefix, generic_control);
    encoder.load_state_dict(prepared, std::string(prefix), /*strict=*/false);

    const Tensor input =
        Tensor::zeros({1, 960, 128}, DType::Float16, Device::GPU);
    try {
        Tensor output =
            encoder.forward(input).to(Device::CPU).astype(DType::Float32);
        if (repeat_forward) {
            output =
                encoder.forward(input).to(Device::CPU).astype(DType::Float32);
        }
        return {output, encoder.route_stats()};
    } catch (const std::exception &error) {
        throw std::runtime_error(run_name + ": " + error.what());
    }
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

TEST(SafeDirectEncoder, ProductionInt8RoutesMatchGenericControl) {
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

    const EncoderRun control =
        run_encoder(weights, prefix, EncoderWorkspaceMode::LowerMemory,
                    /*generic_control=*/true);
    const EncoderRun lower_memory =
        run_encoder(weights, prefix, EncoderWorkspaceMode::LowerMemory,
                    /*generic_control=*/false);
    const EncoderRun boost =
        run_encoder(weights, prefix, EncoderWorkspaceMode::Boost,
                    /*generic_control=*/false, /*repeat_forward=*/true);

    EXPECT_LE(
        max_abs_delta(lower_memory.encoder_output(), control.encoder_output()),
        5e-4f);
    EXPECT_LE(max_abs_delta(boost.encoder_output(), control.encoder_output()),
              5e-4f);
    EXPECT_LE(
        max_abs_delta(lower_memory.encoder_output(), boost.encoder_output()),
        5e-4f);

    EXPECT_TRUE(lower_memory.route_stats().direct_qkv_head_layout_used);
    EXPECT_TRUE(lower_memory.route_stats().direct_silu_used);
    EXPECT_TRUE(lower_memory.route_stats().cached_position_head_layout_used);
    EXPECT_FALSE(lower_memory.route_stats().direct_residual_used);
    EXPECT_FALSE(lower_memory.route_stats().fused_pointwise_glu_used);
    EXPECT_FALSE(lower_memory.route_stats().bounded_workspace_used);
    EXPECT_FALSE(lower_memory.route_stats().process_wide_workspace_used);

    EXPECT_TRUE(boost.route_stats().direct_qkv_head_layout_used);
    EXPECT_TRUE(boost.route_stats().direct_silu_used);
    EXPECT_TRUE(boost.route_stats().cached_position_head_layout_used);
    EXPECT_FALSE(boost.route_stats().direct_residual_used);
    EXPECT_FALSE(boost.route_stats().fused_pointwise_glu_used);
    EXPECT_TRUE(boost.route_stats().bounded_workspace_used);
    EXPECT_TRUE(boost.route_stats().process_wide_workspace_used)
        << "Boost must select the process-wide serialized workspace pool so the "
           "server memory bound holds across HTTP worker threads";
}

} // namespace
