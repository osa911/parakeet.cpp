#include <gtest/gtest.h>

#include <axiom/tensor.hpp>

#include "backends/metal/metal_common.hpp"
#include "backends/metal/metal_workspace_cache.hpp"
#include "parakeet/models/encoder.hpp"
#include "parakeet/models/fastconformer_block_program.hpp"

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace parakeet::models {

// The production program grants this test-only access class friendship without
// exposing an FFN scheduling entry point to callers of the model API.
class FastConformerBlockProgramTestAccess {
  public:
    static auto encode_ffn_pair(const ConformerBlock &block,
                                const axiom::Tensor &input) {
        return FastConformerBlockProgram::encode_ffn_pair_for_testing(block,
                                                                       input);
    }

    static auto encode_attention(const ConformerBlock &block,
                                 const axiom::Tensor &input,
                                 const axiom::Tensor &head_major_position,
                                 const axiom::Tensor &mask) {
        return FastConformerBlockProgram::encode_attention_for_testing(
            block, input, head_major_position, mask);
    }
};

} // namespace parakeet::models

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::Tensor;
using axiom::backends::metal::MetalWorkspaceCache;
using axiom::backends::metal::ScopedMetalWorkspace;
using axiom::backends::metal::WorkspaceBufferStorageMode;
using parakeet::models::ConformerBlock;
using parakeet::models::EncoderConfig;
using parakeet::models::FastConformerBlockProgram;

constexpr size_t kBatch = 1;
constexpr size_t kSequence = 44;
constexpr size_t kHidden = 32;
constexpr size_t kIntermediate = 64;
constexpr size_t kHeads = 4;
constexpr size_t kKernel = 9;
constexpr size_t kAttentionHidden = 128;
constexpr size_t kAttentionHeads = 1;
const Shape kInputShape{kBatch, kSequence, kHidden};

class ScopedEnvironmentVariable {
  public:
    ScopedEnvironmentVariable(const char *name,
                              const std::optional<std::string> &value)
        : name_(name) {
        if (const char *current = std::getenv(name)) {
            previous_ = current;
        }
        if (value.has_value()) {
            setenv(name_.c_str(), value->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ~ScopedEnvironmentVariable() {
        if (previous_.has_value()) {
            setenv(name_.c_str(), previous_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &
    operator=(const ScopedEnvironmentVariable &) = delete;

  private:
    std::string name_;
    std::optional<std::string> previous_;
};

Tensor filled(const Shape &shape, float value) {
    return Tensor::ones(shape, DType::Float32) * value;
}

std::unique_ptr<ConformerBlock> make_block(bool load_int8, Device device,
                                           DType dtype,
                                           size_t hidden = kHidden,
                                           size_t intermediate = kIntermediate,
                                           size_t heads = kHeads) {
    EncoderConfig config;
    config.hidden_size = static_cast<int>(hidden);
    config.ffn_intermediate = static_cast<int>(intermediate);
    config.num_heads = static_cast<int>(heads);
    config.dropout = 0.0f;

    auto block = std::make_unique<ConformerBlock>(config);
    std::map<std::string, Tensor> state;
    const auto norm = [&](const std::string &prefix) {
        state[prefix + "weight"] = Tensor::ones({hidden}, DType::Float32);
        state[prefix + "bias"] = filled({hidden}, 0.03125f);
    };
    const auto linear = [&](const std::string &prefix, size_t output,
                            size_t input) {
        state[prefix + "weight"] = filled({output, input}, 0.015625f);
        state[prefix + "bias"] = filled({output}, 0.0078125f);
    };

    norm("ffn1_.norm_.");
    linear("ffn1_.fc1_.", intermediate, hidden);
    linear("ffn1_.fc2_.", hidden, intermediate);

    norm("attn_.norm_.");
    linear("attn_.mha_.q_proj.", hidden, hidden);
    linear("attn_.mha_.k_proj.", hidden, hidden);
    linear("attn_.mha_.v_proj.", hidden, hidden);
    linear("attn_.mha_.out_proj.", hidden, hidden);
    state["attn_.pos_proj_.weight"] =
        filled({hidden, hidden}, 0.015625f);
    state["attn_.pos_bias_u_"] = filled({heads, hidden / heads}, 0.01f);
    state["attn_.pos_bias_v_"] = filled({heads, hidden / heads}, -0.01f);

    norm("conv_.norm_.");
    state["conv_.pointwise_conv1_.weight"] =
        filled({2 * hidden, hidden, 1}, 0.015625f);
    state["conv_.pointwise_conv1_.bias"] =
        filled({2 * hidden}, 0.0078125f);
    state["conv_.depthwise_conv_.weight"] =
        filled({hidden, 1, kKernel}, 0.015625f);
    state["conv_.depthwise_conv_.bias"] = filled({hidden}, 0.0078125f);
    state["conv_.batch_norm_.weight"] =
        Tensor::ones({hidden}, DType::Float32);
    state["conv_.batch_norm_.bias"] = filled({hidden}, 0.0078125f);
    state["conv_.batch_norm_.running_mean"] =
        Tensor::zeros({hidden}, DType::Float32);
    state["conv_.batch_norm_.running_var"] =
        Tensor::ones({hidden}, DType::Float32);
    state["conv_.batch_norm_.num_batches_tracked"] =
        Tensor::zeros({1}, DType::Float32);
    state["conv_.pointwise_conv2_.weight"] =
        filled({hidden, hidden, 1}, 0.015625f);
    state["conv_.pointwise_conv2_.bias"] = filled({hidden}, 0.0078125f);

    norm("ffn2_.norm_.");
    linear("ffn2_.fc1_.", intermediate, hidden);
    linear("ffn2_.fc2_.", hidden, intermediate);
    norm("final_norm_.");

    block->load_state_dict(state, "", /*strict=*/false);

    if (load_int8) {
        const auto int8_weight = [](size_t output, size_t input) {
            return Tensor::ones({output, input}, DType::Int8);
        };
        const auto scale = [](size_t output, size_t input) {
            return filled({output, input / 32}, 0.015625f)
                .astype(DType::Float16);
        };
        block->load_int8_weights(
            int8_weight(hidden, hidden), scale(hidden, hidden),
            int8_weight(hidden, hidden), scale(hidden, hidden),
            int8_weight(hidden, hidden), scale(hidden, hidden),
            int8_weight(hidden, hidden), scale(hidden, hidden),
            int8_weight(intermediate, hidden), scale(intermediate, hidden),
            int8_weight(hidden, intermediate), scale(hidden, intermediate),
            int8_weight(intermediate, hidden), scale(intermediate, hidden),
            int8_weight(hidden, intermediate), scale(hidden, intermediate));
        block->load_int8_pointwise_conv_weights(
            int8_weight(2 * hidden, hidden), scale(2 * hidden, hidden),
            int8_weight(hidden, hidden), scale(hidden, hidden));
    }

    block->to(dtype);
    block->to(device);
    return block;
}

Tensor input_on(Device device, DType dtype = DType::Float16,
                size_t sequence = kSequence, size_t hidden = kHidden) {
    return filled({kBatch, sequence, hidden}, 0.125f)
        .astype(dtype)
        .to(device);
}

Tensor position_on(Device device, DType dtype = DType::Float16,
                   size_t sequence = kSequence, size_t hidden = kHidden) {
    return filled({2 * sequence - 1, hidden}, 0.0625f)
        .astype(dtype)
        .to(device);
}

Tensor head_major_position_on(Device device, size_t heads,
                              size_t hidden,
                              DType dtype = DType::Float16,
                              size_t sequence = kSequence) {
    return filled({1, heads, 2 * sequence - 1, hidden / heads}, 0.0625f)
        .astype(dtype)
        .to(device);
}

class FastConformerBlockProgramTest : public ::testing::Test {
  protected:
    void SetUp() override {
        block_program_flag_ = std::make_unique<ScopedEnvironmentVariable>(
            "PARAKEET_FASTCONFORMER_BLOCK_PROGRAM", std::string("1"));
        FastConformerBlockProgram::reset_trace_for_testing();
    }

    void TearDown() override { block_program_flag_.reset(); }

  private:
    std::unique_ptr<ScopedEnvironmentVariable> block_program_flag_;
};

TEST_F(FastConformerBlockProgramTest, DeclinesWhenFlagIsDisabled) {
    ScopedEnvironmentVariable disabled("PARAKEET_FASTCONFORMER_BLOCK_PROGRAM",
                                       std::nullopt);
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input_on(Device::GPU), position_on(Device::GPU), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest, DeclinesCpuInput) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input_on(Device::CPU), position_on(Device::GPU), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest, DeclinesNonFloat16Input) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input_on(Device::GPU, DType::Float32),
        position_on(Device::GPU), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest, DeclinesNonContiguousInput) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);
    const Tensor input = filled({kBatch, kHidden, kSequence}, 0.125f)
                             .astype(DType::Float16)
                             .to(Device::GPU)
                             .transpose({0, 2, 1});
    ASSERT_EQ(input.shape(), kInputShape);
    ASSERT_FALSE(input.is_contiguous());

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input, position_on(Device::GPU), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest, DeclinesNonInt8Block) {
    const auto block = make_block(/*load_int8=*/false, Device::GPU,
                                  DType::Float16);

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input_on(Device::GPU), position_on(Device::GPU), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest, DeclinesWithoutPrivateWorkspace) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input_on(Device::GPU), position_on(Device::GPU), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest, DeclinesUnsupportedSequenceBucket) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);
    MetalWorkspaceCache workspace(8 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input_on(Device::GPU, DType::Float16, kSequence + 1),
        position_on(Device::GPU, DType::Float16, kSequence + 1), Tensor()));
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
    scope.close();
}

TEST_F(FastConformerBlockProgramTest,
       DeclinesUntilEverySuppliedOutputPreconditionExists) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);
    const Tensor input = input_on(Device::GPU);
    const Tensor position = position_on(Device::GPU);
    MetalWorkspaceCache workspace(8 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);
    const auto before = workspace.snapshot();

    EXPECT_FALSE(FastConformerBlockProgram::is_supported(
        *block, input, position, Tensor()));
    EXPECT_THROW(static_cast<void>(FastConformerBlockProgram::encode(
                     *block, input, position, Tensor())),
                 std::exception);

    const auto after = workspace.snapshot();
    EXPECT_EQ(after.allocation_requests, before.allocation_requests)
        << "An unfinished program must not acquire HybridWorkspaceSlots";
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
    scope.close();
}

#ifdef AXIOM_METAL_SUPPORT
TEST_F(FastConformerBlockProgramTest,
       EnabledButUnfinishedProgramPreservesExistingForwardBody) {
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);
    const Tensor input = input_on(Device::GPU);
    const Tensor position = position_on(Device::GPU);

    const auto run_forward = [&](bool enable_program) {
        ScopedEnvironmentVariable flag(
            "PARAKEET_FASTCONFORMER_BLOCK_PROGRAM",
            enable_program ? std::optional<std::string>("1") : std::nullopt);
        MetalWorkspaceCache workspace(32 << 20,
                                      WorkspaceBufferStorageMode::Private);
        ScopedMetalWorkspace scope(workspace);
        Tensor result_cpu;
        {
            Tensor result = block->forward(input, position, Tensor());
            EXPECT_EQ(result.shape(), kInputShape);
            result_cpu = result.to(Device::CPU);
        }
        scope.close();
        return result_cpu;
    };

    const Tensor control = run_forward(/*enable_program=*/false);
    const Tensor flagged = run_forward(/*enable_program=*/true);

    ASSERT_EQ(flagged.shape(), control.shape());
    ASSERT_EQ(flagged.dtype(), control.dtype());
    EXPECT_EQ(std::memcmp(flagged.typed_data<uint16_t>(),
                          control.typed_data<uint16_t>(),
                          flagged.numel() * sizeof(uint16_t)),
              0);
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
}

TEST_F(FastConformerBlockProgramTest,
       FullDirectScheduleMatchesWarmCachedBlockOutput) {
    ScopedEnvironmentVariable direct_ffn("PARAKEET_DIRECT_INT8_FFN",
                                         std::string("1"));
    ScopedEnvironmentVariable direct_layer_norm("PARAKEET_DIRECT_F16_LAYERNORM",
                                                std::string("1"));
    ScopedEnvironmentVariable fused_pointwise(
        "PARAKEET_FUSED_INT8_POINTWISE_GLU", std::string("1"));
    ScopedEnvironmentVariable direct_depthwise(
        "PARAKEET_DIRECT_DEPTHWISE_CONV_BN_SILU", std::string("1"));
    ScopedEnvironmentVariable direct_residual("PARAKEET_DIRECT_INT8_RESIDUAL",
                                              std::string("1"));
    ScopedEnvironmentVariable direct_qkv("PARAKEET_DIRECT_INT8_QKV",
                                         std::string("1"));
    ScopedEnvironmentVariable direct_qkv_head_layout(
        "PARAKEET_DIRECT_INT8_QKV_HEAD_LAYOUT", std::string("1"));
    ScopedEnvironmentVariable direct_attention(
        "PARAKEET_RELATIVE_POSITION_ATTENTION", std::string("1"));
    ScopedEnvironmentVariable position_cache(
        "PARAKEET_CACHE_POSITION_PROJECTIONS", std::string("1"));
    ScopedEnvironmentVariable position_cache_head_layout(
        "PARAKEET_CACHE_POSITION_HEAD_LAYOUT", std::string("1"));

    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16, kAttentionHidden,
                                  kAttentionHidden, kAttentionHeads);
    const Tensor input = input_on(Device::GPU, DType::Float16, kSequence,
                                  kAttentionHidden);
    const Tensor position = position_on(Device::GPU, DType::Float16,
                                        kSequence, kAttentionHidden);
    MetalWorkspaceCache workspace(64 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);

    Tensor control_cpu;
    {
        ScopedEnvironmentVariable disable_program(
            "PARAKEET_FASTCONFORMER_BLOCK_PROGRAM", std::nullopt);
        control_cpu = block->forward(input, position, Tensor()).to(Device::CPU);
    }

    ASSERT_TRUE(FastConformerBlockProgram::is_supported(
        *block, input, position, Tensor()));
    const Tensor candidate_cpu =
        block->forward(input, position, Tensor()).to(Device::CPU);

    ASSERT_EQ(candidate_cpu.shape(), control_cpu.shape());
    ASSERT_EQ(candidate_cpu.dtype(), control_cpu.dtype());
    EXPECT_EQ(std::memcmp(candidate_cpu.typed_data<uint16_t>(),
                          control_cpu.typed_data<uint16_t>(),
                          candidate_cpu.nbytes()),
              0);
    EXPECT_GT(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
    scope.close();
}

TEST_F(FastConformerBlockProgramTest,
       FfnSegmentsPreserveReferenceAttentionWithoutNativeAttentionKernel) {
    ScopedEnvironmentVariable direct_ffn("PARAKEET_DIRECT_INT8_FFN",
                                         std::string("1"));
    ScopedEnvironmentVariable direct_layer_norm("PARAKEET_DIRECT_F16_LAYERNORM",
                                                std::string("1"));
    ScopedEnvironmentVariable direct_qkv("PARAKEET_DIRECT_INT8_QKV",
                                         std::nullopt);
    ScopedEnvironmentVariable direct_qkv_head_layout(
        "PARAKEET_DIRECT_INT8_QKV_HEAD_LAYOUT", std::nullopt);
    ScopedEnvironmentVariable position_cache("PARAKEET_CACHE_POSITION_PROJECTIONS",
                                             std::nullopt);
    ScopedEnvironmentVariable position_cache_head_layout(
        "PARAKEET_CACHE_POSITION_HEAD_LAYOUT", std::nullopt);
    ScopedEnvironmentVariable native_relative_attention(
        "PARAKEET_RELATIVE_POSITION_ATTENTION", std::nullopt);

    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16, kAttentionHidden,
                                  kAttentionHidden, kAttentionHeads);
    const Tensor input = input_on(Device::GPU, DType::Float16, kSequence,
                                  kAttentionHidden);
    const Tensor position = position_on(Device::GPU, DType::Float16,
                                        kSequence, kAttentionHidden);
    MetalWorkspaceCache workspace(64 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);

    Tensor control_cpu;
    {
        ScopedEnvironmentVariable disable_program(
            "PARAKEET_FASTCONFORMER_BLOCK_PROGRAM", std::nullopt);
        control_cpu = block->forward(input, position, Tensor()).to(Device::CPU);
    }

    ASSERT_TRUE(FastConformerBlockProgram::is_supported(
        *block, input, position, Tensor()));
    const Tensor candidate_cpu =
        block->forward(input, position, Tensor()).to(Device::CPU);

    ASSERT_EQ(candidate_cpu.shape(), control_cpu.shape());
    ASSERT_EQ(candidate_cpu.dtype(), control_cpu.dtype());
    EXPECT_EQ(std::memcmp(candidate_cpu.typed_data<uint16_t>(),
                          control_cpu.typed_data<uint16_t>(),
                          candidate_cpu.nbytes()),
              0);
    EXPECT_GT(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
    scope.close();
}

TEST_F(FastConformerBlockProgramTest,
       DirectFfnPairReusesSlotsAndKeepsFirstResultAlive) {
    ScopedEnvironmentVariable direct_ffn("PARAKEET_DIRECT_INT8_FFN",
                                         std::string("1"));
    ScopedEnvironmentVariable direct_layer_norm("PARAKEET_DIRECT_F16_LAYERNORM",
                                                std::string("1"));
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16);
    const Tensor input = input_on(Device::GPU);
    MetalWorkspaceCache workspace(32 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);

    {
        const auto [control_first, control_second, candidate_first,
                    candidate_second, logical_values, slot_count,
                    cpu_synchronizations] =
            parakeet::models::FastConformerBlockProgramTestAccess::encode_ffn_pair(
                *block, input);

        auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
        const uint64_t before_cpu_copy = stream.synchronization_count();
        const Tensor control_first_cpu = control_first.to(Device::CPU);
        EXPECT_EQ(stream.synchronization_count(), before_cpu_copy + 1)
            << "The test hook must observe the normal CPU copy sync";
        const Tensor control_second_cpu = control_second.to(Device::CPU);
        const Tensor candidate_first_cpu = candidate_first.to(Device::CPU);
        const Tensor candidate_second_cpu = candidate_second.to(Device::CPU);

        EXPECT_EQ(std::memcmp(control_first_cpu.typed_data<uint16_t>(),
                              candidate_first_cpu.typed_data<uint16_t>(),
                              control_first_cpu.nbytes()),
                  0);
        EXPECT_EQ(std::memcmp(control_second_cpu.typed_data<uint16_t>(),
                              candidate_second_cpu.typed_data<uint16_t>(),
                              control_second_cpu.nbytes()),
                  0);
        EXPECT_NE(candidate_first.storage().get(),
                  candidate_second.storage().get())
            << "FFN1 must stay live while FFN2 reads it";
        EXPECT_EQ(logical_values, 6U);
        EXPECT_LT(slot_count, logical_values);
        EXPECT_EQ(cpu_synchronizations, 0U);
        EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(),
                  4U);
    }
    scope.close();
}

TEST_F(FastConformerBlockProgramTest,
       DirectFfnPairRejectsBeforeWorkspaceAllocation) {
    const auto block = make_block(/*load_int8=*/false, Device::GPU,
                                  DType::Float16);
    const Tensor input = input_on(Device::GPU);
    MetalWorkspaceCache workspace(32 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);
    const auto before = workspace.snapshot();

    EXPECT_THROW(
        static_cast<void>(
            parakeet::models::FastConformerBlockProgramTestAccess::encode_ffn_pair(
                *block, input)),
        std::exception);

    const auto after = workspace.snapshot();
    EXPECT_EQ(after.allocation_requests, before.allocation_requests);
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
    scope.close();
}

TEST_F(FastConformerBlockProgramTest,
       DirectAttentionKeepsQkvLiveAndReusesNormalizationStorage) {
    ScopedEnvironmentVariable compare_reference(
        "PARAKEET_RELATIVE_POSITION_ATTENTION_COMPARE", std::nullopt);
    const auto block = make_block(/*load_int8=*/true, Device::GPU,
                                  DType::Float16, kAttentionHidden,
                                  kAttentionHidden, kAttentionHeads);
    const Tensor input = input_on(Device::GPU, DType::Float16, kSequence,
                                  kAttentionHidden);
    const Tensor position = head_major_position_on(
        Device::GPU, kAttentionHeads, kAttentionHidden);
    MetalWorkspaceCache workspace(32 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);

    {
        const auto [control_q, control_k, control_v, control_attention,
                    candidate_normalized, candidate_q, candidate_k, candidate_v,
                    candidate_attention, logical_values, slot_count,
                    cpu_synchronizations] =
            parakeet::models::FastConformerBlockProgramTestAccess::encode_attention(
                *block, input, position, Tensor());

        auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
        const uint64_t before_cpu_copy = stream.synchronization_count();
        const Tensor control_q_cpu = control_q.to(Device::CPU);
        EXPECT_EQ(stream.synchronization_count(), before_cpu_copy + 1)
            << "The test hook must observe the normal CPU copy sync";
        const Tensor control_k_cpu = control_k.to(Device::CPU);
        const Tensor control_v_cpu = control_v.to(Device::CPU);
        const Tensor control_attention_cpu = control_attention.to(Device::CPU);
        const Tensor candidate_q_cpu = candidate_q.to(Device::CPU);
        const Tensor candidate_k_cpu = candidate_k.to(Device::CPU);
        const Tensor candidate_v_cpu = candidate_v.to(Device::CPU);
        const Tensor candidate_attention_cpu = candidate_attention.to(Device::CPU);

        EXPECT_EQ(std::memcmp(control_q_cpu.typed_data<uint16_t>(),
                              candidate_q_cpu.typed_data<uint16_t>(),
                              control_q_cpu.nbytes()),
                  0);
        EXPECT_EQ(std::memcmp(control_k_cpu.typed_data<uint16_t>(),
                              candidate_k_cpu.typed_data<uint16_t>(),
                              control_k_cpu.nbytes()),
                  0);
        EXPECT_EQ(std::memcmp(control_v_cpu.typed_data<uint16_t>(),
                              candidate_v_cpu.typed_data<uint16_t>(),
                              control_v_cpu.nbytes()),
                  0);
        EXPECT_EQ(std::memcmp(control_attention_cpu.typed_data<uint16_t>(),
                              candidate_attention_cpu.typed_data<uint16_t>(),
                              control_attention_cpu.nbytes()),
                  0);
        EXPECT_NE(candidate_q.storage().get(), candidate_k.storage().get());
        EXPECT_NE(candidate_q.storage().get(), candidate_v.storage().get());
        EXPECT_NE(candidate_k.storage().get(), candidate_v.storage().get());
        EXPECT_TRUE(candidate_normalized.shares_storage(candidate_attention));
        EXPECT_NE(candidate_q.storage().get(), candidate_attention.storage().get());
        EXPECT_NE(candidate_k.storage().get(), candidate_attention.storage().get());
        EXPECT_NE(candidate_v.storage().get(), candidate_attention.storage().get());
        EXPECT_EQ(logical_values, 5U);
        EXPECT_LT(slot_count, logical_values);
        EXPECT_EQ(cpu_synchronizations, 0U);
        EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(),
                  3U);
    }
    scope.close();
}

TEST_F(FastConformerBlockProgramTest,
       DirectAttentionRejectsBeforeWorkspaceAllocation) {
    const auto non_int8_block = make_block(
        /*load_int8=*/false, Device::GPU, DType::Float16, kAttentionHidden,
        kAttentionHidden, kAttentionHeads);
    const auto int8_block = make_block(
        /*load_int8=*/true, Device::GPU, DType::Float16, kAttentionHidden,
        kAttentionHidden, kAttentionHeads);
    const Tensor input = input_on(Device::GPU, DType::Float16, kSequence,
                                  kAttentionHidden);
    const Tensor position = head_major_position_on(
        Device::GPU, kAttentionHeads, kAttentionHidden);
    const Tensor non_direct_position = head_major_position_on(
        Device::GPU, kAttentionHeads, kAttentionHidden, DType::Float32);
    MetalWorkspaceCache workspace(32 << 20,
                                  WorkspaceBufferStorageMode::Private);
    ScopedMetalWorkspace scope(workspace);
    const auto before = workspace.snapshot();

    EXPECT_THROW(
        static_cast<void>(
            parakeet::models::FastConformerBlockProgramTestAccess::encode_attention(
                *non_int8_block, input, position, Tensor())),
        std::exception);
    EXPECT_THROW(
        static_cast<void>(
            parakeet::models::FastConformerBlockProgramTestAccess::encode_attention(
                *int8_block, input, non_direct_position, Tensor())),
        std::exception);

    const auto after = workspace.snapshot();
    EXPECT_EQ(after.allocation_requests, before.allocation_requests);
    EXPECT_EQ(FastConformerBlockProgram::direct_dispatches_for_testing(), 0U);
    scope.close();
}
#endif

} // namespace
