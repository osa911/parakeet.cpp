#include <gtest/gtest.h>

#include <axiom/graph/graph_registry.hpp>
#include <axiom/operations.hpp>
#include <axiom/tensor.hpp>

#include "backends/metal/metal_operations.hpp"
#include "backends/metal/metal_buffer_provider.hpp"
#include "backends/metal/metal_common.hpp"
#include "backends/metal/metal_workspace_cache.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::Tensor;
using axiom::backends::metal::MetalWorkspaceCache;
using axiom::backends::metal::ScopedMetalWorkspace;
using axiom::backends::metal::WorkspaceBufferStorageMode;

class ScopedEnvironmentVariable {
  public:
    ScopedEnvironmentVariable(const char *name, const char *value) : name_(name) {
        const char *previous = std::getenv(name_);
        if (previous) previous_ = previous;
        setenv(name_, value, /* overwrite= */ 1);
    }

    ~ScopedEnvironmentVariable() {
        if (previous_) {
            setenv(name_, previous_->c_str(), /* overwrite= */ 1);
        } else {
            unsetenv(name_);
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) =
        delete;

  private:
    const char *name_;
    std::optional<std::string> previous_;
};

Tensor gpu_fp16(const Shape &shape, uint16_t value) {
    Tensor tensor = Tensor::zeros(shape, DType::Float16, Device::CPU);
    auto *values = tensor.typed_data<uint16_t>();
    for (size_t index = 0; index < tensor.numel(); ++index) {
        values[index] = value;
    }
    return tensor.to(Device::GPU);
}

Tensor gpu_int8(const Shape &shape, int8_t value) {
    Tensor tensor = Tensor::zeros(shape, DType::Int8, Device::CPU);
    auto *values = tensor.typed_data<int8_t>();
    for (size_t index = 0; index < tensor.numel(); ++index) {
        values[index] = value;
    }
    return tensor.to(Device::GPU);
}

TEST(HybridMetalGraph,
     ContiguousCopyUsesOnlyItsGatherDestinationInPrivateWorkspace) {
    Tensor source = Tensor::zeros({2, 3}, DType::Float16, Device::CPU);
    const uint16_t source_values[] = {0x3c00, 0x4000, 0x4200,
                                      0x4400, 0x4500, 0x4600};
    std::memcpy(source.typed_data<uint16_t>(), source_values,
                sizeof(source_values));
    const Tensor strided = source.to(Device::GPU).permute({1, 0});

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    Tensor actual;
    {
        ScopedMetalWorkspace scope(workspace);
        {
            const Tensor contiguous = strided.ascontiguousarray();
            actual = contiguous.to(Device::CPU);
        }
        scope.close();
    }

    const uint16_t expected_values[] = {0x3c00, 0x4400, 0x4000,
                                        0x4500, 0x4200, 0x4600};
    ASSERT_EQ(actual.shape(), Shape({3, 2}));
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(), expected_values,
                          sizeof(expected_values)),
              0);
    EXPECT_EQ(workspace.snapshot().fresh_allocations, 1U)
        << "A GPU contiguous copy must allocate only the gather destination";
}

TEST(HybridMetalGraph, DefersEligibleInt8ProjectionOnlyInsideScope) {
    constexpr size_t kRows = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager = axiom::ops::int8_matmul(activation, weight, scale);
    EXPECT_FALSE(eager.is_lazy());
    EXPECT_EQ(eager.shape(), Shape({kRows, kOutputFeatures}));

    Tensor deferred;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        deferred = axiom::ops::int8_matmul(activation, weight, scale);
    }

    ASSERT_TRUE(deferred.is_lazy());
    ASSERT_NE(deferred.lazy_node(), nullptr);
    EXPECT_EQ(deferred.lazy_node()->op_type, axiom::ops::OpType::Int8MatMul);
    EXPECT_EQ(deferred.lazy_node()->inputs.size(), 3U);
    EXPECT_EQ(deferred.shape(), Shape({kRows, kOutputFeatures}));
    EXPECT_EQ(deferred.dtype(), DType::Float16);
    EXPECT_EQ(deferred.device(), Device::GPU);
}

TEST(HybridMetalGraph, DefersAndMaterializesBiasedProjection) {
    constexpr size_t kRows = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias =
        gpu_fp16({kOutputFeatures}, /* 1.0f */ 0x3c00);

    const Tensor expected =
        axiom::ops::int8_matmul_bias(activation, weight, scale, bias);
    const Tensor expected_cpu = expected.to(Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor deferred;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        deferred = axiom::ops::int8_matmul_bias(activation, weight, scale, bias);
    }

    ASSERT_TRUE(deferred.is_lazy());
    ASSERT_NE(deferred.lazy_node(), nullptr);
    EXPECT_EQ(deferred.lazy_node()->inputs.size(), 4U);
    const Tensor actual_cpu = deferred.to(Device::CPU);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.direct_int8_steps, 1U);
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.typed_data<uint16_t>(),
                          expected_cpu.typed_data<uint16_t>(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, DefersAndMaterializesGroupedHeadLayoutQkv) {
    constexpr size_t kBatch = 1;
    constexpr size_t kTime = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;
    constexpr size_t kHeads = 2;

    const Tensor activation =
        gpu_fp16({kBatch, kTime, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias = gpu_fp16({kOutputFeatures}, /* 0.0f */ 0x0000);

    const auto expected = axiom::ops::int8_qkv_matmul_bias_head_layout(
        activation, weight, scale, bias, weight, scale, bias, weight, scale,
        bias, kHeads);
    std::array<Tensor, 3> expected_cpu;
    for (size_t index = 0; index < expected.size(); ++index) {
        expected_cpu[index] = expected[index].to(Device::CPU);
    }

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    std::array<Tensor, 3> deferred;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        deferred = axiom::ops::int8_qkv_matmul_bias_head_layout(
            activation, weight, scale, bias, weight, scale, bias, weight,
            scale, bias, kHeads);
    }

    for (const Tensor &output : deferred) {
        ASSERT_TRUE(output.is_lazy());
        ASSERT_NE(output.lazy_node(), nullptr);
        EXPECT_EQ(output.lazy_node()->op_type, axiom::ops::OpType::Int8Qkv);
    }
    EXPECT_EQ(deferred[0].lazy_node()->direct_qkv_group,
              deferred[1].lazy_node()->direct_qkv_group);
    EXPECT_EQ(deferred[0].lazy_node()->direct_qkv_group,
              deferred[2].lazy_node()->direct_qkv_group);

    const Tensor first_actual_cpu = deferred[0].to(Device::CPU);
    std::array<Tensor, 3> actual_cpu{first_actual_cpu, deferred[1].to(Device::CPU),
                                     deferred[2].to(Device::CPU)};
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.direct_int8_qkv_steps, 1U);

    EXPECT_NE(deferred[0].storage(), deferred[1].storage());
    EXPECT_NE(deferred[0].storage(), deferred[2].storage());
    EXPECT_NE(deferred[1].storage(), deferred[2].storage());
    for (size_t index = 0; index < actual_cpu.size(); ++index) {
        ASSERT_EQ(actual_cpu[index].shape(), expected_cpu[index].shape());
        ASSERT_EQ(actual_cpu[index].numel(), expected_cpu[index].numel());
        EXPECT_EQ(std::memcmp(actual_cpu[index].typed_data<uint16_t>(),
                              expected_cpu[index].typed_data<uint16_t>(),
                              actual_cpu[index].numel() * sizeof(uint16_t)),
                  0);
    }
}

TEST(HybridMetalGraph, MaterializesGroupedQkvBeforeMpsGraphConsumer) {
    constexpr size_t kBatch = 1;
    constexpr size_t kTime = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kBatch, kTime, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias = gpu_fp16({kOutputFeatures}, /* 0.0f */ 0x0000);

    const auto eager_qkv = axiom::ops::int8_qkv_matmul_bias(
        activation, weight, scale, bias, weight, scale, bias, weight, scale,
        bias);
    const Tensor expected =
        axiom::ops::add(eager_qkv[0], eager_qkv[1]).to(Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor deferred_consumer;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        const auto qkv = axiom::ops::int8_qkv_matmul_bias(
            activation, weight, scale, bias, weight, scale, bias, weight,
            scale, bias);
        ASSERT_TRUE(qkv[0].is_lazy());
        ASSERT_TRUE(qkv[1].is_lazy());
        ASSERT_TRUE(qkv[2].is_lazy());
        deferred_consumer = axiom::ops::add(qkv[0], qkv[1]);
    }

    const Tensor actual = deferred_consumer.to(Device::CPU);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.direct_int8_qkv_steps, 1U);
    EXPECT_EQ(stats.mpsgraph_islands, 1U);
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          actual.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, MaterializesGroupedQkvWithEmptyOptionalBiases) {
    constexpr size_t kBatch = 1;
    constexpr size_t kTime = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kBatch, kTime, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor no_bias;

    const auto expected = axiom::ops::int8_qkv_matmul_bias(
        activation, weight, scale, no_bias, weight, scale, no_bias, weight,
        scale, no_bias);
    std::array<Tensor, 3> expected_cpu;
    for (size_t index = 0; index < expected.size(); ++index) {
        expected_cpu[index] = expected[index].to(Device::CPU);
    }

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    std::array<Tensor, 3> deferred;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        deferred = axiom::ops::int8_qkv_matmul_bias(
            activation, weight, scale, no_bias, weight, scale, no_bias,
            weight, scale, no_bias);
    }

    for (size_t index = 0; index < deferred.size(); ++index) {
        const Tensor actual_cpu = deferred[index].to(Device::CPU);
        ASSERT_EQ(actual_cpu.shape(), expected_cpu[index].shape());
        ASSERT_EQ(actual_cpu.numel(), expected_cpu[index].numel());
        EXPECT_EQ(std::memcmp(actual_cpu.typed_data<uint16_t>(),
                              expected_cpu[index].typed_data<uint16_t>(),
                              actual_cpu.numel() * sizeof(uint16_t)),
                  0);
    }
    EXPECT_EQ(axiom::graph::hybrid_metal_graph_execution_stats()
                  .direct_int8_qkv_steps,
              1U);
}

TEST(HybridMetalGraph, ExecutesGroupedQkvInOrderedProgram) {
    constexpr size_t kBatch = 1;
    constexpr size_t kTime = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kBatch, kTime, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias = gpu_fp16({kOutputFeatures}, /* 0.0f */ 0x0000);

    const auto eager_qkv = axiom::ops::int8_qkv_matmul_bias(
        activation, weight, scale, bias, weight, scale, bias, weight, scale,
        bias);
    const Tensor expected = axiom::ops::add(
        axiom::ops::add(eager_qkv[0], eager_qkv[1]), eager_qkv[2]).to(
        Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    Tensor deferred_consumer;
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        axiom::graph::ScopedHybridMetalProgramExecution program_scope;
        const auto qkv = axiom::ops::int8_qkv_matmul_bias(
            activation, weight, scale, bias, weight, scale, bias, weight,
            scale, bias);
        ASSERT_TRUE(qkv[0].is_lazy());
        ASSERT_TRUE(qkv[1].is_lazy());
        ASSERT_TRUE(qkv[2].is_lazy());
        deferred_consumer = axiom::ops::add(
            axiom::ops::add(qkv[0], qkv[1]), qkv[2]);
        actual = deferred_consumer.to(Device::CPU);
    }

    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.direct_int8_qkv_steps, 1U);
    EXPECT_EQ(stats.mpsgraph_islands, 1U);
    EXPECT_EQ(stats.ordered_program_microprograms, 1U);
    EXPECT_EQ(stats.logical_temporary_values, 3U);
    EXPECT_EQ(stats.temporary_slots, 3U);
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          actual.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, ExecutesGroupedHeadLayoutQkvInOrderedProgram) {
    constexpr size_t kBatch = 1;
    constexpr size_t kTime = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;
    constexpr size_t kHeads = 2;

    const Tensor activation =
        gpu_fp16({kBatch, kTime, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias = gpu_fp16({kOutputFeatures}, /* 0.0f */ 0x0000);

    const auto eager_qkv = axiom::ops::int8_qkv_matmul_bias_head_layout(
        activation, weight, scale, bias, weight, scale, bias, weight, scale,
        bias, kHeads);
    const Tensor expected = axiom::ops::add(
        axiom::ops::add(eager_qkv[0], eager_qkv[1]), eager_qkv[2]).to(
        Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        axiom::graph::ScopedHybridMetalProgramExecution program_scope;
        const auto qkv = axiom::ops::int8_qkv_matmul_bias_head_layout(
            activation, weight, scale, bias, weight, scale, bias, weight,
            scale, bias, kHeads);
        ASSERT_TRUE(qkv[0].is_lazy());
        ASSERT_TRUE(qkv[1].is_lazy());
        ASSERT_TRUE(qkv[2].is_lazy());
        actual = axiom::ops::add(
            axiom::ops::add(qkv[0], qkv[1]), qkv[2]).to(Device::CPU);
    }

    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.direct_int8_qkv_steps, 1U);
    EXPECT_EQ(stats.mpsgraph_islands, 1U);
    EXPECT_EQ(stats.ordered_program_microprograms, 1U);
    EXPECT_EQ(stats.logical_temporary_values, 3U);
    EXPECT_EQ(stats.temporary_slots, 3U);
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          actual.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, MaterializesRankThreeBiasedProjection) {
    constexpr size_t kBatch = 1;
    constexpr size_t kTime = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation = gpu_fp16(
        {kBatch, kTime, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias =
        gpu_fp16({kOutputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor expected =
        axiom::ops::int8_matmul_bias(activation, weight, scale, bias);
    const Tensor expected_cpu = expected.to(Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor deferred;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        deferred = axiom::ops::int8_matmul_bias(activation, weight, scale, bias);
    }

    const Tensor actual_cpu = deferred.to(Device::CPU);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.direct_int8_steps, 1U);
    ASSERT_EQ(actual_cpu.shape(),
              Shape({kBatch, kTime, kOutputFeatures}));
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.typed_data<uint16_t>(),
                          expected_cpu.typed_data<uint16_t>(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, EncodesEligibleProjectionIntoCallerOutput) {
    constexpr size_t kRows = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor expected = axiom::ops::int8_matmul(activation, weight, scale);

    Tensor output = gpu_fp16({kRows, kOutputFeatures}, /* 0.0f */ 0x0000);
    const auto output_storage = output.storage();

    ASSERT_TRUE(axiom::backends::metal::gpu_int8_matmul_into(
        output, activation, weight, scale));
    EXPECT_EQ(output.storage(), output_storage);

    const Tensor actual_cpu = output.to(Device::CPU);
    const Tensor expected_cpu = expected.to(Device::CPU);
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.typed_data<uint16_t>(),
                          expected_cpu.typed_data<uint16_t>(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, MaterializesMpsGraphActivationBeforeDirectProjection) {
    constexpr size_t kRows = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager_input = axiom::ops::add(activation, activation);
    const Tensor expected = axiom::ops::int8_matmul(eager_input, weight, scale);
    const Tensor expected_cpu = expected.to(Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor deferred;
    {
        const Tensor mpsgraph_input = axiom::ops::add(activation, activation);
        axiom::graph::ScopedHybridMetalExecution scope;
        deferred = axiom::ops::int8_matmul(mpsgraph_input, weight, scale);
    }

    const Tensor actual_cpu = deferred.to(Device::CPU);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.mpsgraph_islands, 1U);
    EXPECT_EQ(stats.direct_int8_steps, 1U);
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.typed_data<uint16_t>(),
                          expected_cpu.typed_data<uint16_t>(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, MaterializesDirectProjectionBeforeMpsGraphConsumer) {
    constexpr size_t kRows = 2;
    constexpr size_t kInputFeatures = 32;
    constexpr size_t kOutputFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kInputFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight =
        gpu_int8({kOutputFeatures, kInputFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kOutputFeatures, kInputFeatures / 32}, /* 1.0f */ 0x3c00);
    const Tensor bias =
        gpu_fp16({kOutputFeatures}, /* 0.0f */ 0x0000);

    const Tensor eager_projection =
        axiom::ops::int8_matmul_bias(activation, weight, scale, bias);
    const Tensor expected =
        axiom::ops::add(eager_projection, eager_projection);
    const Tensor expected_cpu = expected.to(Device::CPU);

    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor consumed;
    {
        axiom::graph::ScopedHybridMetalExecution scope;
        const Tensor direct_projection =
            axiom::ops::int8_matmul_bias(activation, weight, scale, bias);
        consumed = axiom::ops::add(direct_projection, direct_projection);
    }

    const Tensor actual_cpu = consumed.to(Device::CPU);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.mpsgraph_islands, 1U);
    EXPECT_EQ(stats.direct_int8_steps, 1U);
    ASSERT_EQ(actual_cpu.numel(), expected_cpu.numel());
    EXPECT_EQ(std::memcmp(actual_cpu.typed_data<uint16_t>(),
                          expected_cpu.typed_data<uint16_t>(),
                          actual_cpu.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, ReusesPrivateSlotAfterItsFinalEncodedRead) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const ScopedEnvironmentVariable final_read_reuse(
        "WASPER_PARAKEET_HYBRID_FINAL_READ_REUSE", "1");
    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager_first_island = axiom::ops::add(activation, activation);
    const Tensor eager_first_projection =
        axiom::ops::int8_matmul(eager_first_island, weight, scale);
    const Tensor eager_second_island = axiom::ops::add(
        eager_first_projection, eager_first_projection);
    const Tensor eager_second_projection =
        axiom::ops::int8_matmul(eager_second_island, weight, scale);
    const Tensor expected = axiom::ops::add(
        eager_second_projection, eager_second_projection).to(Device::CPU);

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        const Tensor first_island = axiom::ops::add(activation, activation);
        const Tensor first_projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        const Tensor second_island = axiom::ops::add(
            first_projection, first_projection);
        const Tensor second_projection =
            axiom::ops::int8_matmul(second_island, weight, scale);
        actual = axiom::ops::add(second_projection, second_projection).to(
            Device::CPU);
    }

    ASSERT_EQ(actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          actual.numel() * sizeof(uint16_t)),
              0);
    EXPECT_LE(workspace.snapshot().fresh_allocations, 4U)
        << "The second projection must reuse the first projection's physical "
           "buffer only after the intervening MPSGraph consumer has encoded "
           "its final read";
}

TEST(HybridMetalGraph,
     FinalReadReuseIsAuthorizedByMpsGraphCompletionEvent) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const ScopedEnvironmentVariable final_read_reuse(
        "WASPER_PARAKEET_HYBRID_FINAL_READ_REUSE", "1");
    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    const auto events_before = stream.workspace_event_stats();
    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        const Tensor first_island = axiom::ops::add(activation, activation);
        const Tensor first_projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        const Tensor second_island = axiom::ops::add(
            first_projection, first_projection);
        actual = axiom::ops::add(second_island, second_island).to(Device::CPU);
    }

    ASSERT_EQ(actual.numel(), kRows * kFeatures);
    const auto events_after = stream.workspace_event_stats();
    EXPECT_EQ(events_after.mpsgraph_completion_signals,
              events_before.mpsgraph_completion_signals + 1)
        << "MPSGraph final-read reuse must be gated by the executable's "
           "completion event rather than by the stream root command buffer";
    EXPECT_EQ(workspace.snapshot().workspace_reuse_event_signals, 1U);
}

TEST(HybridMetalGraph, RecomputesReleasedProjectionOnLaterRead) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager_first_island = axiom::ops::add(activation, activation);
    const Tensor expected_first_projection =
        axiom::ops::int8_matmul(eager_first_island, weight, scale).to(
            Device::CPU);

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    Tensor actual_first_projection;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        const Tensor first_island = axiom::ops::add(activation, activation);
        const Tensor first_projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        const Tensor second_island = axiom::ops::add(
            first_projection, first_projection);
        const Tensor second_projection =
            axiom::ops::int8_matmul(second_island, weight, scale);
        const Tensor final = axiom::ops::add(
            second_projection, second_projection).to(Device::CPU);
        ASSERT_EQ(final.numel(), kRows * kFeatures);

        // The first projection's last graph read happened while materializing
        // `second_island`, so its former private slot may already back a later
        // temporary. Asking for it again must rebuild the lazy projection,
        // rather than returning an empty stale cache entry.
        actual_first_projection = first_projection.to(Device::CPU);
    }

    ASSERT_EQ(actual_first_projection.numel(), expected_first_projection.numel());
    EXPECT_EQ(std::memcmp(actual_first_projection.typed_data<uint16_t>(),
                          expected_first_projection.typed_data<uint16_t>(),
                          actual_first_projection.numel() * sizeof(uint16_t)),
              0);
}

TEST(HybridMetalGraph, ReportsOrderedTemporaryLifetimes) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const ScopedEnvironmentVariable final_read_reuse(
        "WASPER_PARAKEET_HYBRID_FINAL_READ_REUSE", "1");
    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        const Tensor first_island = axiom::ops::add(activation, activation);
        const Tensor first_projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        const Tensor second_island = axiom::ops::add(
            first_projection, first_projection);
        const Tensor second_projection =
            axiom::ops::int8_matmul(second_island, weight, scale);
        actual = axiom::ops::add(second_projection, second_projection).to(
            Device::CPU);
    }

    ASSERT_EQ(actual.numel(), kRows * kFeatures);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.mpsgraph_islands, 3U);
    EXPECT_EQ(stats.direct_int8_steps, 2U);
    EXPECT_EQ(stats.logical_temporary_values, 4U)
        << "The final CPU-visible graph result is not an arena temporary";
    EXPECT_EQ(stats.temporary_slots, 3U)
        << "Only the two direct projections share a slot in the current "
           "ordered schedule";
    EXPECT_EQ(stats.cpu_synchronizations, 0U)
        << "The final CPU read is outside the executor's inter-step schedule";
}

TEST(HybridMetalGraph, ExecutesMixedProgramWithTwoPreallocatedSlots) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager_first_island = axiom::ops::add(activation, activation);
    const Tensor eager_first_projection =
        axiom::ops::int8_matmul(eager_first_island, weight, scale);
    const Tensor eager_second_island = axiom::ops::add(
        eager_first_projection, eager_first_projection);
    const Tensor eager_second_projection =
        axiom::ops::int8_matmul(eager_second_island, weight, scale);
    const Tensor expected = axiom::ops::add(
        eager_second_projection, eager_second_projection).to(Device::CPU);

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalProgramExecution program_scope;
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        const Tensor first_island = axiom::ops::add(activation, activation);
        const Tensor first_projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        const Tensor second_island = axiom::ops::add(
            first_projection, first_projection);
        const Tensor second_projection =
            axiom::ops::int8_matmul(second_island, weight, scale);
        actual = axiom::ops::add(second_projection, second_projection).to(
            Device::CPU);
    }

    ASSERT_EQ(actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          actual.numel() * sizeof(uint16_t)),
              0);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_EQ(stats.logical_temporary_values, 4U);
    EXPECT_EQ(stats.temporary_slots, 2U)
        << "The program reuses each slot after the encoded final read";
    EXPECT_LE(workspace.snapshot().fresh_allocations, 3U)
        << "Two planned slots plus the final output replace four ordinary "
           "temporary allocations";
}

TEST(HybridMetalGraph, HandlesSharedMpsGraphDependencyInOrderedProgram) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager_shared = axiom::ops::add(activation, activation);
    const Tensor eager_first_island =
        axiom::ops::add(eager_shared, activation);
    const Tensor eager_projection =
        axiom::ops::int8_matmul(eager_first_island, weight, scale);
    const Tensor expected =
        axiom::ops::add(eager_shared, eager_projection).to(Device::CPU);

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    axiom::graph::reset_hybrid_metal_graph_execution_stats();
    Tensor actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalProgramExecution program_scope;
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        const Tensor shared = axiom::ops::add(activation, activation);
        const Tensor first_island = axiom::ops::add(shared, activation);
        const Tensor projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        actual = axiom::ops::add(shared, projection).to(Device::CPU);
    }

    ASSERT_EQ(actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          actual.numel() * sizeof(uint16_t)),
              0);
    const auto stats = axiom::graph::hybrid_metal_graph_execution_stats();
    EXPECT_GE(stats.ordered_program_microprograms, 1U);
    EXPECT_EQ(stats.cpu_synchronizations, 0U);
}

TEST(HybridMetalGraph, ReusesFinalizedPlannedSlotsAcrossAdjacentPrograms) {
    constexpr size_t kRows = 2;
    constexpr size_t kFeatures = 32;

    const Tensor activation =
        gpu_fp16({kRows, kFeatures}, /* 1.0f */ 0x3c00);
    const Tensor weight = gpu_int8({kFeatures, kFeatures}, /* 1 */ 1);
    const Tensor scale =
        gpu_fp16({kFeatures, kFeatures / 32}, /* 1.0f */ 0x3c00);

    const Tensor eager_first_island = axiom::ops::add(activation, activation);
    const Tensor eager_first_projection =
        axiom::ops::int8_matmul(eager_first_island, weight, scale);
    const Tensor eager_second_island = axiom::ops::add(
        eager_first_projection, eager_first_projection);
    const Tensor eager_second_projection =
        axiom::ops::int8_matmul(eager_second_island, weight, scale);
    const Tensor expected = axiom::ops::add(
        eager_second_projection, eager_second_projection).to(Device::CPU);

    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    auto run_program = [&] {
        const Tensor first_island = axiom::ops::add(activation, activation);
        const Tensor first_projection =
            axiom::ops::int8_matmul(first_island, weight, scale);
        const Tensor second_island = axiom::ops::add(
            first_projection, first_projection);
        const Tensor second_projection =
            axiom::ops::int8_matmul(second_island, weight, scale);
        return axiom::ops::add(second_projection, second_projection).to(
            Device::CPU);
    };

    Tensor first_actual;
    Tensor second_actual;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        axiom::graph::ScopedHybridMetalProgramExecution program_scope;
        axiom::graph::ScopedHybridMetalExecution hybrid_scope;
        first_actual = run_program();
        second_actual = run_program();
    }

    ASSERT_EQ(first_actual.numel(), expected.numel());
    ASSERT_EQ(second_actual.numel(), expected.numel());
    EXPECT_EQ(std::memcmp(first_actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          expected.numel() * sizeof(uint16_t)),
              0);
    EXPECT_EQ(std::memcmp(second_actual.typed_data<uint16_t>(),
                          expected.typed_data<uint16_t>(),
                          expected.numel() * sizeof(uint16_t)),
              0);
    EXPECT_LE(workspace.snapshot().fresh_allocations, 4U)
        << "Only the two final graph results may require new buffers after "
           "the first program; finalized planned slots must be reusable";
}

TEST(HybridMetalGraph, CreatesLogicalViewsOverOnePlannedSlotLease) {
    MetalWorkspaceCache workspace(
        1024 * 1024, WorkspaceBufferStorageMode::Private);
    size_t lease_count = 0;
    {
        ScopedMetalWorkspace workspace_scope(workspace);
        {
            auto slots = axiom::backends::metal::acquire_hybrid_workspace_slots(
                {4096, 4096});
            const auto first_value = slots->make_storage_view(0, 4096);
            const auto later_value = slots->make_storage_view(0, 4096);
            lease_count = slots->lease_count();

            EXPECT_EQ(workspace.snapshot().fresh_allocations, 2U);
            EXPECT_EQ(
                axiom::backends::metal::as_metal_buffer_provider(first_value.get())
                    ->buffer(),
                axiom::backends::metal::as_metal_buffer_provider(later_value.get())
                    ->buffer());
        }
    }

    EXPECT_EQ(lease_count, 2U)
        << "Two planned slots need two physical workspace leases";
}

} // namespace
