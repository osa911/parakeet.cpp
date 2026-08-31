#include "parakeet/models/fastconformer_block_program.hpp"

#include "parakeet/models/encoder.hpp"

#include "backends/metal/metal_operations.hpp"
#include "backends/metal/metal_common.hpp"
#include "backends/metal/metal_workspace_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <vector>

#include <axiom/error.hpp>
#include <axiom/operations.hpp>

namespace parakeet::models {

namespace {

using axiom::Device;
using axiom::DType;
using axiom::Shape;
using axiom::ShapeUtils;
using axiom::Tensor;
using axiom::nn::BatchNorm1d;
using axiom::nn::Conv1d;
using axiom::nn::LayerNorm;
using axiom::nn::Linear;
using axiom::backends::metal::HybridWorkspaceSlots;

constexpr std::array<size_t, 3> kSupportedSequenceBuckets{44, 120, 144};
constexpr size_t kSupportedBatch = 1;

std::atomic<size_t> direct_dispatches{0};

bool exact_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool is_supported_sequence_bucket(size_t time) {
    return std::find(kSupportedSequenceBuckets.begin(),
                     kSupportedSequenceBuckets.end(), time) !=
           kSupportedSequenceBuckets.end();
}

bool has_materialized_storage(const Tensor &tensor) {
    return !tensor.is_lazy() && tensor.storage() != nullptr;
}

bool is_contiguous_gpu_tensor(const Tensor &tensor, DType dtype,
                              const Shape &shape) {
    return has_materialized_storage(tensor) && tensor.device() == Device::GPU &&
           tensor.dtype() == dtype && tensor.shape() == shape &&
           tensor.is_contiguous();
}

bool is_optional_gpu_bias(const Tensor &bias, size_t size) {
    // Axiom represents an omitted bias as a default Tensor with no storage;
    // it is not necessarily size-empty. Treat only that storage-less sentinel
    // as optional, otherwise require a concrete GPU FP16 vector.
    return !bias.storage() ||
           is_contiguous_gpu_tensor(bias, DType::Float16, Shape{size});
}

bool is_layer_norm_ready(const LayerNorm &norm, size_t hidden) {
    return is_contiguous_gpu_tensor(norm.weight(), DType::Float16,
                                    Shape{hidden}) &&
           is_contiguous_gpu_tensor(norm.bias(), DType::Float16,
                                    Shape{hidden});
}

bool is_int8_linear_ready(const Linear &linear, size_t output,
                          size_t input) {
    return is_contiguous_gpu_tensor(linear.weight(), DType::Int8,
                                    Shape{output, input}) &&
           is_contiguous_gpu_tensor(linear.scale(), DType::Float16,
                                    Shape{output, input / 32}) &&
           is_optional_gpu_bias(linear.bias(), output);
}

bool is_int8_pointwise_ready(const Conv1d &conv, size_t output,
                             size_t input) {
    return conv.has_int8_pointwise_weights() &&
           is_contiguous_gpu_tensor(conv.weight(), DType::Int8,
                                    Shape{output, input}) &&
           is_contiguous_gpu_tensor(conv.scale(), DType::Float16,
                                    Shape{output, input / 32}) &&
           is_optional_gpu_bias(conv.bias(), output);
}

bool is_depthwise_ready(const Conv1d &conv, size_t hidden) {
    return conv.groups() == static_cast<int>(hidden) && conv.padding() == 4 &&
           is_contiguous_gpu_tensor(conv.weight(), DType::Float16,
                                    Shape{hidden, 1, 9}) &&
           is_optional_gpu_bias(conv.bias(), hidden);
}

bool is_batch_norm_ready(const BatchNorm1d &norm, size_t hidden) {
    const Shape shape{hidden};
    return std::isfinite(norm.eps()) && norm.eps() > 0.0f &&
           is_contiguous_gpu_tensor(norm.weight(), DType::Float16, shape) &&
           is_contiguous_gpu_tensor(norm.bias(), DType::Float16, shape) &&
           is_contiguous_gpu_tensor(norm.running_mean(), DType::Float16,
                                    shape) &&
           is_contiguous_gpu_tensor(norm.running_var(), DType::Float16,
                                    shape);
}

class FfnSlotPlan {
  public:
    FfnSlotPlan(const Tensor &input, size_t ffn1_intermediate,
                size_t ffn2_intermediate)
        : input_shape_(input.shape()),
          max_intermediate_(std::max(ffn1_intermediate, ffn2_intermediate)) {
        const size_t flattened_scratch_bytes =
            input_shape_[0] * input_shape_[1] * max_intermediate_ *
            axiom::dtype_size(DType::Float16);
        slots_ = axiom::backends::metal::acquire_hybrid_workspace_slots(
            {input.nbytes(), flattened_scratch_bytes, input.nbytes(),
             input.nbytes()});
    }

    Tensor normalized() const { return make_view(/*slot=*/0, input_shape_); }

    Tensor scratch(size_t intermediate) const {
        return make_view(/*slot=*/1,
                         Shape{input_shape_[0] * input_shape_[1], intermediate});
    }

    Tensor first_result() const {
        return make_view(/*slot=*/2, input_shape_);
    }

    Tensor second_result() const {
        return make_view(/*slot=*/3, input_shape_);
    }

    size_t slot_count() const { return slots_->lease_count(); }

  private:
    Tensor make_view(size_t slot, const Shape &shape) const {
        const size_t bytes = ShapeUtils::size(shape) *
                             axiom::dtype_size(DType::Float16);
        return Tensor(slots_->make_storage_view(slot, bytes), shape,
                      ShapeUtils::calculate_strides(
                          shape, axiom::dtype_size(DType::Float16)),
                      DType::Float16);
    }

    Shape input_shape_;
    size_t max_intermediate_ = 0;
    std::unique_ptr<HybridWorkspaceSlots> slots_;
};

class AttentionSlotPlan {
  public:
    AttentionSlotPlan(const Tensor &input, size_t heads)
        : input_shape_(input.shape()),
          qkv_shape_{input_shape_[0], heads, input_shape_[1],
                     input_shape_[2] / heads} {
        const size_t qkv_bytes = ShapeUtils::size(qkv_shape_) *
                                 axiom::dtype_size(DType::Float16);
        slots_ = axiom::backends::metal::acquire_hybrid_workspace_slots(
            {input.nbytes(), qkv_bytes, qkv_bytes, qkv_bytes});
    }

    Tensor normalized() const { return make_view(/*slot=*/0, input_shape_); }

    Tensor query() const { return make_view(/*slot=*/1, qkv_shape_); }

    Tensor key() const { return make_view(/*slot=*/2, qkv_shape_); }

    Tensor value() const { return make_view(/*slot=*/3, qkv_shape_); }

    // QKV has consumed the normalized value before this view is bound as the
    // direct attention output. Both operations stay on the ordered custom
    // Metal stream; Q/K/V remain in their own slots through the attention
    // dispatch.
    Tensor attention_result(const Tensor &normalized) const {
        return Tensor(normalized.storage(), qkv_shape_,
                      ShapeUtils::calculate_strides(
                          qkv_shape_, axiom::dtype_size(DType::Float16)),
                      DType::Float16);
    }

    size_t slot_count() const { return slots_->lease_count(); }

  private:
    Tensor make_view(size_t slot, const Shape &shape) const {
        const size_t bytes = ShapeUtils::size(shape) *
                             axiom::dtype_size(DType::Float16);
        return Tensor(slots_->make_storage_view(slot, bytes), shape,
                      ShapeUtils::calculate_strides(
                          shape, axiom::dtype_size(DType::Float16)),
                      DType::Float16);
    }

    Shape input_shape_;
    Shape qkv_shape_;
    std::unique_ptr<HybridWorkspaceSlots> slots_;
};

class FullBlockSlotPlan {
  public:
    FullBlockSlotPlan(const Tensor &input, size_t max_intermediate,
                      size_t heads)
        : input_shape_(input.shape()),
          attention_shape_{input_shape_[0], heads, input_shape_[1],
                           input_shape_[2] / heads} {
        const size_t scratch_bytes =
            input_shape_[0] * input_shape_[1] * max_intermediate *
            axiom::dtype_size(DType::Float16);
        slots_ = axiom::backends::metal::acquire_hybrid_workspace_slots(
            {input.nbytes(), input.nbytes(), scratch_bytes, input.nbytes(),
             input.nbytes(), input.nbytes()});
    }

    Tensor normalized() const { return hidden_view(/*slot=*/0); }
    Tensor residual() const { return hidden_view(/*slot=*/1); }
    Tensor scratch(size_t intermediate) const {
        return make_view(/*slot=*/2,
                         Shape{input_shape_[0] * input_shape_[1], intermediate});
    }
    Tensor query() const { return attention_view(/*slot=*/3); }
    Tensor key() const { return attention_view(/*slot=*/4); }
    Tensor value() const { return attention_view(/*slot=*/5); }
    Tensor hidden_in_key_slot() const { return hidden_view(/*slot=*/4); }
    Tensor gathered_attention() const {
        return make_view(/*slot=*/3,
                         Shape{input_shape_[0], input_shape_[1],
                               attention_shape_[1], attention_shape_[3]});
    }
    Tensor pointwise_glu() const {
        return make_view(/*slot=*/5,
                         Shape{input_shape_[0], input_shape_[2], input_shape_[1]});
    }
    Tensor depthwise_physical() const { return hidden_view(/*slot=*/1); }

  private:
    Tensor hidden_view(size_t slot) const { return make_view(slot, input_shape_); }
    Tensor attention_view(size_t slot) const {
        return make_view(slot, attention_shape_);
    }
    Tensor make_view(size_t slot, const Shape &shape) const {
        const size_t bytes = ShapeUtils::size(shape) *
                             axiom::dtype_size(DType::Float16);
        return Tensor(slots_->make_storage_view(slot, bytes), shape,
                      ShapeUtils::calculate_strides(
                          shape, axiom::dtype_size(DType::Float16)),
                      DType::Float16);
    }

    Shape input_shape_;
    Shape attention_shape_;
    std::unique_ptr<HybridWorkspaceSlots> slots_;
};

} // namespace

const Tensor *FastConformerBlockProgram::find_cached_head_major_position(
    const ConformerAttention &attention, const Tensor &pos_emb, size_t heads,
    size_t time) {
    constexpr size_t kDirectHeadDim = 128;
    const Shape expected_shape{1, heads, 2 * time - 1, kDirectHeadDim};
    for (const auto &entry : attention.position_projection_cache_) {
        if (entry.input_shape == pos_emb.shape() &&
            entry.dtype == pos_emb.dtype() && entry.device == pos_emb.device() &&
            entry.head_major &&
            is_contiguous_gpu_tensor(entry.projected, DType::Float16,
                                     expected_shape)) {
            return &entry.projected;
        }
    }
    return nullptr;
}

bool FastConformerBlockProgram::supports_direct_ffn(const FeedForward &ffn,
                                                     const Tensor &input,
                                                     size_t hidden) {
    const bool norm_ready = is_layer_norm_ready(ffn.norm_, hidden);
    const bool fc1_ready = is_int8_linear_ready(
        ffn.fc1_, ffn.fc1_.weight().shape()[0], hidden);
    if (!norm_ready || !fc1_ready) {
        return false;
    }
    const size_t intermediate = ffn.fc1_.weight().shape()[0];
    const bool fc2_ready = is_int8_linear_ready(ffn.fc2_, hidden, intermediate);
    if (intermediate == 0 || intermediate % 32 != 0 || !fc2_ready) {
        return false;
    }

    const size_t rows = input.shape()[0] * input.shape()[1];
    const Tensor input_flat = input.reshape({rows, hidden});
    const Tensor fc1_shape_probe({rows, intermediate}, DType::Float16,
                                 Device::CPU);
    return axiom::backends::metal::gpu_int8_matmul_supports_supplied_output(
               input_flat, ffn.fc1_.weight()) &&
           axiom::backends::metal::gpu_int8_matmul_supports_supplied_output(
               fc1_shape_probe, ffn.fc2_.weight());
}

bool FastConformerBlockProgram::is_supported(const ConformerBlock &block,
                                             const Tensor &input,
                                             const Tensor &pos_emb,
                                             const Tensor &mask) {
#ifndef AXIOM_METAL_SUPPORT
    static_cast<void>(block);
    static_cast<void>(input);
    static_cast<void>(pos_emb);
    static_cast<void>(mask);
    return false;
#else
    if (!exact_flag_enabled("PARAKEET_FASTCONFORMER_BLOCK_PROGRAM") ||
        !exact_flag_enabled("PARAKEET_DIRECT_INT8_FFN") ||
        !exact_flag_enabled("PARAKEET_DIRECT_F16_LAYERNORM") ||
        input.is_lazy() || input.device() != Device::GPU ||
        input.dtype() != DType::Float16 || !input.is_contiguous() ||
        input.shape().size() != 3 || input.shape()[0] != kSupportedBatch ||
        !is_supported_sequence_bucket(input.shape()[1]) ||
        input.shape()[2] == 0 || input.shape()[2] % 32 != 0) {
        return false;
    }

    const size_t hidden = input.shape()[2];
    if (!has_materialized_storage(input)) {
        return false;
    }

    const bool ffn1_supported = supports_direct_ffn(block.ffn1_, input, hidden);
    const bool ffn2_supported = supports_direct_ffn(block.ffn2_, input, hidden);
    const bool private_workspace = axiom::backends::metal::
        active_metal_workspace_uses_private_buffers();
    if (!ffn1_supported || !ffn2_supported) {
        return false;
    }

    if (!private_workspace) {
        return false;
    }

    // The full native relative-position attention kernel is a separate
    // experiment. Its FP16 reduction order is not yet exact on real model
    // inputs, so the supported path below retains the established attention
    // and convolution graph and only plans the two exact FFN segments.
    if (!exact_flag_enabled("PARAKEET_RELATIVE_POSITION_ATTENTION")) {
        return true;
    }

    const size_t time = input.shape()[1];
    // Once the head-major projection has been warmed, the full direct
    // schedule consumes that validated cached tensor, not raw positional
    // embedding data. The encoder may therefore keep raw pos_emb lazy.
    if (pos_emb.device() != Device::GPU || pos_emb.dtype() != DType::Float16 ||
        pos_emb.shape() != Shape{2 * time - 1, hidden} || mask.storage()) {
        return false;
    }

    const auto &attention = block.attn_;
    const size_t heads = static_cast<size_t>(attention.mha_.num_heads());
    constexpr size_t kDirectHeadDim = 128;
    if (heads == 0 || hidden != heads * kDirectHeadDim ||
        !is_layer_norm_ready(attention.norm_, hidden) ||
        !is_int8_linear_ready(attention.mha_.q_proj(), hidden, hidden) ||
        !is_int8_linear_ready(attention.mha_.k_proj(), hidden, hidden) ||
        !is_int8_linear_ready(attention.mha_.v_proj(), hidden, hidden) ||
        !is_int8_linear_ready(attention.mha_.out_proj(), hidden, hidden) ||
        !is_contiguous_gpu_tensor(attention.pos_proj_.weight(),
                                  DType::Float16, Shape{hidden, hidden}) ||
        !is_contiguous_gpu_tensor(attention.pos_bias_u_, DType::Float16,
                                  Shape{heads, hidden / heads}) ||
        !is_contiguous_gpu_tensor(attention.pos_bias_v_, DType::Float16,
                                  Shape{heads, kDirectHeadDim}) ||
        !find_cached_head_major_position(attention, pos_emb, heads, time) ||
        !axiom::backends::metal::
            gpu_relative_position_attention_pipeline_available()) {
        return false;
    }

    const Tensor flattened =
        input.reshape({kSupportedBatch * time, hidden});
    if (!axiom::backends::metal::gpu_int8_matmul_supports_supplied_output(
            flattened, attention.mha_.out_proj().weight())) {
        return false;
    }

    const auto &conv = block.conv_;
    if (!is_layer_norm_ready(conv.norm_, hidden) ||
        !is_int8_pointwise_ready(conv.pointwise_conv1_, 2 * hidden, hidden) ||
        !is_depthwise_ready(conv.depthwise_conv_, hidden) ||
        !is_batch_norm_ready(conv.batch_norm_, hidden) ||
        !is_int8_pointwise_ready(conv.pointwise_conv2_, hidden, hidden) ||
        !is_layer_norm_ready(block.final_norm_, hidden)) {
        return false;
    }

    return true;
#endif
}

Tensor FastConformerBlockProgram::encode(const ConformerBlock &block,
                                         const Tensor &input,
                                         const Tensor &pos_emb,
                                         const Tensor &mask) {
    if (!is_supported(block, input, pos_emb, mask)) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer block program is unavailable for this request");
    }

    const size_t batch = input.shape()[0];
    const size_t time = input.shape()[1];
    const size_t hidden = input.shape()[2];
    const size_t max_intermediate = std::max(
        block.ffn1_.fc1_.weight().shape()[0],
        block.ffn2_.fc1_.weight().shape()[0]);
    const auto require = [](bool encoded, const char *operation) {
        if (!encoded) {
            throw axiom::RuntimeError::internal(
                std::string("FastConformer block program could not encode ") +
                operation);
        }
    };

    // Preserve the established attention and convolution graph until the
    // native relative-position kernel establishes exact real-model parity.
    // The two FFNs are nevertheless explicit scheduled segments: their
    // persistent slots avoid request-local allocation churn while the normal
    // graph continues to own the mathematically sensitive middle of a block.
    if (!exact_flag_enabled("PARAKEET_RELATIVE_POSITION_ATTENTION")) {
        FfnSlotPlan slots(input, block.ffn1_.fc1_.weight().shape()[0],
                          block.ffn2_.fc1_.weight().shape()[0]);
        const auto encode_ffn = [&](const FeedForward &ffn,
                                    const Tensor &residual, Tensor output) {
            Tensor normalized = slots.normalized();
            require(axiom::backends::metal::gpu_layer_norm_into(
                        normalized, residual, ffn.norm_.weight(),
                        ffn.norm_.bias(), -1, ffn.norm_.eps()),
                    "FFN LayerNorm");
            direct_dispatches.fetch_add(1, std::memory_order_relaxed);
            Tensor scratch = slots.scratch(ffn.fc1_.weight().shape()[0]);
            axiom::ops::int8_ffn_silu_residual_into(
                scratch, output, normalized, residual, ffn.fc1_.weight(),
                ffn.fc1_.scale(), ffn.fc1_.bias(), ffn.fc2_.weight(),
                ffn.fc2_.scale(), ffn.fc2_.bias());
            direct_dispatches.fetch_add(1, std::memory_order_relaxed);
            return output;
        };

        Tensor ffn1_result =
            encode_ffn(block.ffn1_, input, slots.first_result());
        Tensor attention_result = block.attn_(ffn1_result, pos_emb, mask);
        Tensor convolution_result = block.conv_(attention_result);
        // The second direct FFN binds a concrete input to the ordered Metal
        // stream. Materializing this graph boundary proves the first FFN's
        // result has reached its final graph read before the second segment.
        static_cast<void>(convolution_result.storage());
        Tensor ffn2_result = encode_ffn(block.ffn2_, convolution_result,
                                        slots.second_result());
        return block.final_norm_(ffn2_result);
    }

    const auto &attention = block.attn_;
    const size_t heads = static_cast<size_t>(attention.mha_.num_heads());
    const size_t head_dim = hidden / heads;
    const Tensor *head_major_position =
        find_cached_head_major_position(attention, pos_emb, heads, time);
    if (!head_major_position) {
        throw axiom::RuntimeError::internal(
            "FastConformer block program lost its warm position cache");
    }

    FullBlockSlotPlan slots(input, max_intermediate, heads);
    const auto encode_ffn = [&](const FeedForward &ffn, const Tensor &residual,
                                Tensor output) {
        Tensor normalized = slots.normalized();
        require(axiom::backends::metal::gpu_layer_norm_into(
                    normalized, residual, ffn.norm_.weight(), ffn.norm_.bias(),
                    -1, ffn.norm_.eps()),
                "FFN LayerNorm");
        direct_dispatches.fetch_add(1, std::memory_order_relaxed);
        Tensor scratch = slots.scratch(ffn.fc1_.weight().shape()[0]);
        axiom::ops::int8_ffn_silu_residual_into(
            scratch, output, normalized, residual, ffn.fc1_.weight(),
            ffn.fc1_.scale(), ffn.fc1_.bias(), ffn.fc2_.weight(),
            ffn.fc2_.scale(), ffn.fc2_.bias());
        direct_dispatches.fetch_add(1, std::memory_order_relaxed);
        return output;
    };

    // FFN1: slot 0 is normalized only through its FFN dispatch; slot 1 keeps
    // the result as attention's residual.
    Tensor ffn1_result =
        encode_ffn(block.ffn1_, input, slots.residual());

    // Attention: Q/K/V remain live through the native attention dispatch.
    Tensor attention_normalized = slots.normalized();
    require(axiom::backends::metal::gpu_layer_norm_into(
                attention_normalized, ffn1_result, attention.norm_.weight(),
                attention.norm_.bias(), -1, attention.norm_.eps()),
            "attention LayerNorm");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    Tensor query = slots.query();
    Tensor key = slots.key();
    Tensor value = slots.value();
    require(axiom::backends::metal::gpu_int8_qkv_matmul_bias_into(
                query, key, value,
                attention_normalized.reshape({batch * time, hidden}),
                attention.mha_.q_proj().weight(), attention.mha_.q_proj().scale(),
                attention.mha_.q_proj().bias(), attention.mha_.k_proj().weight(),
                attention.mha_.k_proj().scale(), attention.mha_.k_proj().bias(),
                attention.mha_.v_proj().weight(), attention.mha_.v_proj().scale(),
                attention.mha_.v_proj().bias(), batch, time, heads),
            "attention QKV");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    const Tensor bias_u = attention.pos_bias_u_.reshape({1, heads, 1, head_dim});
    const Tensor bias_v = attention.pos_bias_v_.reshape({1, heads, 1, head_dim});
    Tensor attention_result = slots.normalized().reshape(
        Shape{batch, heads, time, head_dim});
    require(axiom::backends::metal::gpu_relative_position_attention_into(
                attention_result, query, key, value, *head_major_position,
                bias_u, bias_v, mask,
                1.0f / std::sqrt(static_cast<float>(head_dim))),
            "relative attention");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    Tensor gathered_attention = slots.gathered_attention();
    require(axiom::backends::metal::gpu_make_contiguous_into(
                gathered_attention,
                attention_result.transpose({0, 2, 1, 3})),
            "attention gather");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    Tensor attention_projected = slots.normalized();
    Tensor attention_projected_flat =
        attention_projected.reshape({batch * time, hidden});
    Tensor gathered_attention_flat =
        gathered_attention.reshape({batch * time, hidden});
    require(axiom::backends::metal::gpu_int8_matmul_bias_into(
                attention_projected_flat,
                gathered_attention_flat,
                attention.mha_.out_proj().weight(),
                attention.mha_.out_proj().scale(), attention.mha_.out_proj().bias()),
            "attention output projection");
    axiom::backends::metal::gpu_add_residual_inplace_f16(
        attention_projected_flat,
        ffn1_result.reshape({batch * time, hidden}));
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);

    // Convolution: slot 5 is GLU [B,C,T], then slot 1 becomes the physical
    // [B,T,C] depthwise result. The pointwise-2 projection safely reuses slot 4.
    Tensor conv_normalized = slots.hidden_in_key_slot();
    require(axiom::backends::metal::gpu_layer_norm_into(
                conv_normalized, attention_projected, block.conv_.norm_.weight(),
                block.conv_.norm_.bias(), -1, block.conv_.norm_.eps()),
            "convolution LayerNorm");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    Tensor glu = slots.pointwise_glu();
    require(axiom::backends::metal::gpu_fastconformer_int8_pointwise_glu_f16_into(
                glu, conv_normalized, block.conv_.pointwise_conv1_.weight(),
                block.conv_.pointwise_conv1_.scale(),
                block.conv_.pointwise_conv1_.bias()),
            "convolution pointwise GLU");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    Tensor depthwise_physical = slots.depthwise_physical();
    Tensor depthwise = depthwise_physical.permute({0, 2, 1});
    require(axiom::backends::metal::gpu_depthwise_conv1d_batch_norm_silu_into(
                depthwise, glu, block.conv_.depthwise_conv_.weight(),
                block.conv_.batch_norm_.running_mean(),
                block.conv_.batch_norm_.running_var(),
                block.conv_.batch_norm_.weight(), block.conv_.batch_norm_.bias(),
                block.conv_.batch_norm_.eps()),
            "depthwise convolution");
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    Tensor conv_projected = slots.hidden_in_key_slot();
    Tensor conv_projected_flat =
        conv_projected.reshape({batch * time, hidden});
    require(axiom::backends::metal::gpu_int8_matmul_bias_into(
                conv_projected_flat,
                depthwise_physical.reshape({batch * time, hidden}),
                block.conv_.pointwise_conv2_.weight(),
                block.conv_.pointwise_conv2_.scale(),
                block.conv_.pointwise_conv2_.bias()),
            "convolution output projection");
    axiom::backends::metal::gpu_add_residual_inplace_f16(
        conv_projected_flat,
        attention_projected.reshape({batch * time, hidden}));
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);

    // FFN2 consumes the convolution result. Its final tensor remains outside
    // the reusable slots after the final LayerNorm has been encoded.
    Tensor ffn2_result =
        encode_ffn(block.ffn2_, conv_projected, slots.normalized());
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);
    return block.final_norm_(ffn2_result);
}

bool FastConformerBlockProgram::is_direct_attention_segment_ready(
    const ConformerBlock &block, const Tensor &input,
    const Tensor &head_major_position, const Tensor &mask) {
    if (!has_materialized_storage(input) || input.device() != Device::GPU ||
        input.dtype() != DType::Float16 || !input.is_contiguous() ||
        input.shape().size() != 3 || input.shape()[0] != kSupportedBatch ||
        !is_supported_sequence_bucket(input.shape()[1]) || input.shape()[2] == 0 ||
        input.shape()[2] % 32 != 0 || mask.is_lazy() ||
        !axiom::backends::metal::active_metal_workspace_uses_private_buffers()) {
        return false;
    }

    const size_t time = input.shape()[1];
    const size_t hidden = input.shape()[2];
    const auto &attention = block.attn_;
    const size_t heads = static_cast<size_t>(attention.mha_.num_heads());
    constexpr size_t kDirectHeadDim = 128;
    if (heads == 0 || hidden != heads * kDirectHeadDim ||
        !is_layer_norm_ready(attention.norm_, hidden) ||
        !is_int8_linear_ready(attention.mha_.q_proj(), hidden, hidden) ||
        !is_int8_linear_ready(attention.mha_.k_proj(), hidden, hidden) ||
        !is_int8_linear_ready(attention.mha_.v_proj(), hidden, hidden) ||
        !is_contiguous_gpu_tensor(head_major_position, DType::Float16,
                                  Shape{1, heads,
                                        2 * time - 1,
                                        kDirectHeadDim}) ||
        !is_contiguous_gpu_tensor(attention.pos_bias_u_, DType::Float16,
                                  Shape{heads, kDirectHeadDim}) ||
        !is_contiguous_gpu_tensor(attention.pos_bias_v_, DType::Float16,
                                  Shape{heads, kDirectHeadDim}) ||
        !axiom::backends::metal::
            gpu_relative_position_attention_pipeline_available()) {
        return false;
    }

    if (mask.storage() &&
        (mask.device() != Device::GPU || mask.dtype() != DType::Bool ||
         mask.ndim() != 4 || !mask.is_contiguous() ||
         mask.shape()[0] != kSupportedBatch ||
         (mask.shape()[1] != 1 && mask.shape()[1] != heads) ||
         mask.shape()[2] != time || mask.shape()[3] != time)) {
        return false;
    }

    return true;
}

size_t FastConformerBlockProgram::direct_dispatches_for_testing() {
    return direct_dispatches.load(std::memory_order_relaxed);
}

void FastConformerBlockProgram::reset_trace_for_testing() {
    direct_dispatches.store(0, std::memory_order_relaxed);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, size_t, size_t, size_t>
FastConformerBlockProgram::encode_ffn_pair_for_testing(
    const ConformerBlock &block, const Tensor &input) {
    if (!has_materialized_storage(input) || input.device() != Device::GPU ||
        input.dtype() != DType::Float16 || !input.is_contiguous() ||
        input.shape().size() != 3 || input.shape()[0] != kSupportedBatch ||
        !is_supported_sequence_bucket(input.shape()[1]) || input.shape()[2] == 0 ||
        input.shape()[2] % 32 != 0 ||
        !axiom::backends::metal::active_metal_workspace_uses_private_buffers()) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer FFN program is unavailable for this request");
    }

    const size_t batch = input.shape()[0];
    const size_t time = input.shape()[1];
    const size_t hidden = input.shape()[2];
    const Tensor normalized_flat = input.reshape({batch * time, hidden});
    const auto ffn_ready = [&](const FeedForward &ffn) {
        if (!is_layer_norm_ready(ffn.norm_, hidden) ||
            !is_int8_linear_ready(ffn.fc1_, ffn.fc1_.weight().shape()[0],
                                  hidden)) {
            return false;
        }

        const size_t intermediate = ffn.fc1_.weight().shape()[0];
        if (intermediate == 0 || intermediate % 32 != 0 ||
            !is_int8_linear_ready(ffn.fc2_, hidden, intermediate) ||
            !axiom::backends::metal::gpu_int8_matmul_supports_supplied_output(
                normalized_flat, ffn.fc1_.weight())) {
            return false;
        }

        // The dispatcher reads only dimensions to select its MPSMatrix
        // fallback. Keep this preflight on CPU so rejection happens before a
        // GPU slot is acquired or direct work is encoded.
        const Tensor fc1_shape_probe(
            {batch * time, intermediate}, DType::Float16, Device::CPU);
        return axiom::backends::metal::gpu_int8_matmul_supports_supplied_output(
            fc1_shape_probe, ffn.fc2_.weight());
    };
    if (!ffn_ready(block.ffn1_) || !ffn_ready(block.ffn2_)) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer FFN program is unavailable for this request");
    }

    // The controls use the existing direct FFN body. The focused test enables
    // its existing direct LayerNorm and FFN flags before reaching this helper.
    const Tensor control_first = block.ffn1_(input);
    const Tensor control_second = block.ffn2_(control_first);

    const size_t ffn1_intermediate = block.ffn1_.fc1_.weight().shape()[0];
    const size_t ffn2_intermediate = block.ffn2_.fc1_.weight().shape()[0];
    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    const uint64_t synchronizations_before = stream.synchronization_count();
    FfnSlotPlan slots(input, ffn1_intermediate, ffn2_intermediate);

    const auto encode_direct_ffn = [&](const FeedForward &ffn,
                                       const Tensor &residual,
                                       Tensor normalized, Tensor scratch,
                                       Tensor output) {
        if (!axiom::backends::metal::gpu_layer_norm_into(
                normalized, residual, ffn.norm_.weight(), ffn.norm_.bias(),
                -1, ffn.norm_.eps())) {
            throw axiom::RuntimeError::not_implemented(
                "FastConformer FFN program cannot bind LayerNorm output storage");
        }
        direct_dispatches.fetch_add(1, std::memory_order_relaxed);

        axiom::ops::int8_ffn_silu_residual_into(
            scratch, output, normalized, residual, ffn.fc1_.weight(),
            ffn.fc1_.scale(), ffn.fc1_.bias(), ffn.fc2_.weight(),
            ffn.fc2_.scale(), ffn.fc2_.bias());
        direct_dispatches.fetch_add(1, std::memory_order_relaxed);
        return output;
    };

    // Slot 0 and slot 1 are reused only after the first FFN has encoded its
    // final custom-Metal read. Slot 2 stays live as FFN2's residual input;
    // slot 3 is the final result. No cross-runtime reuse occurs here.
    const Tensor candidate_first =
        encode_direct_ffn(block.ffn1_, input, slots.normalized(),
                          slots.scratch(ffn1_intermediate), slots.first_result());
    const Tensor candidate_second = encode_direct_ffn(
        block.ffn2_, candidate_first, slots.normalized(),
        slots.scratch(ffn2_intermediate), slots.second_result());

    constexpr size_t kLogicalValues = 6; // norm, scratch, result for each FFN
    const size_t cpu_synchronizations =
        static_cast<size_t>(stream.synchronization_count() -
                            synchronizations_before);
    return {control_first, control_second, candidate_first, candidate_second,
            kLogicalValues, slots.slot_count(), cpu_synchronizations};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, size_t, size_t, size_t>
FastConformerBlockProgram::encode_attention_for_testing(
    const ConformerBlock &block, const Tensor &input,
    const Tensor &head_major_position, const Tensor &mask) {
    if (!is_direct_attention_segment_ready(block, input, head_major_position,
                                           mask)) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer direct attention program is unavailable for this request");
    }

    const auto &attention = block.attn_;
    const size_t batch = input.shape()[0];
    const size_t time = input.shape()[1];
    const size_t hidden = input.shape()[2];
    const size_t heads = static_cast<size_t>(attention.mha_.num_heads());
    const size_t head_dim = hidden / heads;
    const Tensor normalized_flat = input.reshape({batch * time, hidden});
    const Tensor bias_u =
        attention.pos_bias_u_.reshape({1, heads, 1, head_dim});
    const Tensor bias_v =
        attention.pos_bias_v_.reshape({1, heads, 1, head_dim});
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Controls use exactly the existing direct kernels with their ordinary
    // allocation behavior. They intentionally do not enter a graph path.
    const Tensor control_normalized = axiom::backends::metal::gpu_layer_norm(
        input, attention.norm_.weight(), attention.norm_.bias(), -1,
        attention.norm_.eps());
    auto control_qkv = axiom::backends::metal::gpu_int8_qkv_matmul_bias(
        control_normalized.reshape({batch * time, hidden}),
        attention.mha_.q_proj().weight(),
        attention.mha_.q_proj().scale(), attention.mha_.q_proj().bias(),
        attention.mha_.k_proj().weight(), attention.mha_.k_proj().scale(),
        attention.mha_.k_proj().bias(), attention.mha_.v_proj().weight(),
        attention.mha_.v_proj().scale(), attention.mha_.v_proj().bias(), batch,
        time, heads);
    const Tensor control_attention =
        axiom::backends::metal::gpu_relative_position_attention(
            control_qkv[0], control_qkv[1], control_qkv[2],
            head_major_position, bias_u, bias_v, mask, scale);

    auto &stream = axiom::backends::metal::MetalExecutionStream::instance();
    const uint64_t synchronizations_before = stream.synchronization_count();
    AttentionSlotPlan slots(input, heads);
    Tensor candidate_normalized = slots.normalized();
    Tensor candidate_q = slots.query();
    Tensor candidate_k = slots.key();
    Tensor candidate_v = slots.value();
    Tensor candidate_attention = slots.attention_result(candidate_normalized);

    if (!axiom::backends::metal::gpu_layer_norm_into(
            candidate_normalized, input, attention.norm_.weight(),
            attention.norm_.bias(), -1, attention.norm_.eps())) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer direct attention cannot bind LayerNorm output storage");
    }
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);

    if (!axiom::backends::metal::gpu_int8_qkv_matmul_bias_into(
            candidate_q, candidate_k, candidate_v,
            candidate_normalized.reshape({batch * time, hidden}),
            attention.mha_.q_proj().weight(), attention.mha_.q_proj().scale(),
            attention.mha_.q_proj().bias(), attention.mha_.k_proj().weight(),
            attention.mha_.k_proj().scale(), attention.mha_.k_proj().bias(),
            attention.mha_.v_proj().weight(), attention.mha_.v_proj().scale(),
            attention.mha_.v_proj().bias(), batch, time, heads)) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer direct attention cannot bind QKV output storage");
    }
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);

    if (!axiom::backends::metal::gpu_relative_position_attention_into(
            candidate_attention, candidate_q, candidate_k, candidate_v,
            head_major_position, bias_u, bias_v, mask, scale)) {
        throw axiom::RuntimeError::not_implemented(
            "FastConformer direct attention cannot bind attention output storage");
    }
    direct_dispatches.fetch_add(1, std::memory_order_relaxed);

    constexpr size_t kLogicalValues = 5; // norm, Q, K, V, attention output
    const size_t cpu_synchronizations =
        static_cast<size_t>(stream.synchronization_count() -
                            synchronizations_before);
    return {control_qkv[0], control_qkv[1], control_qkv[2], control_attention,
            candidate_normalized, candidate_q, candidate_k, candidate_v,
            candidate_attention, kLogicalValues, slots.slot_count(),
            cpu_synchronizations};
}

} // namespace parakeet::models
