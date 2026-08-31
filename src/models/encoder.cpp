#include "parakeet/models/encoder.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <axiom/error.hpp>
#include <axiom/graph/graph_registry.hpp>
#include <axiom/nn/positional.hpp>
#include <axiom/workspace.hpp>

#include "backends/metal/metal_operations.hpp"
#include "backends/metal/metal_workspace_cache.hpp"
#include "parakeet/models/fastconformer_block_program.hpp"
#include "parakeet/profile/signposts.hpp"

namespace parakeet::models {

namespace {

// Diagnostic-only forward-local counter used to localize numerical divergence
// from the experimental FP16 pointwise+GLU kernel. It is reset before every
// encoder forward and has no effect unless PARAKEET_FUSED_F16_POINTWISE_GLU is
// enabled. Production leaves the feature flag unset.
thread_local size_t fused_f16_pointwise_glu_calls_this_forward = 0;
thread_local size_t direct_f16_pointwise_calls_this_forward = 0;
thread_local size_t direct_f16_glu_calls_this_forward = 0;
thread_local size_t direct_f16_glu_comparison_calls_this_forward = 0;
thread_local float direct_f16_glu_max_abs_error_this_forward = 0.0f;
thread_local float direct_f16_glu_default_exp_max_abs_error_this_forward =
    0.0f;
thread_local float direct_f16_glu_rounded_sigmoid_max_abs_error_this_forward =
    0.0f;

axiom::graph::ExecutionScheduleStorageUse schedule_storage_use(
    const Tensor &tensor) {
    auto storage = tensor.storage();
    return axiom::graph::ExecutionScheduleStorageUse{
        storage.get(), tensor.nbytes(), tensor.dtype(), std::move(storage)};
}

void validate_execution_plan_direct_route(
    axiom::graph::ExecutionScheduleDirectRoute route) {
    if (axiom::graph::execution_plan_replay_active() &&
        !axiom::graph::consume_active_execution_plan_direct(route)) {
        throw axiom::RuntimeError(
            "whole encoder execution-plan replay rejected a direct-Metal "
            "route");
    }
}

bool direct_int8_ffn_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_INT8_FFN");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool direct_int8_silu_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_INT8_SILU");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool direct_int8_glu_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_INT8_GLU");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool direct_f16_glu_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_F16_GLU");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

// Test-only real-activation comparison. The candidate remains disabled by
// default, and this separate flag deliberately adds a reference MPSGraph GLU
// plus host readback only when a focused diagnostic requests it.
bool capture_direct_f16_glu_error_enabled() {
    const char *env = std::getenv("PARAKEET_CAPTURE_DIRECT_F16_GLU_ERROR");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

float max_abs_error_on_cpu(const Tensor &actual, const Tensor &expected) {
    const Tensor actual_cpu = actual.to(Device::CPU).astype(DType::Float32);
    const Tensor expected_cpu = expected.to(Device::CPU).astype(DType::Float32);
    if (actual_cpu.shape() != expected_cpu.shape()) {
        throw axiom::RuntimeError(
            "direct FP16 GLU comparison received mismatched shapes");
    }
    const auto *actual_data = actual_cpu.typed_data<float>();
    const auto *expected_data = expected_cpu.typed_data<float>();
    float maximum = 0.0f;
    for (size_t index = 0; index < actual_cpu.numel(); ++index) {
        maximum = std::max(maximum,
                           std::fabs(actual_data[index] - expected_data[index]));
    }
    return maximum;
}

// Diagnostic-only: execute the custom FP16 pointwise projection, then pass
// its materialized output through the ordinary GLU implementation. This
// isolates projection reduction error from the separately tested GLU path.
bool direct_f16_pointwise_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_F16_POINTWISE");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool fused_int8_pointwise_glu_enabled() {
    const char *env = std::getenv("PARAKEET_FUSED_INT8_POINTWISE_GLU");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool fused_f16_pointwise_glu_enabled() {
    const char *env = std::getenv("PARAKEET_FUSED_F16_POINTWISE_GLU");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

// Laboratory-only graph-topology switch.  The released INT8 artifact does
// not contain pointwise INT8 weights, even when the historical fused-route
// environment flag is set.  In that case constructing the [B,T,C] candidate
// view gives its source a second lazy consumer; the graph compiler must then
// materialize a standalone transpose before the normal Conv/GLU graph.  Keep
// the old construction unless a caller explicitly asks to defer an
// unreachable candidate, so A/B timing can isolate only that graph boundary.
bool defer_unreachable_pointwise_view_enabled() {
    const char *env = std::getenv("PARAKEET_DEFER_UNREACHABLE_POINTWISE_VIEW");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool should_build_pointwise_matrix_input(const Conv1d &pointwise_conv) {
    if (!defer_unreachable_pointwise_view_enabled()) {
        return true;
    }
    return (fused_int8_pointwise_glu_enabled() &&
            pointwise_conv.has_int8_pointwise_weights()) ||
           fused_f16_pointwise_glu_enabled() || direct_f16_pointwise_enabled();
}

size_t fused_f16_pointwise_glu_limit() {
    const char *env =
        std::getenv("PARAKEET_FUSED_F16_POINTWISE_GLU_LIMIT");
    if (env == nullptr || *env == '\0') {
        return std::numeric_limits<size_t>::max();
    }

    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return 0;
    }
    return static_cast<size_t>(parsed);
}

size_t direct_f16_pointwise_limit() {
    const char *env = std::getenv("PARAKEET_DIRECT_F16_POINTWISE_LIMIT");
    if (env == nullptr || *env == '\0') {
        return std::numeric_limits<size_t>::max();
    }

    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return 0;
    }
    return static_cast<size_t>(parsed);
}

bool direct_int8_residual_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_INT8_RESIDUAL");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool direct_depthwise_conv_batch_norm_silu_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_DEPTHWISE_CONV_BN_SILU");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool can_use_direct_depthwise_conv_batch_norm_silu(
    const Tensor &activation, const Conv1d &depthwise_conv,
    const BatchNorm1d &batch_norm) {
    if (!direct_depthwise_conv_batch_norm_silu_enabled() ||
        activation.device() != Device::GPU ||
        activation.dtype() != DType::Float16 || activation.ndim() != 3 ||
        activation.shape()[0] == 0 || activation.shape()[1] == 0 ||
        activation.shape()[2] == 0 || depthwise_conv.padding() != 4 ||
        depthwise_conv.groups() != static_cast<int>(activation.shape()[1])) {
        return false;
    }

    const size_t channels = activation.shape()[1];
    const auto is_half_gpu_vector = [&](const Tensor &tensor) {
        return tensor.device() == Device::GPU &&
               tensor.dtype() == DType::Float16 && tensor.ndim() == 1 &&
               tensor.shape() == Shape{channels};
    };
    return depthwise_conv.weight().device() == Device::GPU &&
           depthwise_conv.weight().dtype() == DType::Float16 &&
           depthwise_conv.weight().shape() == Shape{channels, 1, 9} &&
           is_half_gpu_vector(batch_norm.running_mean()) &&
           is_half_gpu_vector(batch_norm.running_var()) &&
           is_half_gpu_vector(batch_norm.weight()) &&
           is_half_gpu_vector(batch_norm.bias()) && batch_norm.eps() > 0.0f;
}

bool cache_position_projections_enabled() {
    const char *env = std::getenv("PARAKEET_CACHE_POSITION_PROJECTIONS");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool int8_position_projections_enabled() {
    const char *env = std::getenv("PARAKEET_INT8_POSITION_PROJECTIONS");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool cache_position_head_layout_enabled() {
    const char *env = std::getenv("PARAKEET_CACHE_POSITION_HEAD_LAYOUT");
    return cache_position_projections_enabled() && env != nullptr &&
           env[0] == '1' && env[1] == '\0';
}

bool can_use_direct_int8_silu(const Tensor &activation) {
    return direct_int8_silu_enabled() && activation.device() == Device::GPU &&
           activation.dtype() == DType::Float16 && activation.ndim() == 3 &&
           activation.is_contiguous() && activation.shape()[0] > 0 &&
           activation.shape()[1] > 0 && activation.shape()[2] > 0;
}

bool can_use_direct_int8_glu(const Tensor &activation) {
    if (!direct_int8_glu_enabled() || activation.device() != Device::GPU ||
        activation.dtype() != DType::Float16 || activation.ndim() != 3 ||
        activation.shape()[0] == 0 || activation.shape()[1] == 0 ||
        activation.shape()[1] % 2 != 0 || activation.shape()[2] == 0 ||
        activation.is_contiguous()) {
        return false;
    }

    const size_t channels = activation.shape()[1] / 2;
    return activation.strides() ==
           Strides{static_cast<int64_t>(activation.shape()[2] * channels * 2 *
                                        activation.itemsize()),
                   static_cast<int64_t>(activation.itemsize()),
                   static_cast<int64_t>(channels * 2 *
                                        activation.itemsize())};
}

bool can_use_direct_f16_glu(const Tensor &activation,
                            const Conv1d &pointwise_conv) {
    return direct_f16_glu_enabled() && activation.device() == Device::GPU &&
           activation.dtype() == DType::Float16 && activation.ndim() == 3 &&
           activation.is_contiguous() && activation.shape()[0] > 0 &&
           activation.shape()[1] > 0 && activation.shape()[1] % 2 == 0 &&
           activation.shape()[2] > 0 &&
           pointwise_conv.weight().dtype() == DType::Float16;
}

bool can_use_fused_int8_pointwise_glu(const Tensor &matrix_input,
                                      const Conv1d &pointwise_conv) {
    if (!fused_int8_pointwise_glu_enabled() ||
        matrix_input.device() != Device::GPU ||
        matrix_input.dtype() != DType::Float16 || matrix_input.ndim() != 3 ||
        matrix_input.shape()[0] == 0 || matrix_input.shape()[1] == 0 ||
        matrix_input.shape()[2] == 0 ||
        !pointwise_conv.has_int8_pointwise_weights()) {
        return false;
    }

    const size_t channels = matrix_input.shape()[2];
    const Tensor &weight = pointwise_conv.weight();
    const Tensor &scale = pointwise_conv.scale();
    const Tensor &bias = pointwise_conv.bias();
    // Match the operation boundary: the generic INT8 Conv1d path promotes
    // model weights to Metal on demand, and Parakeet's pointwise projection
    // intentionally has no bias.  Do not make this fast path depend on a
    // particular loader residency or on an optional tensor being present.
    return channels % 32 == 0 && weight.dtype() == DType::Int8 &&
           weight.shape() == Shape{channels * 2, channels} &&
           scale.dtype() == DType::Float16 &&
           scale.shape() == Shape{channels * 2, channels / 32} &&
           (!bias.storage() ||
            (bias.dtype() == DType::Float16 && bias.ndim() == 1 &&
             bias.shape() == Shape{channels * 2}));
}

bool can_use_fused_f16_pointwise_glu(const Tensor &matrix_input,
                                      const Conv1d &pointwise_conv) {
    if (!fused_f16_pointwise_glu_enabled() ||
        matrix_input.device() != Device::GPU ||
        matrix_input.dtype() != DType::Float16 || matrix_input.ndim() != 3 ||
        matrix_input.shape()[0] == 0 || matrix_input.shape()[1] == 0 ||
        matrix_input.shape()[2] == 0) {
        return false;
    }

    const size_t channels = matrix_input.shape()[2];
    const Tensor &weight = pointwise_conv.weight();
    const Tensor &bias = pointwise_conv.bias();
    // The model keeps a Conv1D [2C,C,1] kernel. The Metal operation consumes
    // its layout-preserving [2C,C] matrix view and, like the generic Conv1D
    // path, permits model parameters to be promoted from CPU on first use.
    const bool eligible =
        channels % 32 == 0 && weight.dtype() == DType::Float16 &&
        weight.shape() == Shape{channels * 2, channels, 1} &&
        (!bias.storage() ||
         (bias.dtype() == DType::Float16 && bias.ndim() == 1 &&
          bias.shape() == Shape{channels * 2}));
    if (!eligible || fused_f16_pointwise_glu_calls_this_forward >=
                         fused_f16_pointwise_glu_limit()) {
        return false;
    }
    ++fused_f16_pointwise_glu_calls_this_forward;
    return true;
}

bool can_use_direct_f16_pointwise(const Tensor &matrix_input,
                                  const Conv1d &pointwise_conv) {
    if (!direct_f16_pointwise_enabled() || matrix_input.device() != Device::GPU ||
        matrix_input.dtype() != DType::Float16 || matrix_input.ndim() != 3 ||
        matrix_input.shape()[0] == 0 || matrix_input.shape()[1] == 0 ||
        matrix_input.shape()[2] == 0) {
        return false;
    }

    const size_t channels = matrix_input.shape()[2];
    const Tensor &weight = pointwise_conv.weight();
    const Tensor &bias = pointwise_conv.bias();
    const bool eligible =
        channels % 32 == 0 && weight.dtype() == DType::Float16 &&
        weight.shape() == Shape{channels * 2, channels, 1} &&
        (!bias.storage() ||
         (bias.dtype() == DType::Float16 && bias.ndim() == 1 &&
          bias.shape() == Shape{channels * 2}));
    if (!eligible || direct_f16_pointwise_calls_this_forward >=
                         direct_f16_pointwise_limit()) {
        return false;
    }
    ++direct_f16_pointwise_calls_this_forward;
    return true;
}

bool can_use_direct_int8_residual(const Tensor &residual,
                                  const Tensor &output) {
    return direct_int8_residual_enabled() &&
           residual.device() == Device::GPU && output.device() == Device::GPU &&
           residual.dtype() == DType::Float16 && output.dtype() == DType::Float16 &&
           residual.ndim() == 3 && output.ndim() == 3 &&
           residual.shape() == output.shape() && residual.is_contiguous() &&
           output.is_contiguous() && !residual.is_lazy() && !output.is_lazy();
}

bool direct_int8_qkv_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_INT8_QKV");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

bool direct_int8_qkv_head_layout_enabled() {
    const char *env = std::getenv("PARAKEET_DIRECT_INT8_QKV_HEAD_LAYOUT");
    return direct_int8_qkv_enabled() && env != nullptr && env[0] == '1' &&
           env[1] == '\0';
}

bool relative_position_attention_enabled() {
    const char *env = std::getenv("PARAKEET_RELATIVE_POSITION_ATTENTION");
    if (env == nullptr || env[0] != '1' || env[1] != '\0') {
        return false;
    }

    // Narrow diagnostic switch for local parity work: cap the number of
    // native relative-attention calls in this process so a full encoder output
    // can show whether numerical differences grow block by block. Production
    // leaves this unset and uses the kernel in every eligible attention block.
    const char *limit_env =
        std::getenv("PARAKEET_RELATIVE_POSITION_ATTENTION_LIMIT");
    if (limit_env == nullptr || limit_env[0] == '\0') {
        return true;
    }
    const int limit = std::atoi(limit_env);
    if (limit <= 0) {
        return true;
    }
    static thread_local int native_calls = 0;
    return native_calls++ < limit;
}

bool can_use_direct_int8_qkv(const Tensor &query, const Tensor &key,
                             const Tensor &value, const Linear &q_proj,
                             const Linear &k_proj, const Linear &v_proj) {
    const bool enabled = direct_int8_qkv_enabled();
    const bool input_valid = query.device() == Device::GPU &&
                             query.dtype() == DType::Float16 &&
                             query.ndim() == 3 && query.is_contiguous();
    const bool self_attention = query.shares_storage(key) &&
                                query.shares_storage(value);
    const bool metadata_valid = q_proj.has_scale() && k_proj.has_scale() &&
                                v_proj.has_scale();
    if (!enabled || !input_valid || !self_attention || !metadata_valid) {
        return false;
    }

    const size_t hidden = query.shape()[2];
    const auto valid_projection = [&](const Linear &projection) {
        return projection.weight().device() == Device::GPU &&
               projection.weight().dtype() == DType::Int8 &&
               projection.weight().shape() == Shape{hidden, hidden} &&
               projection.scale().device() == Device::GPU &&
               projection.scale().dtype() == DType::Float16 &&
               projection.scale().shape() == Shape{hidden, hidden / 32} &&
               (!projection.bias().storage() ||
                (projection.bias().device() == Device::GPU &&
                 projection.bias().dtype() == DType::Float16 &&
                 projection.bias().shape() == Shape{hidden}));
    };
    return hidden > 0 && hidden % 32 == 0 && valid_projection(q_proj) &&
           valid_projection(k_proj) && valid_projection(v_proj);
}

bool safe_direct_qkv_persistent_slots_enabled() {
    const char *env =
        std::getenv("WASPER_PARAKEET_SAFE_DIRECT_QKV_PERSISTENT_SLOTS");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

class SafeDirectQkvPersistentPlan;
thread_local SafeDirectQkvPersistentPlan *active_safe_direct_qkv_plan =
    nullptr;

// This owns storage only. It deliberately does not alter the direct-QKV
// kernel, attention graph, or MPSGraph partition. The plan persists every
// Q/K/V value through the complete lazy encoder graph and transfers its lease
// only once MPSGraph can fence the final graph read with a descriptor event.
class SafeDirectQkvPersistentPlan {
  public:
    SafeDirectQkvPersistentPlan(const Tensor &subsampled_input,
                                size_t layer_count, size_t num_heads)
        : qkv_shape_{subsampled_input.shape()[0], num_heads,
                     subsampled_input.shape()[1],
                     subsampled_input.shape()[2] / num_heads},
          qkv_bytes_(subsampled_input.nbytes()),
          expected_slots_(layer_count * 3) {
        if (active_safe_direct_qkv_plan != nullptr || layer_count == 0 ||
            num_heads == 0 || subsampled_input.device() != Device::GPU ||
            subsampled_input.dtype() != DType::Float16 ||
            subsampled_input.ndim() != 3 ||
            !subsampled_input.is_contiguous() ||
            subsampled_input.shape()[2] % num_heads != 0 || qkv_bytes_ == 0 ||
            !axiom::backends::metal::active_metal_workspace_uses_private_buffers() ||
            !axiom::backends::metal::active_metal_workspace_supports_persistent_slot_bucket() ||
            axiom::backends::metal::has_active_persistent_slot_handoff()) {
            throw RuntimeError(
                "Safe direct QKV persistent plan received an ineligible forward");
        }

        auto table = axiom::backends::metal::acquire_active_persistent_workspace_slot_table(
            std::vector<size_t>(expected_slots_, qkv_bytes_));
        lease_ = table->acquire_lease();
        if (!lease_) {
            throw RuntimeError(
                "Safe direct QKV persistent plan could not acquire slot backing");
        }
        storage_views_.reserve(expected_slots_);
        active_safe_direct_qkv_plan = this;
    }

    ~SafeDirectQkvPersistentPlan() {
        if (active_safe_direct_qkv_plan == this) {
            active_safe_direct_qkv_plan = nullptr;
        }
    }

    SafeDirectQkvPersistentPlan(const SafeDirectQkvPersistentPlan &) = delete;
    SafeDirectQkvPersistentPlan &operator=(
        const SafeDirectQkvPersistentPlan &) = delete;

    static SafeDirectQkvPersistentPlan *active() {
        return active_safe_direct_qkv_plan;
    }

    std::array<Tensor, 3> next_outputs() {
        if (!lease_ || next_slot_ + 3 > expected_slots_) {
            throw RuntimeError(
                "Safe direct QKV persistent plan exhausted its declared slots");
        }

        std::array<Tensor, 3> outputs;
        for (Tensor &output : outputs) {
            auto storage = lease_->make_storage_view(next_slot_++, qkv_bytes_);
            storage_views_.push_back(storage);
            output = Tensor(
                storage, qkv_shape_,
                ShapeUtils::calculate_strides(qkv_shape_,
                                              axiom::dtype_size(DType::Float16)),
                DType::Float16);
        }
        return outputs;
    }

    void handoff_complete_plan(Tensor &terminal_output) {
        if (!lease_ || next_slot_ != expected_slots_ ||
            storage_views_.size() != expected_slots_) {
            throw RuntimeError(
                "Safe direct QKV persistent plan did not bind every layer output");
        }
        for (size_t index = 0; index < storage_views_.size(); ++index) {
            const auto duplicate = std::find(storage_views_.begin(),
                                             storage_views_.begin() + index,
                                             storage_views_[index]);
            if (duplicate != storage_views_.begin() + index) {
                throw RuntimeError(
                    "Safe direct QKV persistent plan repeated storage view at "
                    "slots " +
                    std::to_string(std::distance(storage_views_.begin(), duplicate)) +
                    " and " + std::to_string(index));
            }
        }
        axiom::backends::metal::register_active_persistent_slot_handoff(
            std::move(lease_), std::move(storage_views_));
        if (!terminal_output.is_lazy()) {
            axiom::backends::metal::abandon_active_persistent_slot_handoff();
            throw RuntimeError(
                "Safe direct QKV persistent plan requires a lazy terminal encoder output");
        }
        // Direct QKV kernels force earlier blocks to materialize while they
        // acquire each next activation. Materializing this final output now
        // gives the handoff one known terminal MPSGraph transaction whose
        // descriptor completion event follows every prior request command.
        static_cast<void>(terminal_output.storage());
    }

  private:
    Shape qkv_shape_;
    size_t qkv_bytes_ = 0;
    size_t expected_slots_ = 0;
    size_t next_slot_ = 0;
    std::unique_ptr<axiom::backends::metal::PersistentWorkspaceSlotLease>
        lease_;
    std::vector<std::shared_ptr<Storage>> storage_views_;
};

bool can_begin_safe_direct_qkv_persistent_plan(const Tensor &subsampled_input,
                                                size_t layer_count,
                                                size_t num_heads) {
    return safe_direct_qkv_persistent_slots_enabled() &&
           direct_int8_qkv_enabled() &&
           direct_int8_qkv_head_layout_enabled() && layer_count > 0 &&
           num_heads > 0 && !relative_position_attention_enabled() &&
           !axiom::graph::is_hybrid_metal_qkv_execution_enabled() &&
           subsampled_input.device() == Device::GPU &&
           subsampled_input.dtype() == DType::Float16 &&
           subsampled_input.ndim() == 3 && subsampled_input.is_contiguous() &&
           subsampled_input.shape()[2] % num_heads == 0 &&
           axiom::backends::metal::active_metal_workspace_uses_private_buffers() &&
           axiom::backends::metal::active_metal_workspace_supports_persistent_slot_bucket() &&
           !axiom::backends::metal::has_active_persistent_slot_handoff();
}

bool can_use_direct_int8_ffn(const Tensor &input, const Tensor &normalized,
                             const Linear &fc1, const Linear &fc2) {
    if (!direct_int8_ffn_enabled() || input.device() != Device::GPU ||
        input.dtype() != DType::Float16 || input.ndim() != 3 ||
        !input.is_contiguous() || normalized.device() != Device::GPU ||
        normalized.dtype() != DType::Float16 || normalized.ndim() != 3 ||
        !normalized.is_contiguous() || normalized.shape() != input.shape() ||
        !fc1.has_scale() || !fc2.has_scale() ||
        fc1.weight().device() != Device::GPU ||
        fc1.weight().dtype() != DType::Int8 ||
        fc1.scale().device() != Device::GPU ||
        fc1.scale().dtype() != DType::Float16 ||
        fc2.weight().device() != Device::GPU ||
        fc2.weight().dtype() != DType::Int8 ||
        fc2.scale().device() != Device::GPU ||
        fc2.scale().dtype() != DType::Float16) {
        return false;
    }

    const size_t hidden = input.shape()[2];
    const size_t intermediate = fc1.weight().shape()[0];
    const auto valid_optional_bias = [](const Tensor &bias,
                                        const Shape &expected_shape) {
        return !bias.storage() ||
               (bias.device() == Device::GPU && bias.dtype() == DType::Float16 &&
                bias.shape() == expected_shape);
    };
    return hidden > 0 && hidden % 32 == 0 && intermediate > 0 &&
           intermediate % 32 == 0 && fc1.weight().ndim() == 2 &&
           fc1.weight().shape()[1] == hidden &&
           fc1.scale().shape() == Shape{intermediate, hidden / 32} &&
           valid_optional_bias(fc1.bias(), Shape{intermediate}) &&
           fc2.weight().ndim() == 2 &&
           fc2.weight().shape() == Shape{hidden, intermediate} &&
           fc2.scale().shape() == Shape{hidden, intermediate / 32} &&
           valid_optional_bias(fc2.bias(), Shape{hidden});
}

} // namespace

namespace detail {

size_t position_projection_cache_capacity() {
    // A single entry thrashes when requests rotate through several exact
    // position shapes. Nine retains the small active shape set while keeping
    // long-form memory growth strictly bounded.
    constexpr size_t kDefaultEntries = 9;
    constexpr size_t kMaximumEntries = 9;
    const char *env =
        std::getenv("PARAKEET_POSITION_PROJECTION_CACHE_ENTRIES");
    if (env == nullptr || *env == '\0') return kDefaultEntries;

    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || parsed == 0 ||
        parsed > kMaximumEntries ||
        parsed > std::numeric_limits<size_t>::max()) {
        return kDefaultEntries;
    }
    return static_cast<size_t>(parsed);
}

} // namespace detail

// ─── FeedForward ────────────────────────────────────────────────────────────

FeedForward::FeedForward(float dropout, bool bias)
    : fc1_(bias), fc2_(bias), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, fc1_, fc2_, dropout_);
}

void FeedForward::load_int8_weights(Tensor fc1_w_int8, Tensor fc1_w_scale,
                                    Tensor fc2_w_int8, Tensor fc2_w_scale) {
    fc1_.load_int8_weights(fc1_w_int8, fc1_w_scale);
    fc2_.load_int8_weights(fc2_w_int8, fc2_w_scale);
    // is_int8() is now derived from fc1_.has_scale() + dtype; no flag to set.
}

Device FeedForward::int8_weights_device() const {
    // scale_ is registered as a Module parameter by Linear::load_int8_weights,
    // so Module::to(Device) migrates it correctly — no override needed.
    return fc1_.has_scale() ? fc1_.scale().device() : Device::CPU;
}

bool FeedForward::all_int8_on(Device d) const {
    // Tensors with no storage abstain (vote true vacuously). On a non-int8
    // FeedForward no scales are loaded, so this always returns true.
    // On a fully loaded int8 FeedForward, all four tensors must agree with `d`.
    auto on = [&](const Tensor &t) {
        return !t.storage() || t.device() == d;
    };
    return on(fc1_.weight()) && on(fc1_.scale()) &&
           on(fc2_.weight()) && on(fc2_.scale());
}

Tensor FeedForward::forward(const Tensor &input) const {
    auto x = norm_(input);
    if (can_use_direct_int8_ffn(input, x, fc1_, fc2_)) {
        auto output = ops::int8_ffn_silu_residual(
            x, input, fc1_.weight(), fc1_.scale(), fc1_.bias(),
            fc2_.weight(), fc2_.scale(), fc2_.bias());
        axiom::graph::record_direct_execution_schedule_segment(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Ffn);
        axiom::graph::record_execution_schedule_direct_uses(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Ffn,
            {schedule_storage_use(x), schedule_storage_use(input)},
            {schedule_storage_use(output)}, /*outputs_alias_inputs=*/false);
        validate_execution_plan_direct_route(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Ffn);
        return output;
    }
    x = fc1_(x);
    if (can_use_direct_int8_silu(x)) {
        const Shape activation_shape = x.shape();
        Tensor activation_flat =
            x.reshape({activation_shape[0] * activation_shape[1],
                       activation_shape[2]});
        ops::int8_silu_inplace(activation_flat);
        axiom::graph::record_direct_execution_schedule_segment(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Silu);
        axiom::graph::record_execution_schedule_direct_uses(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Silu,
            {schedule_storage_use(activation_flat)},
            {schedule_storage_use(activation_flat)},
            /*outputs_alias_inputs=*/true);
        validate_execution_plan_direct_route(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Silu);
        x = activation_flat.reshape(activation_shape);
    } else {
        x = ops::silu(x);
    }
    x = dropout_(x);
    x = fc2_(x);
    return input + x * 0.5f; // macaron half-step
}

// ─── ConformerConvModule ────────────────────────────────────────────────────

ConformerConvModule::ConformerConvModule(int groups, float dropout)
    : pointwise_conv1_(/*stride=*/1),
      depthwise_conv_(/*stride=*/1, /*padding=*/4, /*dilation=*/1,
                      /*groups=*/groups),
      pointwise_conv2_(/*stride=*/1), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, pointwise_conv1_, depthwise_conv_, batch_norm_,
                        pointwise_conv2_, dropout_);
}

void ConformerConvModule::load_int8_pointwise_weights(
    Tensor pointwise1_int8, Tensor pointwise1_scale,
    Tensor pointwise2_int8, Tensor pointwise2_scale) {
    pointwise_conv1_.load_int8_pointwise_weights(pointwise1_int8,
                                                  pointwise1_scale);
    pointwise_conv2_.load_int8_pointwise_weights(pointwise2_int8,
                                                  pointwise2_scale);
}

Tensor ConformerConvModule::forward(const Tensor &input) const {
    auto x = norm_(input);
    x = x.permute({0, 2, 1}); // (batch, hidden, seq) for conv1d

    bool used_pointwise_candidate = false;
    if (should_build_pointwise_matrix_input(pointwise_conv1_)) {
        Tensor pointwise_input = x.permute({0, 2, 1}); // (batch, seq, hidden)
        if (can_use_fused_int8_pointwise_glu(pointwise_input,
                                              pointwise_conv1_)) {
            x = ops::fastconformer_int8_pointwise_glu_f16(
                pointwise_input.ascontiguousarray(), pointwise_conv1_.weight(),
                pointwise_conv1_.scale(), pointwise_conv1_.bias());
            used_pointwise_candidate = true;
        } else if (can_use_fused_f16_pointwise_glu(pointwise_input,
                                                    pointwise_conv1_)) {
            const size_t channels = pointwise_input.shape()[2];
            x = ops::fastconformer_f16_pointwise_glu_f16(
                pointwise_input.ascontiguousarray(),
                pointwise_conv1_.weight().reshape({channels * 2, channels}),
                pointwise_conv1_.bias());
            used_pointwise_candidate = true;
        } else if (can_use_direct_f16_pointwise(pointwise_input,
                                                 pointwise_conv1_)) {
            const size_t channels = pointwise_input.shape()[2];
            Tensor projected = ops::fastconformer_f16_pointwise_f16(
                pointwise_input.ascontiguousarray(),
                pointwise_conv1_.weight().reshape({channels * 2, channels}),
                pointwise_conv1_.bias());
            // Match the generic Conv1D output layout before invoking the normal
            // GLU, so the experiment changes only the pointwise reduction.
            x = ops::glu(projected.permute({0, 2, 1}).ascontiguousarray(),
                         /*dim=*/1);
            used_pointwise_candidate = true;
        }
    }
    if (!used_pointwise_candidate) {
        x = pointwise_conv1_(x); // (batch, 2*hidden, seq)
        if (can_use_direct_int8_glu(x)) {
            x = ops::fastconformer_glu_f16(x);
        } else if (can_use_direct_f16_glu(x, pointwise_conv1_)) {
            ++direct_f16_glu_calls_this_forward;
            // Keep a second handle to the exact MPSGraph projection. The
            // direct operation materializes this projection first; the
            // diagnostic reference then sees that same concrete activation.
            // It is intentionally not part of the normal candidate route.
            const Tensor projected = x;
            Tensor direct = ops::fastconformer_glu_channels_first_f16(projected);
            if (capture_direct_f16_glu_error_enabled()) {
                const Tensor default_exp =
                    ops::fastconformer_glu_channels_first_f16_default_exp(
                        projected);
                const Tensor rounded_sigmoid =
                    ops::fastconformer_glu_channels_first_f16_rounded_sigmoid(
                        projected);
                const Tensor reference = ops::glu(projected, /*dim=*/1);
                ++direct_f16_glu_comparison_calls_this_forward;
                direct_f16_glu_max_abs_error_this_forward = std::max(
                    direct_f16_glu_max_abs_error_this_forward,
                    max_abs_error_on_cpu(direct, reference));
                direct_f16_glu_default_exp_max_abs_error_this_forward = std::max(
                    direct_f16_glu_default_exp_max_abs_error_this_forward,
                    max_abs_error_on_cpu(default_exp, reference));
                direct_f16_glu_rounded_sigmoid_max_abs_error_this_forward =
                    std::max(
                        direct_f16_glu_rounded_sigmoid_max_abs_error_this_forward,
                        max_abs_error_on_cpu(rounded_sigmoid, reference));
            }
            x = std::move(direct);
        } else {
            x = ops::glu(x, /*dim=*/1);
        }
    }

    if (can_use_direct_depthwise_conv_batch_norm_silu(
            x, depthwise_conv_, batch_norm_)) {
        x = ops::depthwise_conv1d_batch_norm_silu(
            x, depthwise_conv_.weight(), batch_norm_.running_mean(),
            batch_norm_.running_var(), batch_norm_.weight(),
            batch_norm_.bias(), batch_norm_.eps());
    } else {
        x = depthwise_conv_(x);
        x = batch_norm_(x);
        x = ops::silu(x);
    }

    x = pointwise_conv2_(x);
    x = dropout_(x);
    x = x.permute({0, 2, 1}); // back to (batch, seq, hidden)

    if (can_use_direct_int8_residual(input, x)) {
        ops::int8_add_residual_inplace(x, input);
        axiom::graph::record_direct_execution_schedule_segment(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Residual);
        axiom::graph::record_execution_schedule_direct_uses(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Residual,
            {schedule_storage_use(x), schedule_storage_use(input)},
            {schedule_storage_use(x)}, /*outputs_alias_inputs=*/true);
        validate_execution_plan_direct_route(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Residual);
        return x;
    }
    return input + x;
}

// ─── ConformerAttention ─────────────────────────────────────────────────────

ConformerAttention::ConformerAttention(int num_heads, float dropout)
    : mha_(num_heads), pos_proj_(false), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, mha_, pos_proj_, dropout_);
    AX_REGISTER_PARAMETERS(pos_bias_u_, pos_bias_v_);
}

void ConformerAttention::load_int8_weights(Tensor q_int8, Tensor q_scale,
                                           Tensor k_int8, Tensor k_scale,
                                           Tensor v_int8, Tensor v_scale,
                                           Tensor o_int8, Tensor o_scale) {
    // const_cast: the mha_ accessors expose const Linear & for read access,
    // but load_int8_weights is a one-time setup mutation that registers scale_
    // as a Module parameter inside Linear. Safe because mha_ is our own member.
    const_cast<Linear &>(mha_.q_proj()).load_int8_weights(q_int8, q_scale);
    const_cast<Linear &>(mha_.k_proj()).load_int8_weights(k_int8, k_scale);
    const_cast<Linear &>(mha_.v_proj()).load_int8_weights(v_int8, v_scale);
    const_cast<Linear &>(mha_.out_proj()).load_int8_weights(o_int8, o_scale);
    // is_int8() is now derived from mha_.q_proj().has_scale() + dtype; no flag to set.
}

void ConformerAttention::load_int8_position_projection_weights(
    Tensor weights, Tensor scale) {
    pos_proj_.load_int8_weights(std::move(weights), std::move(scale));
}

Device ConformerAttention::int8_weights_device() const {
    // scale_ is registered as a Module parameter by Linear::load_int8_weights,
    // so Module::to(Device) migrates it correctly — no override needed.
    return mha_.q_proj().has_scale() ? mha_.q_proj().scale().device()
                                     : Device::CPU;
}

bool ConformerAttention::all_int8_on(Device d) const {
    // See FeedForward::all_int8_on() for the abstention rule. On a fully
    // loaded int8 ConformerAttention, all 8 tensors (4 weights + 4 scales)
    // must agree with `d`.
    auto on = [&](const Tensor &t) {
        return !t.storage() || t.device() == d;
    };
    return on(mha_.q_proj().weight())   && on(mha_.q_proj().scale())   &&
           on(mha_.k_proj().weight())   && on(mha_.k_proj().scale())   &&
           on(mha_.v_proj().weight())   && on(mha_.v_proj().scale())   &&
           on(mha_.out_proj().weight()) && on(mha_.out_proj().scale());
}

void ConformerAttention::clear_position_projection_cache() {
    position_projection_cache_.clear();
    position_projection_cache_clock_ = 0;
}

Module &ConformerAttention::to(Device device) {
    clear_position_projection_cache();
    return Module::to(device);
}

Module &ConformerAttention::to(DType dtype) {
    clear_position_projection_cache();
    return Module::to(dtype);
}

Tensor ConformerAttention::projected_position(const Tensor &pos_emb,
                                              size_t num_heads,
                                              bool head_major) const {
    if (!cache_position_projections_enabled()) {
        Tensor projected = pos_proj_(pos_emb);
        if (!head_major) {
            return projected;
        }
        const size_t pos_len = projected.shape()[0];
        const size_t d_model = projected.shape()[1];
        if (num_heads == 0 || d_model % num_heads != 0) {
            throw ShapeError(
                "projected_position head-major layout requires an output "
                "width divisible by num_heads");
        }
        return projected
            .reshape({1, pos_len, num_heads, d_model / num_heads})
            .transpose({0, 2, 1, 3});
    }
    const auto matches_cache_key = [&](const PositionProjectionCacheEntry &entry) {
        return entry.input_shape == pos_emb.shape() &&
               entry.dtype == pos_emb.dtype() &&
               entry.device == pos_emb.device() &&
               entry.head_major == head_major;
    };
    const auto cache_hit = std::find_if(position_projection_cache_.begin(),
                                        position_projection_cache_.end(),
                                        matches_cache_key);
    if (cache_hit != position_projection_cache_.end()) {
        cache_hit->last_used = ++position_projection_cache_clock_;
        return cache_hit->projected;
    }

    // Request workspace leases must end at the forward boundary. The cached
    // tensor intentionally lives across requests, so allocate just this
    // immutable model-derived result outside the request workspace.
    axiom::ScopedGpuWorkspaceAllocationBypass persistent_allocation;
    Tensor projected = pos_proj_(pos_emb);
    if (head_major) {
        const size_t pos_len = projected.shape()[0];
        const size_t d_model = projected.shape()[1];
        if (num_heads == 0 || d_model % num_heads != 0) {
            throw ShapeError(
                "projected_position head-major layout requires an output "
                "width divisible by num_heads");
        }
        projected = projected
                        .reshape(
                            {1, pos_len, num_heads, d_model / num_heads})
                        .transpose({0, 2, 1, 3})
                        .ascontiguousarray();
        // reshape/transpose is lazy on the Metal graph path. Materialize its
        // one-time contiguous copy while the workspace bypass is active so
        // this cache never acquires request-scoped backing or forces a gather
        // again during every attention matmul.
        static_cast<void>(projected.storage());
    }
    if (!head_major && persistent_allocation.active()) {
        // Linear is lazy on the Metal graph path. A cache entry that survives
        // the request must materialize while its persistent-allocation bypass
        // is active, rather than retaining a workspace lease on a later hit.
        static_cast<void>(projected.storage());
    }
    PositionProjectionCacheEntry entry{pos_emb.shape(), pos_emb.dtype(),
                                       pos_emb.device(), head_major,
                                       std::move(projected),
                                       ++position_projection_cache_clock_};
    const size_t capacity = detail::position_projection_cache_capacity();
    if (position_projection_cache_.size() >= capacity) {
        const auto eviction = std::min_element(
            position_projection_cache_.begin(), position_projection_cache_.end(),
            [](const PositionProjectionCacheEntry &left,
               const PositionProjectionCacheEntry &right) {
                return left.last_used < right.last_used;
            });
        *eviction = std::move(entry);
        return eviction->projected;
    }
    position_projection_cache_.push_back(std::move(entry));
    return position_projection_cache_.back().projected;
}

Tensor ConformerAttention::rel_shift(const Tensor &x) {
    // x: (batch, heads, seq_len, 2*seq_len-1)
    // Returns: (batch, heads, seq_len, seq_len)
    auto shape = x.shape();
    size_t batch = shape[0];
    size_t heads = shape[1];
    size_t seq_len = shape[2];
    size_t pos_len = shape[3]; // 2*seq_len - 1

    // Pad left column with zero: (batch, heads, seq_len, 2*seq_len)
    auto padded = ops::pad(x, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});

    // Reshape to (batch, heads, 2*seq_len, seq_len)
    padded = padded.reshape({batch, heads, pos_len + 1, seq_len});

    // Slice off first row: (batch, heads, 2*seq_len-1, seq_len)
    padded = padded.slice({Slice(), Slice(), Slice(1), Slice()});

    // Reshape back: (batch, heads, seq_len, 2*seq_len-1)
    padded = padded.reshape({batch, heads, seq_len, pos_len});

    // Take first seq_len columns: (batch, heads, seq_len, seq_len)
    return padded.slice(
        {Slice(), Slice(), Slice(), Slice(0, static_cast<int64_t>(seq_len))});
}

Tensor ConformerAttention::rel_position_attention(const Tensor &query,
                                                  const Tensor &key,
                                                  const Tensor &value,
                                                  const Tensor &pos_emb,
                                                  const Tensor &mask) const {
    // query/key/value: (batch, seq, d_model)
    // pos_emb: (2*seq-1, d_model)
    //
    // WAS-28 PR #4a — inner-Attn signposts. PR #2's `Attn` signpost showed
    // attention as 54.8% of encoder wall but said nothing about which of the
    // 10 dispatchable ops below dominates. These nested signposts give the
    // breakdown to pick the right lever (fused QKV, pos_proj cache,
    // rel_shift kernel, fused softmax+score, etc.) for PR #4b.
    //
    // Bracing requirement (see signposts.hpp): each BEGIN/END pair lives in
    // its own `{ }` block — the macros emit per-name local variables that
    // would collide on a second BEGIN with the same name in the same scope.
    // CPU-side caveat: the inner timings are CPU wall around command-buffer
    // encoding; the GPU executes asynchronously between blocks. Pair with
    // Metal System Trace for true GPU-side attribution.

    int num_heads = mha_.num_heads();

    // Project Q, K, V — Linear::forward() dispatches to int8_matmul
    // automatically when weight is Int8 + scale is loaded (WAS-27 fast path).
    // Bias is applied inside Linear::forward(), no explicit add needed here.
    Tensor q, k, v;
    bool qkv_head_layout = false;
    if (SafeDirectQkvPersistentPlan::active() != nullptr &&
        !can_use_direct_int8_qkv(query, key, value, mha_.q_proj(),
                                 mha_.k_proj(), mha_.v_proj())) {
        throw RuntimeError(
            "Safe direct QKV persistent plan lost direct-QKV eligibility");
    }
    if (can_use_direct_int8_qkv(query, key, value, mha_.q_proj(),
                                 mha_.k_proj(), mha_.v_proj())) {
        PARAKEET_SP_BEGIN(QkvProj);
        qkv_head_layout = direct_int8_qkv_head_layout_enabled();
        auto *persistent_plan = SafeDirectQkvPersistentPlan::active();
        std::array<Tensor, 3> qkv;
        if (persistent_plan != nullptr) {
            if (!qkv_head_layout) {
                throw RuntimeError(
                    "Safe direct QKV persistent plan requires head-major outputs");
            }
            qkv = persistent_plan->next_outputs();
            const auto batch = query.shape()[0];
            const auto time = query.shape()[1];
            const auto hidden = query.shape()[2];
            if (!axiom::backends::metal::gpu_int8_qkv_matmul_bias_into(
                    qkv[0], qkv[1], qkv[2],
                    query.reshape({batch * time, hidden}),
                    mha_.q_proj().weight(), mha_.q_proj().scale(),
                    mha_.q_proj().bias(), mha_.k_proj().weight(),
                    mha_.k_proj().scale(), mha_.k_proj().bias(),
                    mha_.v_proj().weight(), mha_.v_proj().scale(),
                    mha_.v_proj().bias(), batch, time,
                    static_cast<size_t>(num_heads))) {
                throw RuntimeError(
                    "Safe direct QKV persistent plan could not bind QKV outputs");
            }
        } else {
            qkv = qkv_head_layout
                      ? axiom::ops::int8_qkv_matmul_bias_head_layout(
                            query, mha_.q_proj().weight(),
                            mha_.q_proj().scale(), mha_.q_proj().bias(),
                            mha_.k_proj().weight(), mha_.k_proj().scale(),
                            mha_.k_proj().bias(), mha_.v_proj().weight(),
                            mha_.v_proj().scale(), mha_.v_proj().bias(),
                            static_cast<size_t>(num_heads))
                      : axiom::ops::int8_qkv_matmul_bias(
                            query, mha_.q_proj().weight(),
                            mha_.q_proj().scale(), mha_.q_proj().bias(),
                            mha_.k_proj().weight(), mha_.k_proj().scale(),
                            mha_.k_proj().bias(), mha_.v_proj().weight(),
                            mha_.v_proj().scale(), mha_.v_proj().bias());
        }
        PARAKEET_SP_END(QkvProj);
        // These observations exist only to qualify an explicitly enabled
        // whole-encoder schedule capture. Evaluating schedule_storage_use()
        // materializes a lazy Q/K/V tensor, so it must not turn the ordinary
        // no-capture path into an eager boundary.
        if (axiom::graph::execution_schedule_capture_enabled()) {
            axiom::graph::record_direct_execution_schedule_segment(
                qkv_head_layout
                    ? axiom::graph::ExecutionScheduleDirectRoute::Int8QkvHeadLayout
                    : axiom::graph::ExecutionScheduleDirectRoute::Int8Qkv);
            axiom::graph::record_execution_schedule_direct_uses(
                qkv_head_layout
                    ? axiom::graph::ExecutionScheduleDirectRoute::Int8QkvHeadLayout
                    : axiom::graph::ExecutionScheduleDirectRoute::Int8Qkv,
                {schedule_storage_use(query)},
                {schedule_storage_use(qkv[0]), schedule_storage_use(qkv[1]),
                 schedule_storage_use(qkv[2])},
                /*outputs_alias_inputs=*/false);
        }
        validate_execution_plan_direct_route(
            qkv_head_layout
                ? axiom::graph::ExecutionScheduleDirectRoute::Int8QkvHeadLayout
                : axiom::graph::ExecutionScheduleDirectRoute::Int8Qkv);
        q = std::move(qkv[0]);
        k = std::move(qkv[1]);
        v = std::move(qkv[2]);
    } else {
        {
            PARAKEET_SP_BEGIN(QProj);
            q = mha_.q_proj()(query);
            PARAKEET_SP_END(QProj);
        }
        {
            PARAKEET_SP_BEGIN(KProj);
            k = mha_.k_proj()(key);
            PARAKEET_SP_END(KProj);
        }
        {
            PARAKEET_SP_BEGIN(VProj);
            v = mha_.v_proj()(value);
            PARAKEET_SP_END(VProj);
        }
    }

    auto d_model = static_cast<int>(query.shape().back());
    int head_dim = d_model / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto batch = query.shape()[0];
    auto seq_len = query.shape()[1];

    // Reshape to multi-head: (batch, seq, heads, head_dim) → (batch, heads,
    // seq, head_dim)
    auto nh = static_cast<size_t>(num_heads);
    auto hd = static_cast<size_t>(head_dim);
    if (!qkv_head_layout) {
        q = q.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
        k = k.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
        v = v.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    }

    // q: (batch, heads, seq, head_dim)
    // pos_bias_u/v: (heads, head_dim) → broadcast as (1, heads, 1, head_dim)
    auto bias_u = pos_bias_u_.reshape({1, nh, 1, hd});
    auto bias_v = pos_bias_v_.reshape({1, nh, 1, hd});

    // Position attention: project position embeddings
    const bool position_head_layout =
        qkv_head_layout && cache_position_head_layout_enabled();
    Tensor p;
    {
        PARAKEET_SP_BEGIN(PosProj);
        p = projected_position(pos_emb, nh, position_head_layout);
        PARAKEET_SP_END(PosProj);
    }
    auto pos_len = position_head_layout ? p.shape()[2] : p.shape()[0];
    // Reshape to (1, 2*seq-1, heads, head_dim) → (1, heads, 2*seq-1,
    // head_dim). The cached head-major projection already has this layout.
    if (!position_head_layout) {
        p = p.reshape({1, pos_len, nh, hd}).transpose({0, 2, 1, 3});
    }

    // A direct QKV head layout plus a cached contiguous head-major position
    // projection gives the native relative-attention kernel exactly the
    // [B,H,T,D] / [1,H,2T-1,D] buffers it requires. Leave every other path on
    // the established MPSGraph decomposition, including first-use cache fills.
    Tensor out;
    const bool use_relative_position_attention =
        relative_position_attention_enabled() && qkv_head_layout &&
        position_head_layout;
    if (use_relative_position_attention) {
        PARAKEET_SP_BEGIN(RelativePositionAttention);
        out = ops::relative_position_attention(q, k, v, p, bias_u, bias_v,
                                                mask, scale);
        PARAKEET_SP_END(RelativePositionAttention);
    } else {
        // Content attention: (Q + pos_bias_u) @ K^T → (batch, heads, seq, seq)
        Tensor content_score;
        {
            PARAKEET_SP_BEGIN(ContentScore);
            content_score = ops::matmul(q + bias_u, k, false, true);
            PARAKEET_SP_END(ContentScore);
        }

        // (Q + pos_bias_v) @ P^T → (batch, heads, seq, 2*seq-1)
        Tensor pos_score;
        {
            PARAKEET_SP_BEGIN(PosScore);
            pos_score = ops::matmul(q + bias_v, p, false, true);
            PARAKEET_SP_END(PosScore);
        }

        // Shift to align relative positions
        {
            PARAKEET_SP_BEGIN(RelShift);
            pos_score = rel_shift(pos_score);
            PARAKEET_SP_END(RelShift);
        }

        // Combined scores + optional mask + softmax. Keep the outer scope for
        // historical comparisons, but split its stages below. `storage()` can
        // materialize a lazy Metal graph, so treating this whole section as
        // "softmax" would attribute a potential compile/submit boundary to the
        // reduction kernel.
        Tensor attn_weights;
        {
            PARAKEET_SP_BEGIN(Softmax);
            Tensor scores;
            {
                PARAKEET_SP_BEGIN(ScoreMerge);
                scores = (content_score + pos_score) * scale;
                PARAKEET_SP_END(ScoreMerge);
            }
            bool has_mask;
            {
                PARAKEET_SP_BEGIN(MaskStorage);
                has_mask = static_cast<bool>(mask.storage());
                PARAKEET_SP_END(MaskStorage);
            }
            if (has_mask) {
                PARAKEET_SP_BEGIN(MaskedFill);
                scores = ops::masked_fill(scores, mask, -1e9f);
                PARAKEET_SP_END(MaskedFill);
            }
            {
                PARAKEET_SP_BEGIN(SoftmaxReduction);
                attn_weights = ops::softmax(scores, -1);
                PARAKEET_SP_END(SoftmaxReduction);
            }
            PARAKEET_SP_END(Softmax);
        }

        // Weighted sum: (batch, heads, seq, head_dim)
        {
            PARAKEET_SP_BEGIN(AttnMatmul);
            out = ops::matmul(attn_weights, v);
            PARAKEET_SP_END(AttnMatmul);
        }
    }

    // Reshape back: (batch, seq, d_model)
    out = out.transpose({0, 2, 1, 3});
    out = out.reshape({batch, seq_len, static_cast<size_t>(d_model)});

    {
        PARAKEET_SP_BEGIN(OutProj);
        out = mha_.out_proj()(out);
        PARAKEET_SP_END(OutProj);
    }
    return out;
}

Tensor ConformerAttention::forward(const Tensor &input, const Tensor &pos_emb,
                                   const Tensor &mask) const {
    auto x = norm_(input);
    x = rel_position_attention(x, x, x, pos_emb, mask);
    x = dropout_(x);
    if (can_use_direct_int8_residual(input, x)) {
        ops::int8_add_residual_inplace(x, input);
        axiom::graph::record_direct_execution_schedule_segment(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Residual);
        axiom::graph::record_execution_schedule_direct_uses(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Residual,
            {schedule_storage_use(x), schedule_storage_use(input)},
            {schedule_storage_use(x)}, /*outputs_alias_inputs=*/true);
        validate_execution_plan_direct_route(
            axiom::graph::ExecutionScheduleDirectRoute::Int8Residual);
        return x;
    }
    return input + x;
}

// ─── ConformerBlock ─────────────────────────────────────────────────────────

ConformerBlock::ConformerBlock(const EncoderConfig &config)
    : ffn1_(config.dropout), attn_(config.num_heads, config.dropout),
      conv_(config.hidden_size, config.dropout), ffn2_(config.dropout) {
    AX_REGISTER_MODULES(ffn1_, attn_, conv_, ffn2_, final_norm_);
}

void ConformerBlock::load_int8_weights(
    Tensor q_int8, Tensor q_scale,
    Tensor k_int8, Tensor k_scale,
    Tensor v_int8, Tensor v_scale,
    Tensor o_int8, Tensor o_scale,
    Tensor ffn1_fc1_int8, Tensor ffn1_fc1_scale,
    Tensor ffn1_fc2_int8, Tensor ffn1_fc2_scale,
    Tensor ffn2_fc1_int8, Tensor ffn2_fc1_scale,
    Tensor ffn2_fc2_int8, Tensor ffn2_fc2_scale) {

    attn_.load_int8_weights(
        std::move(q_int8),  std::move(q_scale),
        std::move(k_int8),  std::move(k_scale),
        std::move(v_int8),  std::move(v_scale),
        std::move(o_int8),  std::move(o_scale));

    ffn1_.load_int8_weights(
        std::move(ffn1_fc1_int8), std::move(ffn1_fc1_scale),
        std::move(ffn1_fc2_int8), std::move(ffn1_fc2_scale));

    ffn2_.load_int8_weights(
        std::move(ffn2_fc1_int8), std::move(ffn2_fc1_scale),
        std::move(ffn2_fc2_int8), std::move(ffn2_fc2_scale));
}

void ConformerBlock::load_int8_pointwise_conv_weights(
    Tensor pointwise1_int8, Tensor pointwise1_scale,
    Tensor pointwise2_int8, Tensor pointwise2_scale) {
    conv_.load_int8_pointwise_weights(std::move(pointwise1_int8),
                                      std::move(pointwise1_scale),
                                      std::move(pointwise2_int8),
                                      std::move(pointwise2_scale));
}

void ConformerBlock::load_int8_position_projection_weights(Tensor weights,
                                                            Tensor scale) {
    attn_.load_int8_position_projection_weights(std::move(weights),
                                                 std::move(scale));
}

void ConformerBlock::clear_position_projection_cache() {
    attn_.clear_position_projection_cache();
}

Tensor ConformerBlock::forward(const Tensor &input, const Tensor &pos_emb,
                               const Tensor &mask) const {
    const char *block_program =
        std::getenv("PARAKEET_FASTCONFORMER_BLOCK_PROGRAM");
    if (block_program != nullptr && block_program[0] == '1' &&
        block_program[1] == '\0') {
        // A preceding block may end in an MPSGraph value. The programmed
        // FFN segments bind concrete Metal buffers, so establish this one
        // graph boundary before checking their supplied-output contract.
        // This materialization preserves the existing ordered stream; it
        // does not synchronize the CPU.
        Tensor program_input = input;
        if (program_input.is_lazy()) {
            static_cast<void>(program_input.storage());
        }
        const bool supported = FastConformerBlockProgram::is_supported(
            *this, program_input, pos_emb, mask);
        if (supported) {
            return FastConformerBlockProgram::encode(*this, program_input,
                                                      pos_emb, mask);
        }
    }

    // Signposts: Instruments aggregates by literal name → across 18 blocks,
    // "FFN1" sums to total FFN1 wall-time, etc. Each PARAKEET_SP_BEGIN
    // defines its own scoped variables (token-pasted from the identifier)
    // so we use { } blocks to keep scopes clean.
    Tensor x;
    {
        PARAKEET_SP_BEGIN(FFN1);
        x = ffn1_(input);
        PARAKEET_SP_END(FFN1);
    }
    {
        PARAKEET_SP_BEGIN(Attn);
        x = attn_(x, pos_emb, mask);
        PARAKEET_SP_END(Attn);
    }
    {
        PARAKEET_SP_BEGIN(Conv);
        x = conv_(x);
        PARAKEET_SP_END(Conv);
    }
    {
        PARAKEET_SP_BEGIN(FFN2);
        x = ffn2_(x);
        PARAKEET_SP_END(FFN2);
    }
    {
        PARAKEET_SP_BEGIN(BlockFinalNorm);
        x = final_norm_(x);
        PARAKEET_SP_END(BlockFinalNorm);
    }
    return x;
}

// ─── ConvSubsampling (Conv2d) ────────────────────────────────────────────────

ConvSubsampling::ConvSubsampling(int channels)
    : conv1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}),
      dw1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      dw2_(/*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      conv2_(/*stride=*/{1, 1}, /*padding=*/{0, 0}),
      conv3_(/*stride=*/{1, 1}, /*padding=*/{0, 0}), proj_(true) {
    AX_REGISTER_MODULES(conv1_, dw1_, conv2_, dw2_, conv3_, proj_);
}

Tensor ConvSubsampling::forward(const Tensor &input) const {
    // input: (batch, mel_length, mel_bins)
    auto x = input.unsqueeze(1); // (batch, 1, mel_length, mel_bins)

    x = conv1_(x);
    x = ops::relu(x);

    x = dw1_(x);
    x = conv2_(x);
    x = ops::relu(x);

    x = dw2_(x);
    x = conv3_(x);
    x = ops::relu(x);

    // Flatten channels and freq: (batch, C, T/8, F/8) → (batch, T/8, C*F/8)
    auto shape = x.shape();
    x = x.permute({0, 2, 1, 3}); // (batch, T/8, C, F/8)
    x = x.ascontiguousarray();
    x = x.reshape({shape[0], shape[2], shape[1] * shape[3]});

    return proj_(x); // (batch, T/8, d_model)
}

// ─── FastConformerEncoder ───────────────────────────────────────────────────

FastConformerEncoder::FastConformerEncoder(const EncoderConfig &config)
    : config_(config), subsampling_(config.subsampling_channels) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back<ConformerBlock>(config);
    }
    AX_REGISTER_MODULES(subsampling_, layers_);
}

void FastConformerEncoder::load_state_dict(
    const std::map<std::string, Tensor> &state_dict,
    const std::string &prefix, bool strict) {

    // Reloading weights can change dtype/device of subsequent forwards;
    // cache entries from the previous configuration would never hit again.
    pos_emb_cache_.clear();
    for (auto &block : layers_.each<ConformerBlock>()) {
        block.clear_position_projection_cache();
    }

    // Always load fp16 weights first via the base Module logic.
    Module::load_state_dict(state_dict, prefix, strict);

    // Detect whether any _quantized key in the map belongs to this encoder.
    is_int8_ = false;
    static const std::string kQuantizedSuffix = "_quantized";
    for (const auto &entry : state_dict) {
        const std::string &name = entry.first;
        if (name.size() >= kQuantizedSuffix.size() &&
            name.ends_with(kQuantizedSuffix) &&
            name.rfind(prefix, 0) == 0) {
            is_int8_ = true;
            break;
        }
    }

    if (is_int8_) {
        load_int8_weights_(state_dict, prefix);
    }
}

void FastConformerEncoder::load_int8_weights_(
    const std::map<std::string, Tensor> &state_dict,
    const std::string &prefix) {

    // Helper: look up a required key, throw descriptive error if missing.
    auto get = [&](const std::string &key) -> Tensor {
        auto it = state_dict.find(key);
        if (it == state_dict.end()) {
            throw RuntimeError::internal(
                "int8 weight key '" + key + "' not found in state_dict");
        }
        return it->second;
    };
    auto has = [&](const std::string &key) {
        return state_dict.find(key) != state_dict.end();
    };

    int num_layers = config_.num_layers;

    for (int i = 0; i < num_layers; ++i) {
        // Full key prefix for this layer, e.g. "encoder_.layers_.0."
        std::string lp = prefix + "layers_." + std::to_string(i) + ".";

        // Attention projection keys.
        // Axiom fp16 key: lp + "attn_.mha_.q_proj.weight"
        // Quantizer output (strips ".weight", appends _quantized/_scale):
        //   lp + "attn_.mha_.q_proj_quantized"
        //   lp + "attn_.mha_.q_proj_scale"
        const std::string ap = lp + "attn_.mha_.";

        // FeedForward keys.
        // Axiom fp16 key: lp + "ffn1_.fc1_.weight"
        // Quantizer output: lp + "ffn1_.fc1__quantized"  (double underscore
        //   because the registered submodule name is "fc1_" and we strip the
        //   ".weight" suffix from "fc1_.weight")
        const std::string f1p = lp + "ffn1_.";
        const std::string f2p = lp + "ffn2_.";
        const std::string cp = lp + "conv_.";

        auto &block = static_cast<ConformerBlock &>(layers_[static_cast<size_t>(i)]);
        block.load_int8_weights(
            // attention q/k/v/out_proj
            get(ap + "q_proj_quantized"),   get(ap + "q_proj_scale"),
            get(ap + "k_proj_quantized"),   get(ap + "k_proj_scale"),
            get(ap + "v_proj_quantized"),   get(ap + "v_proj_scale"),
            get(ap + "out_proj_quantized"), get(ap + "out_proj_scale"),
            // ffn1 fc1 / fc2
            get(f1p + "fc1__quantized"), get(f1p + "fc1__scale"),
            get(f1p + "fc2__quantized"), get(f1p + "fc2__scale"),
            // ffn2 fc1 / fc2
            get(f2p + "fc1__quantized"), get(f2p + "fc1__scale"),
            get(f2p + "fc2__quantized"), get(f2p + "fc2__scale")
        );

        const std::string pointwise1_weight = cp + "pointwise_conv1__quantized";
        const std::string pointwise1_scale = cp + "pointwise_conv1__scale";
        const std::string pointwise2_weight = cp + "pointwise_conv2__quantized";
        const std::string pointwise2_scale = cp + "pointwise_conv2__scale";
        const bool has_any_pointwise = has(pointwise1_weight) || has(pointwise1_scale) ||
                                       has(pointwise2_weight) || has(pointwise2_scale);
        const bool has_all_pointwise = has(pointwise1_weight) && has(pointwise1_scale) &&
                                       has(pointwise2_weight) && has(pointwise2_scale);
        if (has_any_pointwise && !has_all_pointwise) {
            throw RuntimeError("Incomplete INT8 pointwise-convolution weights for " + lp);
        }
        if (has_all_pointwise) {
            block.load_int8_pointwise_conv_weights(
                get(pointwise1_weight), get(pointwise1_scale),
                get(pointwise2_weight), get(pointwise2_scale));
        }

        const std::string position_projection_weight =
            lp + "attn_.pos_proj__quantized";
        const std::string position_projection_scale = lp + "attn_.pos_proj__scale";
        const bool has_any_position_projection = has(position_projection_weight) ||
                                                 has(position_projection_scale);
        const bool has_all_position_projection = has(position_projection_weight) &&
                                                 has(position_projection_scale);
        if (has_any_position_projection && !has_all_position_projection) {
            throw RuntimeError("Incomplete INT8 position-projection weights for " + lp);
        }
        if (has_all_position_projection && int8_position_projections_enabled()) {
            block.load_int8_position_projection_weights(
                get(position_projection_weight), get(position_projection_scale));
        }
    }
}

// WAS-28 PR #3 — sinusoidal_position_embedding consumed ~22.5% of encoder
// wall on every forward but is pure-function on (seq_len, d_model, dtype,
// device). Cache hits return in <<1 ms and let multi-chunk transcribes pay
// the compute once. Single-threaded by contract: a parakeet engine
// serialises transcribe calls per instance, so no mutex.
Tensor FastConformerEncoder::pos_emb(int seq_len, int d_model, DType dtype,
                                     Device device) const {
    PosEmbKey key{seq_len, d_model, dtype, device};
    auto it = pos_emb_cache_.find(key);
    if (it == pos_emb_cache_.end()) {
        Tensor pe = axiom::nn::sinusoidal_position_embedding(
            seq_len, d_model, dtype, device);
        it = pos_emb_cache_.emplace(key, std::move(pe)).first;
    }
    return it->second;
}

// Migrating to a different device makes every cached tensor unusable
// (wrong device) — clear before delegating so the base recursion can
// migrate the rest of the encoder.
Module &FastConformerEncoder::to(Device device) {
    pos_emb_cache_.clear();
    return Module::to(device);
}

// Same story for dtype casts.
Module &FastConformerEncoder::to(DType dtype) {
    pos_emb_cache_.clear();
    return Module::to(dtype);
}

Tensor FastConformerEncoder::forward(const Tensor &input,
                                     const Tensor &mask) const {
    // Top-level encoder phase signposts. Inner ConformerBlock::forward
    // emits FFN1/Attn/Conv/FFN2/BlockFinalNorm signposts; those nest
    // inside ConformerBlocks so Instruments shows the hierarchy.
    //
    // Attribution caveat: only the outer Encoder interval is GPU-
    // inclusive (the call returns after the final GPU sync that
    // materializes the output tensor). Inner intervals measure CPU-
    // side wall around command-buffer encoding ops; the actual GPU
    // execution happens asynchronously between blocks. Pair with the
    // Metal System Trace instrument for true GPU-side attribution.
    PARAKEET_SP_BEGIN(Encoder);
    fused_f16_pointwise_glu_calls_this_forward = 0;
    direct_f16_pointwise_calls_this_forward = 0;
    direct_f16_glu_calls_this_forward = 0;
    direct_f16_glu_comparison_calls_this_forward = 0;
    direct_f16_glu_max_abs_error_this_forward = 0.0f;
    direct_f16_glu_default_exp_max_abs_error_this_forward = 0.0f;
    direct_f16_glu_rounded_sigmoid_max_abs_error_this_forward = 0.0f;

    Tensor x;
    {
        PARAKEET_SP_BEGIN(Subsampling);
        x = subsampling_(input);
        PARAKEET_SP_END(Subsampling);
    }

    int seq_len = static_cast<int>(x.shape()[1]);
    int d_model = static_cast<int>(x.shape()[2]);
    Tensor pos_emb_tensor;
    {
        PARAKEET_SP_BEGIN(PosEmb);
        pos_emb_tensor = pos_emb(seq_len, d_model, x.dtype(), x.device());
        PARAKEET_SP_END(PosEmb);
    }

    {
        PARAKEET_SP_BEGIN(ConformerBlocks);
        std::unique_ptr<SafeDirectQkvPersistentPlan> persistent_qkv_plan;
        const size_t layer_count = static_cast<size_t>(config_.num_layers);
        const size_t num_heads = static_cast<size_t>(config_.num_heads);
        if (can_begin_safe_direct_qkv_persistent_plan(x, layer_count,
                                                      num_heads)) {
            persistent_qkv_plan = std::make_unique<SafeDirectQkvPersistentPlan>(
                x, layer_count, num_heads);
        }
        for (const auto &block : layers_.each<ConformerBlock>()) {
            x = block(x, pos_emb_tensor, mask);
        }
        if (persistent_qkv_plan) {
            persistent_qkv_plan->handoff_complete_plan(x);
        }
        PARAKEET_SP_END(ConformerBlocks);
    }

    PARAKEET_SP_END(Encoder);
    last_f16_pointwise_route_stats_ = {
        .projection_calls = direct_f16_pointwise_calls_this_forward,
        .glu_calls = fused_f16_pointwise_glu_calls_this_forward,
        .direct_glu_calls = direct_f16_glu_calls_this_forward,
        .direct_glu_comparison_calls =
            direct_f16_glu_comparison_calls_this_forward,
        .direct_glu_max_abs_error = direct_f16_glu_max_abs_error_this_forward,
        .direct_glu_default_exp_max_abs_error =
            direct_f16_glu_default_exp_max_abs_error_this_forward,
        .direct_glu_rounded_sigmoid_max_abs_error =
            direct_f16_glu_rounded_sigmoid_max_abs_error_this_forward,
    };
    return x;
}

} // namespace parakeet::models
