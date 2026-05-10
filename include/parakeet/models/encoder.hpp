#pragma once

#include <map>
#include <string>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>
#include <axiom/ops/int8_matmul.hpp>

#include "parakeet/models/config.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;

// ─── Feed-Forward Module (Macaron-style half-step) ──────────────────────────

class FeedForward : public Module {
  public:
    explicit FeedForward(float dropout = 0.1f, bool bias = true);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

    // Called by FastConformerEncoder::load_state_dict when int8 weights are
    // detected in the state dict. Stores the pre-quantized weight + scale
    // tensors for fc1 and fc2. After this call, forward() routes through
    // ops::int8_matmul instead of the underlying Linear::forward().
    void load_int8_weights(Tensor fc1_w_int8, Tensor fc1_w_scale,
                           Tensor fc2_w_int8, Tensor fc2_w_scale);

    // Clears the int8 routing flag and drops the bare weight/scale tensors.
    // Called by FastConformerEncoder::load_state_dict at the top of every
    // load (mirroring its own `is_int8_ = false` reset) so a second load
    // with an fp16-only state_dict doesn't leave forward() routing through
    // ops::int8_matmul against now-empty weight tensors.
    void reset_int8();

    bool is_int8() const { return is_int8_; }

    // Override Module::to(Device) so the bare int8 weight + fp16 scale fields
    // (which are NOT registered as Module parameters) are migrated alongside
    // the registered submodules. Without this override, ops::int8_matmul
    // receives a GPU activation paired with CPU-resident weights and aborts
    // at the Metal command-encoder layer.
    Module &to(Device device) override;
    // Defensive no-op for dtype migration: int8 weights and their fp16 scales
    // have fixed dtypes by the quantization scheme — astype()ing them would
    // corrupt the bit pattern. The base-class implementation is invoked for
    // registered params/submodules; the int8 fields are intentionally skipped.
    Module &to(DType dtype) override;

    // Test/diagnostic helper: device of the bare fc1 int8 weight tensor (or
    // CPU if not loaded). Used to verify Module::to(Device) actually migrated
    // the int8 fields. (See Int8DeviceCoercion gtest.)
    //
    // NOTE: prefer `all_int8_on(Device)` for regression coverage — this single-
    // field accessor was the original gate but it only observes `fc1_w_int8_`,
    // so a regression that breaks the migration of `fc2_w_int8_` or either of
    // the per-block fp16 scales would silently pass the device-coercion test.
    Device int8_weights_device() const;

    // Returns true iff EVERY bare int8 weight + fp16 scale tensor (fc1_w_int8_,
    // fc1_w_scale_, fc2_w_int8_, fc2_w_scale_) is currently resident on `d`.
    // Tensors with no storage (i.e. load_int8_weights was never called) do
    // not vote — so on a non-int8 FeedForward this returns true regardless of
    // `d`. Used by the Int8DeviceCoercion test to guard ALL fields, not just
    // the canary one returned by `int8_weights_device()`.
    bool all_int8_on(Device d) const;

  private:
    LayerNorm norm_;
    Linear fc1_;
    Linear fc2_;
    Dropout dropout_;

    bool is_int8_ = false;
    Tensor fc1_w_int8_;
    Tensor fc1_w_scale_;
    Tensor fc2_w_int8_;
    Tensor fc2_w_scale_;
};

// ─── Conformer Convolution Module ───────────────────────────────────────────

class ConformerConvModule : public Module {
  public:
    explicit ConformerConvModule(int groups = 1, float dropout = 0.1f);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    LayerNorm norm_;
    Conv1d pointwise_conv1_; // hidden_size → 2*hidden_size (for GLU)
    Conv1d depthwise_conv_;  // groups=hidden_size, kernel_size=9
    BatchNorm1d batch_norm_;
    Conv1d pointwise_conv2_; // hidden_size → hidden_size
    Dropout dropout_;
};

// ─── Multi-Head Self-Attention with Relative Positional Encoding ────────────

class ConformerAttention : public Module {
  public:
    explicit ConformerAttention(int num_heads = 8, float dropout = 0.1f);

    Tensor forward(const Tensor &input, const Tensor &pos_emb,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input, const Tensor &pos_emb,
                      const Tensor &mask = Tensor()) const {
        return forward(input, pos_emb, mask);
    }

    // Called by FastConformerEncoder::load_state_dict when int8 weights are
    // detected. Stores int8 weight + fp16 scale tensors for q, k, v, and
    // out_proj. After this call, rel_position_attention() routes q/k/v/out
    // matmuls through ops::int8_matmul instead of Linear::forward().
    void load_int8_weights(Tensor q_int8, Tensor q_scale, Tensor k_int8,
                           Tensor k_scale, Tensor v_int8, Tensor v_scale,
                           Tensor o_int8, Tensor o_scale);

    // See FeedForward::reset_int8() — same rationale (reset on re-load so
    // forward() doesn't route through int8_matmul with stale empty tensors).
    void reset_int8();

    bool is_int8() const { return is_int8_; }

    // See FeedForward::to() — same rationale: bare int8 + scale fields are
    // not registered as Module parameters, so the base Module::to(Device)
    // would leave them on CPU. Defensive no-op for dtype.
    Module &to(Device device) override;
    Module &to(DType dtype) override;

    // Test/diagnostic helper — see FeedForward::int8_weights_device().
    // Same single-field-only caveat applies; prefer `all_int8_on(Device)`.
    Device int8_weights_device() const;

    // Returns true iff EVERY bare q/k/v/o int8 weight + fp16 scale tensor is
    // currently resident on `d`. See FeedForward::all_int8_on() for full
    // rationale.
    bool all_int8_on(Device d) const;

  private:
    LayerNorm norm_;
    MultiHeadAttention mha_;
    Linear pos_proj_;
    Dropout dropout_;
    Tensor pos_bias_u_; // (num_heads, head_dim) — learned position bias
    Tensor pos_bias_v_; // (num_heads, head_dim) — learned position bias

    bool is_int8_ = false;
    Tensor q_w_int8_,  q_w_scale_;
    Tensor k_w_int8_,  k_w_scale_;
    Tensor v_w_int8_,  v_w_scale_;
    Tensor o_w_int8_,  o_w_scale_;

    // Relative position attention (bypasses mha_.forward)
    Tensor rel_position_attention(const Tensor &query, const Tensor &key,
                                  const Tensor &value, const Tensor &pos_emb,
                                  const Tensor &mask) const;
    static Tensor rel_shift(const Tensor &x);
};

// ─── Conformer Block ────────────────────────────────────────────────────────

class ConformerBlock : public Module {
  public:
    explicit ConformerBlock(const EncoderConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &pos_emb,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input, const Tensor &pos_emb,
                      const Tensor &mask = Tensor()) const {
        return forward(input, pos_emb, mask);
    }

    // Called by FastConformerEncoder::load_state_dict to inject all 8 int8
    // weight pairs (4 attention + 2 ffn1 + 2 ffn2) into the block's
    // sub-modules. Delegates to ConformerAttention::load_int8_weights and
    // FeedForward::load_int8_weights.
    void load_int8_weights(
        // Attention: q, k, v, out_proj
        Tensor q_int8, Tensor q_scale,
        Tensor k_int8, Tensor k_scale,
        Tensor v_int8, Tensor v_scale,
        Tensor o_int8, Tensor o_scale,
        // FFN1: fc1, fc2
        Tensor ffn1_fc1_int8, Tensor ffn1_fc1_scale,
        Tensor ffn1_fc2_int8, Tensor ffn1_fc2_scale,
        // FFN2: fc1, fc2
        Tensor ffn2_fc1_int8, Tensor ffn2_fc1_scale,
        Tensor ffn2_fc2_int8, Tensor ffn2_fc2_scale);

    // Forwards `reset_int8()` to all int8-bearing children (attn_, ffn1_,
    // ffn2_). Called by FastConformerEncoder::load_state_dict at the top of
    // every load so a re-load with fp16-only weights doesn't leave child
    // modules routing through stale int8 weights.
    void reset_int8();

  private:
    FeedForward ffn1_;
    ConformerAttention attn_;
    ConformerConvModule conv_;
    FeedForward ffn2_;
    LayerNorm final_norm_;
};

// ─── Convolutional Subsampling (Conv2d, matches NeMo FastConformer 110M) ────
//
// NeMo structure (nn.Sequential indices):
//   [0] Conv2d(1, C, 3, stride=2, pad=1)              — regular strided
//   [1] SiLU
//   [2] Conv2d(C, C, 3, groups=C, stride=2, pad=1)    — depthwise strided
//   [3] Conv2d(C, C, 1)                                — pointwise
//   [4] SiLU
//   [5] Conv2d(C, C, 3, groups=C, stride=2, pad=1)    — depthwise strided
//   [6] Conv2d(C, C, 1)                                — pointwise
//   Linear(C * mel_bins/8, d_model)                    — projection
//
// Total downsample: stride-2 × 3 = 8×

class ConvSubsampling : public Module {
  public:
    explicit ConvSubsampling(int channels = 256);

    // (batch, mel_length, mel_bins) → (batch, mel_length/8, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Conv2d conv1_;         // regular strided 3×3 (stride=2)
    Conv2d dw1_, dw2_;     // depthwise strided 3×3 (stride=2, groups=C)
    Conv2d conv2_, conv3_; // pointwise 1×1 (stride=1)
    Linear proj_;
};

// ─── FastConformer Encoder ──────────────────────────────────────────────────

class FastConformerEncoder : public Module {
  public:
    explicit FastConformerEncoder(const EncoderConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

    // Overrides Module::load_state_dict to detect int8 quantized weights
    // (keys ending in "_quantized") and inject them into the conformer
    // layer sub-modules via their load_int8_weights() setters.
    // The base fp16 load is always performed first (strict=false).
    void load_state_dict(const std::map<std::string, Tensor> &state_dict,
                         const std::string &prefix = "",
                         bool strict = true) override;

    // Returns true if int8 quantized weights were detected during the most
    // recent load_state_dict call. False if the encoder is running fp16.
    bool is_int8() const { return is_int8_; }

  private:
    EncoderConfig config_;
    ConvSubsampling subsampling_;
    ModuleList layers_;

    bool is_int8_ = false;

    // Scans state_dict for _quantized keys and injects int8 weight pairs
    // into each ConformerBlock's sub-modules.
    void load_int8_weights_(const std::map<std::string, Tensor> &state_dict,
                            const std::string &prefix);
};

} // namespace parakeet::models
