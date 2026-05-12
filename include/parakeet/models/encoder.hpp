#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

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
    // detected in the state dict. Delegates to fc1_.load_int8_weights() and
    // fc2_.load_int8_weights() so Linear::forward() dispatches automatically.
    void load_int8_weights(Tensor fc1_w_int8, Tensor fc1_w_scale,
                           Tensor fc2_w_int8, Tensor fc2_w_scale);

    // Derived from primary state: true iff fc1_'s weight is Int8 and its
    // scale_ parameter is loaded. No separate bool field — eliminates the
    // stale-state hazard where reloading fp16 weights after int8 would leave
    // a cached bool returning true.
    bool is_int8() const {
        return fc1_.has_scale() && fc1_.weight().dtype() == DType::Int8;
    }

    // Test/diagnostic helper: device of fc1_'s scale tensor (or CPU if not
    // loaded). Used to verify Module::to(Device) migrated the int8 fields.
    // (See Int8DeviceCoercion gtest.)
    //
    // NOTE: prefer `all_int8_on(Device)` for regression coverage — this single-
    // field accessor only observes fc1_'s scale, so a regression that breaks
    // the migration of fc2_'s weight/scale would silently pass the test.
    Device int8_weights_device() const;

    // Returns true iff EVERY int8 weight + scale tensor for fc1_ and fc2_ is
    // currently resident on `d`. Tensors with no storage do not vote — so on
    // a non-int8 FeedForward this returns true regardless of `d`. Used by the
    // Int8DeviceCoercion test to guard ALL fields, not just the canary.
    bool all_int8_on(Device d) const;

  private:
    LayerNorm norm_;
    Linear fc1_;
    Linear fc2_;
    Dropout dropout_;
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
    // detected. Concatenates q/k/v along the output dim and loads the fused
    // qkv_proj_; loads mha_.out_proj_ separately. Linear::forward() then
    // dispatches automatically through the int8 fast path.
    void load_int8_weights(Tensor q_int8, Tensor q_scale, Tensor k_int8,
                           Tensor k_scale, Tensor v_int8, Tensor v_scale,
                           Tensor o_int8, Tensor o_scale);

    // Derived from primary state: true iff qkv_proj_'s weight is Int8 and
    // its scale_ parameter is loaded. No separate bool field — same rationale
    // as FeedForward::is_int8().
    bool is_int8() const {
        return qkv_proj_.has_scale() &&
               qkv_proj_.weight().dtype() == DType::Int8;
    }

    // Test/diagnostic helper — see FeedForward::int8_weights_device().
    // Returns device of q_proj's scale tensor (or CPU if not loaded).
    // Same single-field-only caveat applies; prefer `all_int8_on(Device)`.
    Device int8_weights_device() const;

    // Returns true iff EVERY q/k/v/o int8 weight + scale tensor is currently
    // resident on `d`. See FeedForward::all_int8_on() for full rationale.
    bool all_int8_on(Device d) const;

    // WAS-28 PR #4b: q/k/v projections are fused into qkv_proj_ (out_features
    // = 3*hidden). mha_.q_proj_/k_proj_/v_proj_ remain as registered Linear
    // submodules but their weight_ tensors are never populated — the override
    // of load_state_dict below remaps q/k/v.weight keys into qkv_proj_.weight
    // via row-wise concat before delegating to base, so 1× memory.
    // mha_.out_proj_ continues to be loaded normally (out is not fused).
    const Linear &qkv_proj() const { return qkv_proj_; }

    // Overrides Module::load_state_dict to redirect q/k/v.weight keys into a
    // single concatenated qkv_proj_.weight. Without the override, base load
    // would populate mha_.q/k/v_proj_'s weight tensors AND leave qkv_proj_
    // empty, swapping the dispatch perf win for memory waste.
    void load_state_dict(const std::map<std::string, Tensor> &state_dict,
                         const std::string &prefix = "",
                         bool strict = true) override;

  private:
    LayerNorm norm_;
    MultiHeadAttention mha_;
    // Fused Q/K/V projection — weight shape (3*hidden, hidden), int8 + scale
    // when quantized. Replaces three separate matmul dispatches with one.
    Linear qkv_proj_;
    Linear pos_proj_;
    Dropout dropout_;
    Tensor pos_bias_u_; // (num_heads, head_dim) — learned position bias
    Tensor pos_bias_v_; // (num_heads, head_dim) — learned position bias

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

    // Overrides invalidate pos_emb_cache_: cached tensors carry the
    // pre-migration dtype/device and would silently leak otherwise (never
    // hit, never evicted). Matches the defensive pattern at FeedForward's
    // is_int8() — same stale-state hazard class.
    Module &to(Device device) override;
    Module &to(DType dtype) override;

    // Returns true if int8 quantized weights were detected during the most
    // recent load_state_dict call. False if the encoder is running fp16.
    bool is_int8() const { return is_int8_; }

    // Returns the sinusoidal positional embedding for these parameters,
    // computing and caching it on the first call and reusing the cached
    // tensor on every subsequent call with the same key. Called once per
    // forward() pass; exposed publicly so callers (and tests) can warm the
    // cache for known bucket shapes ahead of time.
    Tensor pos_emb(int seq_len, int d_model, DType dtype,
                   Device device) const;

    // Number of distinct (seq_len, d_model, dtype, device) tuples currently
    // memoised. Exposed for diagnostics + WAS-28 regression tests;
    // consumers do not depend on entry-level details.
    size_t pos_emb_cache_size() const { return pos_emb_cache_.size(); }

  private:
    EncoderConfig config_;
    ConvSubsampling subsampling_;
    ModuleList layers_;

    bool is_int8_ = false;

    // Cache key for memoised sinusoidal_position_embedding results.
    // (d_model, dtype, device) are effectively constant once the encoder is
    // loaded, but we key on all four for correctness — test code constructs
    // encoders with different configs in the same process.
    struct PosEmbKey {
        int seq_len;
        int d_model;
        DType dtype;
        Device device;
        bool operator==(const PosEmbKey &) const = default;
    };
    struct PosEmbKeyHash {
        size_t operator()(const PosEmbKey &k) const noexcept {
            // boost::hash_combine pattern.
            auto mix = [](size_t h, size_t v) {
                return h ^ (v + 0x9e3779b9 + (h << 6) + (h >> 2));
            };
            size_t h = std::hash<int>{}(k.seq_len);
            h = mix(h, std::hash<int>{}(k.d_model));
            h = mix(h, std::hash<int>{}(static_cast<int>(k.dtype)));
            h = mix(h, std::hash<int>{}(static_cast<int>(k.device)));
            return h;
        }
    };
    // forward() is const so the cache must be mutable. Single-threaded by
    // contract: parakeet engines serialise transcribe calls per instance.
    //
    // No eviction policy: cache size is bounded by the number of distinct
    // (seq_len, d_model, dtype, device) tuples a caller feeds. Wasper's
    // /transcribe path bucketing (WAS-13) keeps this tiny (~3-5 entries
    // post-warmup); other consumers without bucketing could grow this
    // arbitrarily. At d_model=1024, fp16, seq_len=3000 one entry is
    // ~12 MB. If a non-bucketed consumer materialises, switch to LRU.
    mutable std::unordered_map<PosEmbKey, Tensor, PosEmbKeyHash>
        pos_emb_cache_;

    // Scans state_dict for _quantized keys and injects int8 weight pairs
    // into each ConformerBlock's sub-modules.
    void load_int8_weights_(const std::map<std::string, Tensor> &state_dict,
                            const std::string &prefix);
};

} // namespace parakeet::models
