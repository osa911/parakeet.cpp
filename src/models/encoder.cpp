#include "parakeet/models/encoder.hpp"

#include <cmath>
#include <string>

#include <axiom/error.hpp>
#include <axiom/nn/positional.hpp>

#include "parakeet/profile/signposts.hpp"

namespace parakeet::models {

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
    x = fc1_(x);
    x = ops::silu(x);
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

Tensor ConformerConvModule::forward(const Tensor &input) const {
    auto x = norm_(input);
    x = x.permute({0, 2, 1}); // (batch, hidden, seq) for conv1d

    x = pointwise_conv1_(x); // (batch, 2*hidden, seq)
    x = ops::glu(x, /*dim=*/1);

    x = depthwise_conv_(x);
    x = batch_norm_(x);
    x = ops::silu(x);

    x = pointwise_conv2_(x);
    x = dropout_(x);
    x = x.permute({0, 2, 1}); // back to (batch, seq, hidden)

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

    int num_heads = mha_.num_heads();

    // Project Q, K, V — Linear::forward() dispatches to int8_matmul
    // automatically when weight is Int8 + scale is loaded (WAS-27 fast path).
    // Bias is applied inside Linear::forward(), no explicit add needed here.
    auto q = mha_.q_proj()(query);
    auto k = mha_.k_proj()(key);
    auto v = mha_.v_proj()(value);

    auto d_model = static_cast<int>(q.shape().back());
    int head_dim = d_model / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto batch = q.shape()[0];
    auto seq_len = q.shape()[1];

    // Reshape to multi-head: (batch, seq, heads, head_dim) → (batch, heads,
    // seq, head_dim)
    auto nh = static_cast<size_t>(num_heads);
    auto hd = static_cast<size_t>(head_dim);
    q = q.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    k = k.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    v = v.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});

    // q: (batch, heads, seq, head_dim)
    // pos_bias_u/v: (heads, head_dim) → broadcast as (1, heads, 1, head_dim)
    auto bias_u = pos_bias_u_.reshape({1, nh, 1, hd});
    auto bias_v = pos_bias_v_.reshape({1, nh, 1, hd});

    // Content attention: (Q + pos_bias_u) @ K^T → (batch, heads, seq, seq)
    auto content_score = ops::matmul(q + bias_u, k, false, true);

    // Position attention: project position embeddings
    auto p = pos_proj_(pos_emb); // (2*seq-1, d_model)
    auto pos_len = p.shape()[0];
    // Reshape to (1, 2*seq-1, heads, head_dim) → (1, heads, 2*seq-1, head_dim)
    p = p.reshape({1, pos_len, nh, hd}).transpose({0, 2, 1, 3});

    // (Q + pos_bias_v) @ P^T → (batch, heads, seq, 2*seq-1)
    auto pos_score = ops::matmul(q + bias_v, p, false, true);

    // Shift to align relative positions
    pos_score = rel_shift(pos_score);

    // Combined scores
    auto scores = (content_score + pos_score) * scale;

    // Apply mask if present
    if (mask.storage()) {
        scores = ops::masked_fill(scores, mask, -1e9f);
    }

    // Softmax over last dim
    auto attn_weights = ops::softmax(scores, -1);

    // Weighted sum: (batch, heads, seq, head_dim)
    auto out = ops::matmul(attn_weights, v);

    // Reshape back: (batch, seq, d_model)
    out = out.transpose({0, 2, 1, 3});
    out = out.reshape({batch, seq_len, static_cast<size_t>(d_model)});

    // Output projection — Linear::forward() handles int8 dispatch automatically.
    return mha_.out_proj()(out);
}

Tensor ConformerAttention::forward(const Tensor &input, const Tensor &pos_emb,
                                   const Tensor &mask) const {
    auto x = norm_(input);
    x = rel_position_attention(x, x, x, pos_emb, mask);
    x = dropout_(x);
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

Tensor ConformerBlock::forward(const Tensor &input, const Tensor &pos_emb,
                               const Tensor &mask) const {
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
    }
}

Tensor FastConformerEncoder::forward(const Tensor &input,
                                     const Tensor &mask) const {
    // Top-level encoder phase signposts. Inner ConformerBlock::forward
    // emits FFN1/Attn/Conv/FFN2/BlockFinalNorm signposts; those nest
    // inside ConformerBlocks so Instruments shows the hierarchy.
    PARAKEET_SP_BEGIN(Encoder);

    Tensor x;
    {
        PARAKEET_SP_BEGIN(Subsampling);
        x = subsampling_(input);
        PARAKEET_SP_END(Subsampling);
    }

    int seq_len = static_cast<int>(x.shape()[1]);
    int d_model = static_cast<int>(x.shape()[2]);
    Tensor pos_emb;
    {
        PARAKEET_SP_BEGIN(PosEmb);
        pos_emb = axiom::nn::sinusoidal_position_embedding(
            seq_len, d_model, x.dtype(), x.device());
        PARAKEET_SP_END(PosEmb);
    }

    {
        PARAKEET_SP_BEGIN(ConformerBlocks);
        for (const auto &block : layers_.each<ConformerBlock>()) {
            x = block(x, pos_emb, mask);
        }
        PARAKEET_SP_END(ConformerBlocks);
    }

    PARAKEET_SP_END(Encoder);
    return x;
}

} // namespace parakeet::models
