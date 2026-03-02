#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/models/config.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;

// ─── Standard Transformer Block ──────────────────────────────────────────────

// Pre-norm transformer: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual

class TransformerBlock : public Module {
  public:
    explicit TransformerBlock(const TransformerConfig &config = {});

    // input: (batch, seq, hidden)
    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    LayerNorm norm1_;
    MultiHeadAttention mha_;
    Dropout dropout1_;
    LayerNorm norm2_;
    Linear fc1_;
    Linear fc2_;
    Dropout dropout2_;
    bool pre_ln_ = true;
};

// ─── Transformer Encoder ─────────────────────────────────────────────────────

class TransformerEncoder : public Module {
  public:
    explicit TransformerEncoder(const TransformerConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    ModuleList layers_;
    LayerNorm final_norm_;
    bool has_final_norm_ = false;
};

} // namespace parakeet::models
