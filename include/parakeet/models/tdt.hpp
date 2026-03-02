#pragma once

#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/decode/timestamp.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/encoder.hpp"
#include "parakeet/models/rnnt.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;
using decode::TimestampedToken;

// ─── TDT Joint Network ─────────────────────────────────────────────────────

// Like RNNT Joint but with dual heads:
//   shared = Linear(enc) + Linear(pred, no bias) → ReLU
//   label_logits  = Linear(shared → vocab_size)   → log_softmax
//   duration_logits = Linear(shared → n_durations) → log_softmax

class TDTJoint : public Module {
  public:
    TDTJoint(const JointConfig &config, int num_durations);

    struct Output {
        Tensor label_log_probs;    // (batch, vocab_size)
        Tensor duration_log_probs; // (batch, num_durations)
    };

    Output forward(const Tensor &encoder_out,
                   const Tensor &prediction_out) const;

    const JointConfig &config() const { return config_; }
    int num_durations() const { return num_durations_; }

  private:
    JointConfig config_;
    int num_durations_;
    Linear enc_proj_;
    Linear pred_proj_;
    Linear label_proj_;
    Linear duration_proj_;
};

// ─── Parakeet TDT Model ────────────────────────────────────────────────────

class ParakeetTDT : public Module {
  public:
    explicit ParakeetTDT(const TDTConfig &config = {});

    const TDTConfig &config() const { return config_; }

    FastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &joint() { return joint_; }

  private:
    TDTConfig config_;
    FastConformerEncoder encoder_;
    RNNTPrediction prediction_;
    TDTJoint joint_;
};

// ─── TDT Greedy Decode ─────────────────────────────────────────────────────

// Component-based: decode using prediction + joint directly.
// If lengths is non-empty, use lengths[b] instead of seq_len for each element.
std::vector<std::vector<int>>
tdt_greedy_decode(RNNTPrediction &prediction, TDTJoint &joint,
                  const Tensor &encoder_out, const std::vector<int> &durations,
                  int blank_id = 1024, int max_symbols_per_step = 10,
                  const std::vector<int> &lengths = {});

// Convenience: decode using a full ParakeetTDT model.
std::vector<std::vector<int>>
tdt_greedy_decode(ParakeetTDT &model, const Tensor &encoder_out,
                  const std::vector<int> &durations, int blank_id = 1024,
                  int max_symbols_per_step = 10,
                  const std::vector<int> &lengths = {});

// ─── Timestamped TDT Greedy Decode ───────────────────────────────────────────

// Component-based: decode with timestamps using prediction + joint directly.
std::vector<std::vector<TimestampedToken>> tdt_greedy_decode_with_timestamps(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

// Convenience: decode with timestamps using a full ParakeetTDT model.
std::vector<std::vector<TimestampedToken>> tdt_greedy_decode_with_timestamps(
    ParakeetTDT &model, const Tensor &encoder_out,
    const std::vector<int> &durations, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

} // namespace parakeet::models
