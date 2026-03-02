#pragma once

#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/decode/timestamp.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/ctc.hpp"
#include "parakeet/models/encoder.hpp"
#include "parakeet/models/tdt.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;
using decode::TimestampedToken;

// ─── Parakeet TDT-CTC Hybrid Model ─────────────────────────────────────────

// Shared encoder with both a TDT head and a CTC head.
// Can switch decoder at inference time.

class ParakeetTDTCTC : public Module {
  public:
    explicit ParakeetTDTCTC(const TDTCTCConfig &config = {});

    const TDTCTCConfig &config() const { return config_; }

    FastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &tdt_joint() { return tdt_joint_; }
    CTCDecoder &ctc_decoder() { return ctc_decoder_; }

  private:
    TDTCTCConfig config_;
    FastConformerEncoder encoder_;

    // TDT head
    RNNTPrediction prediction_;
    TDTJoint tdt_joint_;

    // CTC head
    CTCDecoder ctc_decoder_;
};

// ─── TDT-CTC Decode Helpers ───────────────────────────────────────────────

// TDT greedy decode using the TDT head of a ParakeetTDTCTC model.
std::vector<std::vector<int>>
tdt_greedy_decode(ParakeetTDTCTC &model, const Tensor &encoder_out,
                  const std::vector<int> &durations, int blank_id = 1024,
                  int max_symbols_per_step = 10,
                  const std::vector<int> &lengths = {});

// Timestamped TDT greedy decode using the TDT head of a ParakeetTDTCTC model.
std::vector<std::vector<TimestampedToken>> tdt_greedy_decode_with_timestamps(
    ParakeetTDTCTC &model, const Tensor &encoder_out,
    const std::vector<int> &durations, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

} // namespace parakeet::models
