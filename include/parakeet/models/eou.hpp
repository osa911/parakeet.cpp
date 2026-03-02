#pragma once

#include <functional>
#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/decode/timestamp.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/lstm.hpp"
#include "parakeet/models/rnnt.hpp"
#include "parakeet/models/streaming_encoder.hpp"
#include "parakeet/models/tdt.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;
using decode::TimestampedToken;

// ─── Parakeet EOU Model ──────────────────────────────────────────────────────

class ParakeetEOU : public Module {
  public:
    explicit ParakeetEOU(const EOUConfig &config = make_eou_120m_config());

    const EOUConfig &config() const { return config_; }

    StreamingFastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &joint() { return joint_; }

  private:
    EOUConfig config_;
    StreamingFastConformerEncoder encoder_;
    RNNTPrediction prediction_;
    TDTJoint joint_;
};

// ─── Streaming RNNT/TDT Decode ──────────────────────────────────────────────

// Decode state maintained across chunks
struct StreamingDecodeState {
    std::vector<LSTMState> lstm_states;
    Tensor last_token; // (1,) int32
    std::vector<int> tokens;
    std::vector<TimestampedToken> timestamped_tokens;
    int frame_offset = 0; // absolute frame position across chunks
    bool initialized = false;
};

// Decode a single encoder chunk incrementally.
// Returns new tokens emitted in this chunk.
std::vector<int> rnnt_streaming_decode_chunk(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_chunk,
    const std::vector<int> &durations, StreamingDecodeState &state,
    int blank_id = 1024, int max_symbols_per_step = 10);

} // namespace parakeet::models
