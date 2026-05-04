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

    // Highest log-probability the joint network assigned to a tracked token
    // (e.g. <EOU>) across frames in the most recent decode call. Reset at
    // the start of every rnnt_streaming_decode_chunk invocation. -inf means
    // no tracking is active or no frames were processed.
    float last_tracked_score = -1e30f;
};

// Decode a single encoder chunk incrementally.
// Returns new tokens emitted in this chunk.
//
// `tracked_token_id` (default -1 = disabled): if set to a valid vocab index,
// the joint's log-probability for that token is captured per-frame and the
// chunk-max is written to state.last_tracked_score. Used by callers that
// want to react to a token's confidence before the decoder commits it
// (e.g. semantic endpointing on the EOU token).
std::vector<int> rnnt_streaming_decode_chunk(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_chunk,
    const std::vector<int> &durations, StreamingDecodeState &state,
    int blank_id = 1024, int max_symbols_per_step = 10,
    int tracked_token_id = -1);

} // namespace parakeet::models
