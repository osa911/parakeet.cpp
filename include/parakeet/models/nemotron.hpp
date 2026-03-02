#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/models/config.hpp"
#include "parakeet/models/rnnt.hpp"
#include "parakeet/models/streaming_encoder.hpp"
#include "parakeet/models/tdt.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;

// ─── Parakeet Nemotron Model ─────────────────────────────────────────────────

class ParakeetNemotron : public Module {
  public:
    explicit ParakeetNemotron(
        const NemotronConfig &config = make_nemotron_600m_config());

    const NemotronConfig &config() const { return config_; }

    StreamingFastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &joint() { return joint_; }

  private:
    NemotronConfig config_;
    StreamingFastConformerEncoder encoder_;
    RNNTPrediction prediction_;
    TDTJoint joint_;
};

} // namespace parakeet::models
