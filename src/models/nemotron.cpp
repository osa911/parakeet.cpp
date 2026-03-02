#include "parakeet/models/nemotron.hpp"

namespace parakeet::models {

// ─── ParakeetNemotron ────────────────────────────────────────────────────────

ParakeetNemotron::ParakeetNemotron(const NemotronConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

} // namespace parakeet::models
