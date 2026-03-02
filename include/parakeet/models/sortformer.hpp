#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/models/config.hpp"
#include "parakeet/models/streaming_encoder.hpp"
#include "parakeet/models/transformer.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;

// ─── AOSC (Arrival-Order Speaker Cache) ──────────────────────────────────────

class AOSCCache {
  public:
    explicit AOSCCache(int max_speakers = 4);

    // Update cache with new frame activity scores.
    // probs: (T, max_speakers) sigmoid probabilities
    void update(const Tensor &probs);

    // Get speaker ordering (arrival order).
    // Returns mapping: output_speaker_id → internal_speaker_id
    std::vector<int> speaker_order() const;

    void reset();

  private:
    int max_speakers_;
    std::vector<bool> speaker_active_;
    std::vector<int> arrival_order_; // speakers in order of first appearance
};

// ─── Sortformer Model ────────────────────────────────────────────────────────

class Sortformer : public Module {
  public:
    explicit Sortformer(
        const SortformerConfig &config = make_sortformer_117m_config());

    // Batch diarization: (1, n_frames, n_mels) → segments
    std::vector<DiarizationSegment> diarize(const Tensor &features) const;

    // Streaming: process a chunk, return segments so far
    std::vector<DiarizationSegment> diarize_chunk(const Tensor &features,
                                                  EncoderCache &enc_cache,
                                                  AOSCCache &aosc_cache) const;

    // Raw forward: features → (batch, T, max_speakers) sigmoid probs
    Tensor forward(const Tensor &features) const;

    const SortformerConfig &config() const { return config_; }

  private:
    SortformerConfig config_;
    StreamingFastConformerEncoder nest_encoder_;
    Linear projection_; // encoder_hidden → transformer_hidden
    TransformerEncoder transformer_;
    Linear output_proj_;    // transformer_hidden → max_speakers
                            // (single_hidden_to_spks)
    Linear first_hidden_;   // transformer_hidden → transformer_hidden
    Linear hidden_to_spks_; // 2*transformer_hidden → max_speakers (concat path)

    // Post-process sigmoid probabilities to segments
    std::vector<DiarizationSegment>
    probs_to_segments(const Tensor &probs) const;
};

} // namespace parakeet::models
