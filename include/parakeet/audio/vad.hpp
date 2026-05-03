#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/io/safetensors.hpp>

#include "parakeet/decode/timestamp.hpp"
#include "parakeet/models/silero_vad.hpp"

namespace parakeet::audio {

using namespace decode;

// ─── Speech Segment ──────────────────────────────────────────────────────────

struct SpeechSegment {
    int64_t start_sample; // in original audio
    int64_t end_sample;
};

// ─── VAD Configuration ──────────────────────────────────────────────────────

struct VADConfig {
    float threshold = 0.5f;
    float neg_threshold = 0.35f; // threshold - 0.15
    int min_speech_duration_ms = 250;
    int min_silence_duration_ms = 100;
    int speech_pad_ms = 30;
    float max_speech_duration_s = std::numeric_limits<float>::infinity();
};

// ─── Silero VAD Wrapper ──────────────────────────────────────────────────────

class SileroVAD {
  public:
    explicit SileroVAD(const std::string &weights_path);

    void to_gpu();
    void to_half();

    /// Detect speech segments in audio waveform.
    /// audio: (num_samples,) raw float32 16kHz mono
    std::vector<SpeechSegment> detect(const axiom::Tensor &audio,
                                      int sample_rate = 16000,
                                      const VADConfig &config = {});

    /// Process a single chunk (streaming). Returns speech probability.
    ///
    /// Chunks are coerced to the model's native window
    /// (`model.config().num_samples`, 512 samples for Silero v5 at 16 kHz)
    /// before inference: shorter chunks are zero-padded on the right,
    /// longer chunks are truncated to the first `num_samples`. For
    /// best-quality probabilities frame your input at exactly 512 samples
    /// (32 ms). Note that truncation discards trailing audio — passing a
    /// 1024-sample chunk yields one probability for the first 512 samples,
    /// not two probabilities.
    float process_chunk(const axiom::Tensor &chunk, int sample_rate = 16000);

    /// Reset streaming state.
    void reset();

  private:
    models::SileroVADModel model_;
    axiom::Tensor h_, c_;
    axiom::Tensor context_;
    bool use_gpu_ = false;
    bool use_fp16_ = false;
    bool initialized_ = false;

    void ensure_states();
};

// ─── Audio Collection ────────────────────────────────────────────────────────

/// Concatenate speech segments from audio, removing silence.
axiom::Tensor collect_speech(const axiom::Tensor &audio,
                             const std::vector<SpeechSegment> &segments);

// ─── Timestamp Remapping ─────────────────────────────────────────────────────

/// Maps timestamps from compressed (silence-removed) audio back to the
/// original timeline using a piecewise linear mapping.
class TimestampRemapper {
  public:
    TimestampRemapper(const std::vector<SpeechSegment> &segments,
                      int sample_rate = 16000);

    /// Remap a single compressed-domain timestamp (seconds) to original.
    float remap(float compressed_seconds) const;

    /// Remap a word timestamp.
    WordTimestamp remap(const WordTimestamp &w) const;

    /// Remap a vector of word timestamps.
    std::vector<WordTimestamp>
    remap(const std::vector<WordTimestamp> &words) const;

    /// Remap a single timestamped token (frame-level).
    TimestampedToken remap_token(const TimestampedToken &t) const;

    /// Remap a vector of timestamped tokens.
    std::vector<TimestampedToken>
    remap_tokens(const std::vector<TimestampedToken> &tokens) const;

  private:
    std::vector<float> compressed_starts_; // cumulative start of each segment
    std::vector<SpeechSegment> segments_;
    int sample_rate_;
    float total_compressed_; // total compressed duration in seconds
};

} // namespace parakeet::audio
