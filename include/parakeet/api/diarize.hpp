#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/io/safetensors.hpp>

#include "parakeet/api/transcribe.hpp"
#include "parakeet/audio/audio.hpp"
#include "parakeet/audio/audio_io.hpp"
#include "parakeet/audio/vad.hpp"
#include "parakeet/decode/timestamp.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/sortformer.hpp"

namespace parakeet::api {

using namespace models;
using namespace decode;
using namespace audio;

// ─── Diarized Transcription Types ───────────────────────────────────────────

struct DiarizedWord {
    std::string word;
    float start = 0.0f;      // seconds
    float end = 0.0f;        // seconds
    int speaker_id = -1;     // -1 = no overlapping segment
    float confidence = 1.0f; // from ASR word confidence
};

struct DiarizedResult {
    std::string text;
    std::vector<DiarizedWord> words;
    std::vector<DiarizationSegment> segments;   // raw diarization output
    std::vector<WordTimestamp> word_timestamps; // raw ASR timestamps
};

// ─── Alignment ──────────────────────────────────────────────────────────────

/// Assign speaker IDs to words by maximum temporal overlap.
/// Words with no overlapping segment get speaker_id = -1.
std::vector<DiarizedWord>
diarize_transcription(const std::vector<WordTimestamp> &words,
                      const std::vector<DiarizationSegment> &segments);

// ─── High-Level Diarized Transcription API ──────────────────────────────────

/// Combines ASR (Transcriber) with Sortformer diarization to produce
/// speaker-attributed words.
///
///   DiarizedTranscriber dt("asr.safetensors", "sortformer.safetensors",
///                          "vocab.txt");
///   auto result = dt.transcribe("audio.wav");
///   for (auto &w : result.words)
///       std::cout << "Speaker " << w.speaker_id << ": " << w.word << "\n";
///
class DiarizedTranscriber {
  public:
    DiarizedTranscriber(
        const std::string &asr_weights, const std::string &sortformer_weights,
        const std::string &vocab_path,
        const TDTCTCConfig &config = make_110m_config(),
        const SortformerConfig &sf_config = make_sortformer_117m_config());

    void to_gpu();
    void to_half();

    /// Enable VAD preprocessing. Call after to_half()/to_gpu().
    void enable_vad(const std::string &vad_weights_path);

    DiarizedResult transcribe(const std::string &audio_path,
                              Decoder decoder = Decoder::TDT);
    DiarizedResult transcribe(const axiom::Tensor &samples,
                              Decoder decoder = Decoder::TDT);

  private:
    Transcriber transcriber_;
    Sortformer sortformer_;
    SortformerConfig sf_config_;
    bool use_gpu_ = false;
    bool use_fp16_ = false;
    std::unique_ptr<audio::SileroVAD> vad_;
};

} // namespace parakeet::api
