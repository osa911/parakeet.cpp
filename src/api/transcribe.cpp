#include "parakeet/api/transcribe.hpp"

#include <axiom/io/safetensors.hpp>

#include <vector>

namespace parakeet::api {

// ─── StreamingTranscriber ────────────────────────────────────────────────────

StreamingTranscriber::StreamingTranscriber(const std::string &weights_path,
                                           const std::string &vocab_path,
                                           const EOUConfig &config)
    : config_(config), model_(config) {
    // Configure preprocessor with correct mel bins from encoder config
    AudioConfig audio_cfg;
    audio_cfg.n_mels = config.encoder.mel_bins;
    preprocessor_ = StreamingAudioPreprocessor(audio_cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
    tokenizer_.load(vocab_path);
}

std::string
StreamingTranscriber::transcribe_chunk(const axiom::Tensor &samples) {
    // 1. Preprocess chunk
    auto features = preprocessor_.process_chunk(samples);
    if (!features.storage()) {
        return ""; // not enough samples yet
    }

    if (use_fp16_)
        features = features.half();
    if (use_gpu_) {
        features = features.gpu();
    }

    // 2. Encode chunk
    auto encoder_out = model_.encoder().forward_chunk(features, encoder_cache_);
    if (!encoder_out.storage()) {
        return ""; // not enough frames for subsampling
    }

    // 3. Decode chunk (blank = last token in vocab)
    int blank_id = config_.joint.vocab_size - 1;
    auto new_tokens = rnnt_streaming_decode_chunk(
        model_.prediction(), model_.joint(), encoder_out, config_.durations,
        decode_state_, blank_id);

    // 4. Convert new tokens to text
    if (!new_tokens.empty() && tokenizer_.loaded()) {
        auto text = tokenizer_.decode(new_tokens);
        if (partial_callback_) {
            partial_callback_(text);
        }
        return text;
    }

    return "";
}

void StreamingTranscriber::reset() {
    preprocessor_.reset();
    encoder_cache_ = EncoderCache{};
    decode_state_ = StreamingDecodeState{};
    text_delta_offset_ = 0;
}

void StreamingTranscriber::reset_decoder() {
    decode_state_ = StreamingDecodeState{};
    text_delta_offset_ = 0;
}

std::string StreamingTranscriber::finalize(int pad_samples) {
    if (pad_samples > 0) {
        std::vector<float> pad(static_cast<size_t>(pad_samples), 0.0f);
        transcribe_chunk(pad.data(), pad.size());
    }
    return get_text();
}

std::string StreamingTranscriber::get_text() const {
    if (tokenizer_.loaded() && !decode_state_.tokens.empty()) {
        return tokenizer_.decode(decode_state_.tokens);
    }
    return "";
}

std::string StreamingTranscriber::get_text_delta() {
    auto full = get_text();
    if (full.size() <= text_delta_offset_) return "";
    auto d = full.substr(text_delta_offset_);
    text_delta_offset_ = full.size();
    return d;
}

// ─── NemotronTranscriber ─────────────────────────────────────────────────────

NemotronTranscriber::NemotronTranscriber(const std::string &weights_path,
                                         const std::string &vocab_path,
                                         const NemotronConfig &config)
    : config_(config), model_(config) {
    // Configure preprocessor with correct mel bins from encoder config
    AudioConfig audio_cfg;
    audio_cfg.n_mels = config.encoder.mel_bins;
    preprocessor_ = StreamingAudioPreprocessor(audio_cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
    tokenizer_.load(vocab_path);
}

std::string
NemotronTranscriber::transcribe_chunk(const axiom::Tensor &samples) {
    auto features = preprocessor_.process_chunk(samples);
    if (!features.storage()) {
        return "";
    }

    if (use_fp16_)
        features = features.half();
    if (use_gpu_) {
        features = features.gpu();
    }

    auto encoder_out = model_.encoder().forward_chunk(features, encoder_cache_);
    if (!encoder_out.storage()) {
        return "";
    }

    int blank_id = config_.joint.vocab_size - 1;
    auto new_tokens = rnnt_streaming_decode_chunk(
        model_.prediction(), model_.joint(), encoder_out, config_.durations,
        decode_state_, blank_id);

    if (!new_tokens.empty() && tokenizer_.loaded()) {
        auto text = tokenizer_.decode(new_tokens);
        if (partial_callback_) {
            partial_callback_(text);
        }
        return text;
    }

    return "";
}

void NemotronTranscriber::reset() {
    preprocessor_.reset();
    encoder_cache_ = EncoderCache{};
    decode_state_ = StreamingDecodeState{};
    text_delta_offset_ = 0;
}

void NemotronTranscriber::reset_decoder() {
    decode_state_ = StreamingDecodeState{};
    text_delta_offset_ = 0;
}

std::string NemotronTranscriber::finalize(int pad_samples) {
    if (pad_samples > 0) {
        std::vector<float> pad(static_cast<size_t>(pad_samples), 0.0f);
        transcribe_chunk(pad.data(), pad.size());
    }
    return get_text();
}

std::string NemotronTranscriber::get_text() const {
    if (tokenizer_.loaded() && !decode_state_.tokens.empty()) {
        return tokenizer_.decode(decode_state_.tokens);
    }
    return "";
}

std::string NemotronTranscriber::get_text_delta() {
    auto full = get_text();
    if (full.size() <= text_delta_offset_) return "";
    auto d = full.substr(text_delta_offset_);
    text_delta_offset_ = full.size();
    return d;
}

} // namespace parakeet::api
