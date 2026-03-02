#include "parakeet/models/silero_vad.hpp"

#include <cmath>

namespace parakeet::models {

// ─── SileroVADModel ──────────────────────────────────────────────────────────

SileroVADModel::SileroVADModel(const SileroVADConfig &config)
    : config_(config), stft_conv_(/*stride=*/config.hop_length,
                                  /*padding=*/0), // (258, 1, 256), stride=128
      conv0_(/*stride=*/1, /*padding=*/1),        // (129→128, k=3, p=1)
      conv1_(/*stride=*/2, /*padding=*/1),        // (128→64, k=3, s=2, p=1)
      conv2_(/*stride=*/2, /*padding=*/1),        // (64→64, k=3, s=2, p=1)
      conv3_(/*stride=*/1, /*padding=*/1),        // (64→128, k=3, p=1)
      rnn_(/*num_layers=*/1),
      out_conv_(/*stride=*/1, /*padding=*/0) // (128→1, k=1)
{
    AX_REGISTER_MODULES(stft_conv_, conv0_, conv1_, conv2_, conv3_, rnn_,
                        out_conv_);
}

// Reflection padding: mirror the edges of 1D input
// pads left and right by `pad` samples
// input: (batch, channels, length) → (batch, channels, length + 2*pad)
static Tensor reflection_pad_1d(const Tensor &x, int pad) {
    // x shape: (batch, C, L)
    // Reflects pad samples from each edge (excluding the edge itself)
    auto L = static_cast<int>(x.shape()[2]);

    std::vector<Tensor> parts;

    if (pad > 0) {
        // Left: x[:, :, 1:pad+1] reversed along axis 2
        auto left = x.slice({Slice(), Slice(), Slice(1, pad + 1)}).flip(2);
        parts.push_back(left);
    }

    parts.push_back(x);

    if (pad > 0) {
        // Right: x[:, :, L-pad-1:L-1] reversed along axis 2
        auto right =
            x.slice({Slice(), Slice(), Slice(L - pad - 1, L - 1)}).flip(2);
        parts.push_back(right);
    }

    return Tensor::cat(parts, /*axis=*/2);
}

std::tuple<Tensor, Tensor, Tensor>
SileroVADModel::forward(const Tensor &x, const Tensor &h, const Tensor &c) {
    // x: (batch, num_samples + context_size) = (batch, 576)
    // h, c: (1, batch, 128) — LSTM hidden/cell states

    int pad = config_.filter_length / 2; // 128

    // ── STFT ─────────────────────────────────────────────────────────────────
    // Reshape to (batch, 1, length) for Conv1d
    auto signal = x.unsqueeze(1);

    // ReflectionPad1d(filter_length/2)
    signal = reflection_pad_1d(signal, pad);

    // Conv1d STFT: (batch, 1, padded) → (batch, 258, frames)
    auto spec = stft_conv_(signal);

    // Split into real and imaginary parts at cutoff = filter_length/2 + 1 = 129
    int cutoff = config_.filter_length / 2 + 1; // 129
    auto real_part = spec.slice({Slice(), Slice(0, cutoff), Slice()});
    auto imag_part = spec.slice({Slice(), Slice(cutoff), Slice()});

    // Skip first frame (index 0), keep frames 1..end
    real_part = real_part.slice({Slice(), Slice(), Slice(1)});
    imag_part = imag_part.slice({Slice(), Slice(), Slice(1)});

    // Magnitude: sqrt(real^2 + imag^2)
    auto magnitude = (real_part * real_part + imag_part * imag_part).sqrt();
    // magnitude: (batch, 129, 4)

    // ── Encoder ──────────────────────────────────────────────────────────────
    auto enc = ops::relu(conv0_(magnitude)); // (batch, 128, 4)
    enc = ops::relu(conv1_(enc));            // (batch, 64, 2)
    enc = ops::relu(conv2_(enc));            // (batch, 64, 1)
    enc = ops::relu(conv3_(enc));            // (batch, 128, 1)

    // ── Decoder ──────────────────────────────────────────────────────────────
    // Permute (batch, 128, 1) → (batch, 1, 128) → squeeze → (batch, 128)
    auto dec_in = enc.squeeze(2); // (batch, 128)

    // LSTM step: expects (batch, features), states = [(h, c)]
    // h, c come in as (1, batch, 128) — squeeze to (batch, 128)
    auto h_sq = h.squeeze(0);
    auto c_sq = c.squeeze(0);
    std::vector<LSTMState> states = {{h_sq, c_sq}};
    auto lstm_out = rnn_.step(dec_in, states);

    // Extract updated states and unsqueeze back to (1, batch, 128)
    auto h_new = states[0].first.unsqueeze(0);
    auto c_new = states[0].second.unsqueeze(0);

    // ReLU → Conv1d(128→1, k=1)
    auto dec = ops::relu(lstm_out);
    // Reshape to (batch, 128, 1) for Conv1d
    dec = dec.unsqueeze(2);
    dec = out_conv_(dec); // (batch, 1, 1)

    // Sigmoid → scalar probability
    auto prob = ops::sigmoid(dec).squeeze(2).squeeze(1); // (batch,)

    return {prob, h_new, c_new};
}

std::pair<Tensor, Tensor> SileroVADModel::get_initial_states(int batch_size) {
    auto h = Tensor::zeros({1, static_cast<size_t>(batch_size), 128});
    auto c = Tensor::zeros({1, static_cast<size_t>(batch_size), 128});
    return {h, c};
}

Tensor SileroVADModel::predict(const Tensor &audio) {
    // audio: (num_samples,) — raw 16kHz float32 waveform
    int total_samples = static_cast<int>(audio.shape()[0]);
    int window = config_.num_samples;   // 512
    int context = config_.context_size; // 64

    // Initialize context buffer (zeros)
    auto ctx = Tensor::zeros({static_cast<size_t>(context)});

    // Initialize LSTM states
    auto [h, c] = get_initial_states(1);

    std::vector<Tensor> probs;

    for (int offset = 0; offset < total_samples; offset += window) {
        int remaining = total_samples - offset;
        int chunk_len = std::min(window, remaining);

        Tensor chunk;
        if (chunk_len < window) {
            // Pad last chunk with zeros
            auto partial = audio.slice({Slice(offset, offset + chunk_len)});
            auto pad = Tensor::zeros({static_cast<size_t>(window - chunk_len)});
            chunk = Tensor::cat({partial, pad}, 0);
        } else {
            chunk = audio.slice({Slice(offset, offset + window)});
        }

        // Prepend context → (576,)
        auto input = Tensor::cat({ctx, chunk}, 0);

        // Update context for next iteration (last 64 samples of current chunk)
        ctx = chunk.slice({Slice(window - context)});

        // Add batch dimension: (1, 576)
        input = input.unsqueeze(0);

        auto [prob, h_new, c_new] = forward(input, h, c);
        h = h_new;
        c = c_new;

        probs.push_back(prob.squeeze(0)); // scalar
    }

    return Tensor::stack(probs, 0); // (N,)
}

} // namespace parakeet::models
