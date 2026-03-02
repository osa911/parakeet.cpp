#pragma once

#include <tuple>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/models/lstm.hpp"

namespace parakeet::models {

using namespace axiom;
using namespace axiom::nn;

// ─── Silero VAD v5 Configuration ─────────────────────────────────────────────

struct SileroVADConfig {
    int sample_rate = 16000;
    int filter_length = 256; // STFT window (16000/62.5)
    int hop_length = 128;    // filter_length / 2
    int num_samples = 512;   // window size (filter_length * 2)
    int context_size = 64;   // filter_length / 4
};

// ─── Silero VAD v5 Model ─────────────────────────────────────────────────────
//
// Architecture (309K params, 16 tensors):
//   Input: (batch, 576) = 512 audio samples + 64 context prepended
//
//   STFT:
//     ReflectionPad1d(128) → (batch, 832)
//     Conv1d(258 out, 1 in, kernel=256, stride=128) → (batch, 258, 5)
//     Split real/imag at cutoff=129, skip frame 0 → magnitude (batch, 129, 4)
//
//   Encoder (4x Conv1d + ReLU):
//     Conv1d(129→128, k=3, p=1)       → (batch, 128, 4)
//     Conv1d(128→64,  k=3, s=2, p=1)  → (batch, 64, 2)
//     Conv1d(64→64,   k=3, s=2, p=1)  → (batch, 64, 1)
//     Conv1d(64→128,  k=3, p=1)       → (batch, 128, 1)
//
//   Decoder:
//     Permute → (1, batch, 128)
//     LSTM(128→128), state: h(1,1,128), c(1,1,128)
//     ReLU → Conv1d(128→1, k=1) → Sigmoid
//
//   Output: speech probability [0, 1], updated (h, c) states

class SileroVADModel : public Module {
  public:
    SileroVADModel(const SileroVADConfig &config = {});

    // Single chunk forward: x(batch, 576), h(1,batch,128), c(1,batch,128)
    // Returns: (prob, h_new, c_new)
    std::tuple<Tensor, Tensor, Tensor> forward(const Tensor &x, const Tensor &h,
                                               const Tensor &c);

    // Process full audio: returns per-window speech probabilities
    Tensor predict(const Tensor &audio);

    // Get zero-initialized LSTM states
    std::pair<Tensor, Tensor> get_initial_states(int batch_size = 1);

    const SileroVADConfig &config() const { return config_; }

  private:
    SileroVADConfig config_;

    // STFT: Conv1d with pre-computed basis (no bias), stride=128
    Conv1d stft_conv_;

    // Encoder: 4 conv layers
    Conv1d conv0_; // (129→128, k=3, p=1)
    Conv1d conv1_; // (128→64, k=3, s=2, p=1)
    Conv1d conv2_; // (64→64, k=3, s=2, p=1)
    Conv1d conv3_; // (64→128, k=3, p=1)

    // Decoder: LSTM(128→128, 1 layer) + output projection
    LSTM rnn_;
    Conv1d out_conv_; // (128→1, k=1)
};

} // namespace parakeet::models
