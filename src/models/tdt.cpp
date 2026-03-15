#include "parakeet/models/tdt.hpp"

#include <cmath>

namespace parakeet::models {

// ─── TDTJoint ───────────────────────────────────────────────────────────────

TDTJoint::TDTJoint(const JointConfig &config, int num_durations)
    : config_(config), num_durations_(num_durations), enc_proj_(true),
      pred_proj_(true), label_proj_(true), duration_proj_(true) {
    AX_REGISTER_MODULES(enc_proj_, pred_proj_, label_proj_, duration_proj_);
}

TDTJoint::Output TDTJoint::forward(const Tensor &encoder_out,
                                   const Tensor &prediction_out) const {
    return forward_projected(enc_proj_(encoder_out), prediction_out);
}

TDTJoint::Output TDTJoint::forward_projected(
    const Tensor &enc_projected, const Tensor &prediction_out) const {
    auto hidden = enc_projected + pred_proj_(prediction_out);
    hidden = ops::relu(hidden);

    auto label_logits = ops::log_softmax(label_proj_(hidden), /*axis=*/-1);

    // Pure RNNT models have no duration head (num_durations_ == 0)
    Tensor dur_logits;
    if (num_durations_ > 0) {
        dur_logits = ops::log_softmax(duration_proj_(hidden), /*axis=*/-1);
    }

    return {label_logits, dur_logits};
}

// ─── ParakeetTDT ────────────────────────────────────────────────────────────

ParakeetTDT::ParakeetTDT(const TDTConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

// ─── TDT Greedy Decode ─────────────────────────────────────────────────────

std::vector<std::vector<int>>
tdt_greedy_decode(RNNTPrediction &prediction, TDTJoint &joint,
                  const Tensor &encoder_out, const std::vector<int> &durations,
                  int blank_id, int max_symbols_per_step,
                  const std::vector<int> &lengths) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    // Pre-project all encoder frames once (avoids per-frame enc_proj_)
    auto enc_projected = joint.project_encoder(encoder_out);

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = enc_projected.slice({Slice(b, b + 1)});
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;

        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}, encoder_out.dtype()),
                         Tensor::zeros({1, hs}, encoder_out.dtype())};
        }

        auto token = Tensor({1}, DType::Int32);
        token.fill(blank_id);
        int t = 0;

        while (t < T) {
            auto enc_t = enc.slice({Slice(), Slice(t, t + 1)});

            bool frame_advanced = false;
            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto saved_states = states;

                auto pred = prediction.step(token, states);
                pred = pred.unsqueeze(1);

                auto [label_lp, dur_lp] =
                    joint.forward_projected(enc_t, pred);

                // GPU argmax: transfers only 4 bytes instead of vocab_size*4
                int token_id =
                    ops::argmax(label_lp.squeeze(0).squeeze(0), -1)
                        .item<int>();

                int skip = 1;
                if (dur_lp.storage()) {
                    int dur_idx =
                        ops::argmax(dur_lp.squeeze(0).squeeze(0), -1)
                            .item<int>();
                    skip = (dur_idx < static_cast<int>(durations.size()))
                               ? durations[dur_idx]
                               : 1;
                }

                if (token_id == blank_id) {
                    states = saved_states;
                    t += std::max(skip, 1);
                    frame_advanced = true;
                    break;
                }

                results[b].push_back(token_id);
                token = Tensor({1}, DType::Int32);
                token.fill(token_id);

                if (skip > 0) {
                    t += skip;
                    frame_advanced = true;
                    break;
                }
            }

            if (!frame_advanced) {
                t += 1;
            }
        }
    }

    return results;
}

std::vector<std::vector<int>>
tdt_greedy_decode(ParakeetTDT &model, const Tensor &encoder_out,
                  const std::vector<int> &durations, int blank_id,
                  int max_symbols_per_step, const std::vector<int> &lengths) {
    return tdt_greedy_decode(model.prediction(), model.joint(), encoder_out,
                             durations, blank_id, max_symbols_per_step,
                             lengths);
}

// ─── Timestamped TDT Greedy Decode ────────────────────────────────────────

std::vector<std::vector<TimestampedToken>> tdt_greedy_decode_with_timestamps(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, int blank_id, int max_symbols_per_step,
    const std::vector<int> &lengths) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    // Pre-project all encoder frames once
    auto enc_projected = joint.project_encoder(encoder_out);

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = enc_projected.slice({Slice(b, b + 1)});
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;

        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}, encoder_out.dtype()),
                         Tensor::zeros({1, hs}, encoder_out.dtype())};
        }

        auto token = Tensor({1}, DType::Int32);
        token.fill(blank_id);
        int t = 0;

        while (t < T) {
            auto enc_t = enc.slice({Slice(), Slice(t, t + 1)});

            bool frame_advanced = false;
            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto saved_states = states;

                auto pred = prediction.step(token, states);
                pred = pred.unsqueeze(1);

                auto [label_lp, dur_lp] =
                    joint.forward_projected(enc_t, pred);

                // Pull label log-probs to CPU for argmax + confidence
                auto label_1d =
                    label_lp.squeeze(0).squeeze(0).to_contiguous_cpu();
                const float *label_data = label_1d.typed_data<float>();
                int vocab_size = static_cast<int>(label_1d.shape()[0]);
                int token_id = 0;
                float best_lp = label_data[0];
                for (int v = 1; v < vocab_size; ++v) {
                    if (label_data[v] > best_lp) {
                        best_lp = label_data[v];
                        token_id = v;
                    }
                }
                float confidence = std::exp(best_lp);

                int skip = 1;
                if (dur_lp.storage()) {
                    int dur_idx =
                        ops::argmax(dur_lp.squeeze(0).squeeze(0), -1)
                            .item<int>();
                    skip = (dur_idx < static_cast<int>(durations.size()))
                               ? durations[dur_idx]
                               : 1;
                }

                if (token_id == blank_id) {
                    states = saved_states;
                    t += std::max(skip, 1);
                    frame_advanced = true;
                    break;
                }

                int end_frame = t + std::max(skip, 1) - 1;
                if (end_frame >= T)
                    end_frame = T - 1;
                results[b].push_back({token_id, t, end_frame, confidence});

                token = Tensor({1}, DType::Int32);
                token.fill(token_id);

                if (skip > 0) {
                    t += skip;
                    frame_advanced = true;
                    break;
                }
            }

            if (!frame_advanced) {
                t += 1;
            }
        }
    }

    return results;
}

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps(ParakeetTDT &model, const Tensor &encoder_out,
                                  const std::vector<int> &durations,
                                  int blank_id, int max_symbols_per_step,
                                  const std::vector<int> &lengths) {
    return tdt_greedy_decode_with_timestamps(model.prediction(), model.joint(),
                                             encoder_out, durations, blank_id,
                                             max_symbols_per_step, lengths);
}

} // namespace parakeet::models
