#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <axiom/axiom.hpp>

#include "parakeet/decode/timestamp.hpp"
#include "parakeet/decode/vocab.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/rnnt.hpp"
#include "parakeet/models/tdt.hpp"
#include "parakeet/models/tdt_ctc.hpp"

namespace parakeet::decode {

using namespace axiom;
using models::ParakeetTDT;
using models::ParakeetTDTCTC;
using models::RNNTPrediction;
using models::TDTJoint;

// ─── Context Trie for Phrase Boosting ───────────────────────────────────────

struct TrieNode {
    std::unordered_map<int, int> children; // token_id → child node index
    bool is_end = false;
};

class ContextTrie {
  public:
    ContextTrie();

    // Insert a token ID sequence into the trie.
    void insert(const std::vector<int> &token_ids);

    // Build from a list of phrases using the given tokenizer.
    void build(const std::vector<std::string> &phrases,
               const Tokenizer &tokenizer);

    // Get the set of boosted token IDs from the union of children across
    // all active states.
    std::unordered_set<int>
    get_boosted_tokens(const std::unordered_set<int> &active_states) const;

    // Advance active states given a new token emission.
    // Always includes root (state 0).
    std::unordered_set<int>
    advance(const std::unordered_set<int> &active_states, int token_id) const;

    // Number of nodes in the trie.
    size_t size() const { return nodes_.size(); }

    // Whether the trie has any phrases.
    bool empty() const { return nodes_.size() <= 1; }

  private:
    std::vector<TrieNode> nodes_;
};

// ─── Boosted CTC Decode ────────────────────────────────────────────────────

std::vector<std::vector<int>>
ctc_greedy_decode_boosted(const Tensor &log_probs, const ContextTrie &trie,
                          float boost_score = 5.0f, int blank_id = 1024,
                          const std::vector<int> &lengths = {});

std::vector<std::vector<TimestampedToken>>
ctc_greedy_decode_with_timestamps_boosted(const Tensor &log_probs,
                                          const ContextTrie &trie,
                                          float boost_score = 5.0f,
                                          int blank_id = 1024,
                                          const std::vector<int> &lengths = {});

// ─── Boosted TDT Decode ────────────────────────────────────────────────────

// Component-based
std::vector<std::vector<int>> tdt_greedy_decode_boosted(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, const ContextTrie &trie,
    float boost_score = 5.0f, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps_boosted(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, const ContextTrie &trie,
    float boost_score = 5.0f, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

// Convenience: ParakeetTDT
std::vector<std::vector<int>>
tdt_greedy_decode_boosted(ParakeetTDT &model, const Tensor &encoder_out,
                          const std::vector<int> &durations,
                          const ContextTrie &trie, float boost_score = 5.0f,
                          int blank_id = 1024, int max_symbols_per_step = 10,
                          const std::vector<int> &lengths = {});

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps_boosted(
    ParakeetTDT &model, const Tensor &encoder_out,
    const std::vector<int> &durations, const ContextTrie &trie,
    float boost_score = 5.0f, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

// Convenience: ParakeetTDTCTC
std::vector<std::vector<int>>
tdt_greedy_decode_boosted(ParakeetTDTCTC &model, const Tensor &encoder_out,
                          const std::vector<int> &durations,
                          const ContextTrie &trie, float boost_score = 5.0f,
                          int blank_id = 1024, int max_symbols_per_step = 10,
                          const std::vector<int> &lengths = {});

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps_boosted(
    ParakeetTDTCTC &model, const Tensor &encoder_out,
    const std::vector<int> &durations, const ContextTrie &trie,
    float boost_score = 5.0f, int blank_id = 1024,
    int max_symbols_per_step = 10, const std::vector<int> &lengths = {});

} // namespace parakeet::decode
