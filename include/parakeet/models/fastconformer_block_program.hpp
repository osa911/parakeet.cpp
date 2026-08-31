#pragma once

#include <cstddef>
#include <tuple>

#include <axiom/tensor.hpp>

namespace parakeet::models {

class ConformerBlock;
class ConformerAttention;
class FeedForward;
class FastConformerBlockProgramTestAccess;

class FastConformerBlockProgram {
  public:
    static bool is_supported(const ConformerBlock &block,
                             const axiom::Tensor &input,
                             const axiom::Tensor &pos_emb,
                             const axiom::Tensor &mask);
    static axiom::Tensor encode(const ConformerBlock &block,
                                const axiom::Tensor &input,
                                const axiom::Tensor &pos_emb,
                                const axiom::Tensor &mask);

    static size_t direct_dispatches_for_testing();
    static void reset_trace_for_testing();

  private:
    friend class FastConformerBlockProgramTestAccess;

    static const axiom::Tensor *find_cached_head_major_position(
        const ConformerAttention &attention, const axiom::Tensor &pos_emb,
        size_t heads, size_t time);
    static bool supports_direct_ffn(const FeedForward &ffn,
                                    const axiom::Tensor &input,
                                    size_t hidden);

    static std::tuple<axiom::Tensor, axiom::Tensor, axiom::Tensor,
                      axiom::Tensor, size_t, size_t, size_t>
    encode_ffn_pair_for_testing(const ConformerBlock &block,
                                const axiom::Tensor &input);

    static std::tuple<axiom::Tensor, axiom::Tensor, axiom::Tensor,
                      axiom::Tensor, axiom::Tensor, axiom::Tensor,
                      axiom::Tensor, axiom::Tensor, axiom::Tensor, size_t,
                      size_t, size_t>
    encode_attention_for_testing(const ConformerBlock &block,
                                 const axiom::Tensor &input,
                                 const axiom::Tensor &head_major_position,
                                 const axiom::Tensor &mask);

    static bool is_direct_attention_segment_ready(
        const ConformerBlock &block, const axiom::Tensor &input,
        const axiom::Tensor &head_major_position, const axiom::Tensor &mask);
};

} // namespace parakeet::models
