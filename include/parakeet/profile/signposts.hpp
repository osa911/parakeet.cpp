#pragma once
//
// Opt-in encoder profiling signposts for Xcode Instruments / xctrace.
//
// Apple's os_signpost_interval_begin/_end groups intervals in
// Instruments by the *literal name string* (third macro argument).
// That string must be a compile-time literal — passing a `const char*`
// variable lands in the message field instead, which Instruments does
// NOT use for aggregation.
//
// We work around this with stringification macros: each phase is
// referred to by a C++ identifier (e.g. `FFN1`), and the macros emit
// both a token-pasted variable name AND a `#name`-stringified literal
// for os_signpost. As a result Instruments aggregates intervals per
// phase name across runs — e.g., "FFN1" sums 18x per encoder forward.
//
// Cost: ~5 ns per begin/end pair when os_signpost is disabled (the
// Apple runtime fast-paths `os_signpost_enabled() == false`). No
// production-build behavior change.
//
// Outside macOS the macros are no-ops so this header is safe to
// include unconditionally.
//
// Phase-name constraint: the name must be a valid C++ identifier
// (used as a token-pasted variable name). Use UpperCamelCase joined
// (`FFN1`, `ConformerBlocks`, `BlockFinalNorm`), not space-separated.
//
// Bracing requirement: ALWAYS wrap each begin/end pair in `{ }`. The
// macros emit local declarations (token-pasted log + id variables)
// that the end macro reads back; they cannot be `do { } while(0)`-
// wrapped. Two `PARAKEET_SP_BEGIN(SameName)` calls in the same scope
// fail to compile (variable redeclaration) — intentional, forces
// each interval to live in its own brace block.
//

#if defined(__APPLE__)
#include <os/log.h>
#include <os/signpost.h>
#endif

namespace parakeet::profile {

#if defined(__APPLE__)
inline os_log_t encoder_log() {
    static os_log_t log = os_log_create("com.wasper.parakeet", "encoder");
    return log;
}
#endif

}  // namespace parakeet::profile

#if defined(__APPLE__)

#define PARAKEET_SP_BEGIN(name)                                                \
    os_log_t _sp_log_##name = ::parakeet::profile::encoder_log();              \
    os_signpost_id_t _sp_id_##name =                                           \
        os_signpost_id_generate(_sp_log_##name);                               \
    os_signpost_interval_begin(_sp_log_##name, _sp_id_##name, #name)

#define PARAKEET_SP_END(name)                                                  \
    os_signpost_interval_end(_sp_log_##name, _sp_id_##name, #name)

#else  // !__APPLE__

#define PARAKEET_SP_BEGIN(name) ((void)0)
#define PARAKEET_SP_END(name) ((void)0)

#endif
