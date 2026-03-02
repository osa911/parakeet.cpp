/**
 * parakeet_c.h — Flat C API for parakeet.cpp
 *
 * Pure C99 header. All types are opaque pointers or plain structs.
 * Thread-safe error reporting via thread-local storage.
 *
 * String pointers returned by result/audio accessors point into the parent
 * handle's memory and are valid until that handle is freed.
 */

#ifndef PARAKEET_C_H
#define PARAKEET_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ─────────────────────────────────────────────────────────────── */

const char *parakeet_version(void);

/* ── Error Handling ──────────────────────────────────────────────────────── */

typedef enum {
    PARAKEET_OK = 0,
    PARAKEET_ERROR_INVALID_ARGUMENT = 1,
    PARAKEET_ERROR_IO = 2,
    PARAKEET_ERROR_MODEL = 3,
    PARAKEET_ERROR_RUNTIME = 4,
    PARAKEET_ERROR_NULL_HANDLE = 5,
} parakeet_error_t;

/** Retrieve the last error message (thread-local). NULL if no error. */
const char *parakeet_last_error(void);

/** Clear the last error message. */
void parakeet_clear_error(void);

/* ── Opaque Handle Types ─────────────────────────────────────────────────── */

typedef struct parakeet_config_s                *parakeet_config_t;
typedef struct parakeet_options_s               *parakeet_options_t;
typedef struct parakeet_transcriber_s           *parakeet_transcriber_t;
typedef struct parakeet_tdt_transcriber_s       *parakeet_tdt_transcriber_t;
typedef struct parakeet_streaming_transcriber_s *parakeet_streaming_transcriber_t;
typedef struct parakeet_nemotron_transcriber_s  *parakeet_nemotron_transcriber_t;
typedef struct parakeet_diarized_transcriber_s  *parakeet_diarized_transcriber_t;
typedef struct parakeet_result_s               *parakeet_result_t;
typedef struct parakeet_diarized_result_s       *parakeet_diarized_result_t;
typedef struct parakeet_audio_s                *parakeet_audio_t;

/* ── Plain C Structs (returned by value) ─────────────────────────────────── */

typedef struct {
    const char *word; /* points into parent result handle */
    float start;
    float end;
    float confidence;
} parakeet_word_timestamp_t;

typedef struct {
    int token_id;
    int start_frame;
    int end_frame;
    float confidence;
} parakeet_timestamped_token_t;

typedef struct {
    const char *word; /* points into parent result handle */
    float start;
    float end;
    int speaker_id;
    float confidence;
} parakeet_diarized_word_t;

typedef struct {
    int speaker_id;
    float start;
    float end;
} parakeet_diarization_segment_t;

/* ── Enums ───────────────────────────────────────────────────────────────── */

typedef enum {
    PARAKEET_DECODER_CTC = 0,
    PARAKEET_DECODER_TDT = 1,
} parakeet_decoder_t;

typedef enum {
    PARAKEET_AUDIO_FORMAT_UNKNOWN = 0,
    PARAKEET_AUDIO_FORMAT_WAV,
    PARAKEET_AUDIO_FORMAT_FLAC,
    PARAKEET_AUDIO_FORMAT_MP3,
    PARAKEET_AUDIO_FORMAT_OGG,
} parakeet_audio_format_t;

/* ── Config Presets ──────────────────────────────────────────────────────── */

parakeet_config_t parakeet_config_110m(void);
parakeet_config_t parakeet_config_tdt_600m(void);
parakeet_config_t parakeet_config_eou_120m(void);
parakeet_config_t parakeet_config_nemotron_600m(int latency_frames);
parakeet_config_t parakeet_config_sortformer_117m(void);
void              parakeet_config_free(parakeet_config_t config);

/* ── Transcribe Options (builder pattern) ────────────────────────────────── */

parakeet_options_t parakeet_options_create(void);
void               parakeet_options_free(parakeet_options_t opts);
parakeet_error_t   parakeet_options_set_decoder(parakeet_options_t opts,
                                                parakeet_decoder_t decoder);
parakeet_error_t   parakeet_options_set_timestamps(parakeet_options_t opts,
                                                   int enable);
parakeet_error_t   parakeet_options_set_boost_score(parakeet_options_t opts,
                                                    float score);
parakeet_error_t   parakeet_options_add_boost_phrase(parakeet_options_t opts,
                                                     const char *phrase);

/* ── Audio I/O ───────────────────────────────────────────────────────────── */

/** Read audio from a file path. Returns NULL on failure (check parakeet_last_error). */
parakeet_audio_t parakeet_audio_read_file(const char *path,
                                          int target_sample_rate);

/** Create audio from float32 PCM samples. */
parakeet_audio_t parakeet_audio_read_f32(const float *pcm, size_t n, int rate,
                                         int target_rate);

/** Create audio from int16 PCM samples. */
parakeet_audio_t parakeet_audio_read_i16(const int16_t *pcm, size_t n, int rate,
                                         int target_rate);

/** Create audio from encoded bytes (WAV/FLAC/MP3/OGG auto-detected). */
parakeet_audio_t parakeet_audio_read_encoded(const uint8_t *data, size_t len,
                                             int target_rate);

void parakeet_audio_free(parakeet_audio_t audio);

int   parakeet_audio_sample_rate(parakeet_audio_t audio);
int   parakeet_audio_num_samples(parakeet_audio_t audio);
float parakeet_audio_duration(parakeet_audio_t audio);
parakeet_audio_format_t parakeet_audio_format(parakeet_audio_t audio);

/** Get pointer to contiguous float32 sample data. Valid while handle exists. */
const float *parakeet_audio_samples(parakeet_audio_t audio, size_t *out_count);

/** Query audio file duration from header only (no full decode). */
float parakeet_audio_file_duration(const char *path);

/* ── Transcriber (110M TDT-CTC) ─────────────────────────────────────────── */

/** Create a 110M TDT-CTC transcriber. config may be NULL for default. */
parakeet_transcriber_t parakeet_transcriber_create(const char *weights,
                                                   const char *vocab,
                                                   parakeet_config_t config);
void             parakeet_transcriber_free(parakeet_transcriber_t t);
parakeet_error_t parakeet_transcriber_to_gpu(parakeet_transcriber_t t);
parakeet_error_t parakeet_transcriber_to_half(parakeet_transcriber_t t);

/** Transcribe an audio file. Returns NULL on failure. opts may be NULL. */
parakeet_result_t parakeet_transcriber_transcribe_file(
    parakeet_transcriber_t t, const char *path, parakeet_options_t opts);

/** Transcribe from float32 PCM (16kHz mono). Returns NULL on failure. opts may be NULL. */
parakeet_result_t parakeet_transcriber_transcribe_pcm(
    parakeet_transcriber_t t, const float *samples, size_t n,
    parakeet_options_t opts);

/** Batch transcribe multiple audio files. Returns array of count results.
 *  Free with parakeet_result_batch_free(). Returns NULL on failure. */
parakeet_result_t *parakeet_transcriber_transcribe_batch(
    parakeet_transcriber_t t, const char **paths, size_t count,
    parakeet_options_t opts);

/* ── TDT Transcriber (600M) ─────────────────────────────────────────────── */

/** Create a 600M TDT transcriber. config may be NULL for default. */
parakeet_tdt_transcriber_t parakeet_tdt_transcriber_create(
    const char *weights, const char *vocab, parakeet_config_t config);
void             parakeet_tdt_transcriber_free(parakeet_tdt_transcriber_t t);
parakeet_error_t parakeet_tdt_transcriber_to_gpu(parakeet_tdt_transcriber_t t);
parakeet_error_t parakeet_tdt_transcriber_to_half(parakeet_tdt_transcriber_t t);
parakeet_result_t parakeet_tdt_transcriber_transcribe_file(
    parakeet_tdt_transcriber_t t, const char *path, parakeet_options_t opts);
parakeet_result_t parakeet_tdt_transcriber_transcribe_pcm(
    parakeet_tdt_transcriber_t t, const float *samples, size_t n,
    parakeet_options_t opts);

/** Batch transcribe multiple audio files. Returns array of count results.
 *  Free with parakeet_result_batch_free(). Returns NULL on failure. */
parakeet_result_t *parakeet_tdt_transcriber_transcribe_batch(
    parakeet_tdt_transcriber_t t, const char **paths, size_t count,
    parakeet_options_t opts);

/* ── Streaming Transcriber (EOU 120M) ────────────────────────────────────── */

/** Create streaming EOU transcriber. config may be NULL for default. */
parakeet_streaming_transcriber_t parakeet_streaming_transcriber_create(
    const char *weights, const char *vocab, parakeet_config_t config);
void             parakeet_streaming_transcriber_free(
    parakeet_streaming_transcriber_t t);
parakeet_error_t parakeet_streaming_transcriber_to_gpu(
    parakeet_streaming_transcriber_t t);
parakeet_error_t parakeet_streaming_transcriber_to_half(
    parakeet_streaming_transcriber_t t);

/**
 * Feed float32 PCM samples. out_text receives pointer to new text from this
 * chunk (valid until next feed/free). Returns error code.
 */
parakeet_error_t parakeet_streaming_transcriber_feed_f32(
    parakeet_streaming_transcriber_t t, const float *samples, size_t n,
    const char **out_text);

/** Feed int16 PCM samples. See feed_f32 for out_text lifetime. */
parakeet_error_t parakeet_streaming_transcriber_feed_i16(
    parakeet_streaming_transcriber_t t, const int16_t *samples, size_t n,
    const char **out_text);

/** Reset streaming state for a new utterance. */
parakeet_error_t parakeet_streaming_transcriber_reset(
    parakeet_streaming_transcriber_t t);

/** Get full transcription accumulated so far. Valid until next feed/reset/free. */
const char *parakeet_streaming_transcriber_get_text(
    parakeet_streaming_transcriber_t t);

/* ── Nemotron Transcriber (600M) ─────────────────────────────────────────── */

/** Create Nemotron streaming transcriber. config may be NULL for default. */
parakeet_nemotron_transcriber_t parakeet_nemotron_transcriber_create(
    const char *weights, const char *vocab, parakeet_config_t config);
void             parakeet_nemotron_transcriber_free(
    parakeet_nemotron_transcriber_t t);
parakeet_error_t parakeet_nemotron_transcriber_to_gpu(
    parakeet_nemotron_transcriber_t t);
parakeet_error_t parakeet_nemotron_transcriber_to_half(
    parakeet_nemotron_transcriber_t t);
parakeet_error_t parakeet_nemotron_transcriber_feed_f32(
    parakeet_nemotron_transcriber_t t, const float *samples, size_t n,
    const char **out_text);
parakeet_error_t parakeet_nemotron_transcriber_feed_i16(
    parakeet_nemotron_transcriber_t t, const int16_t *samples, size_t n,
    const char **out_text);
parakeet_error_t parakeet_nemotron_transcriber_reset(
    parakeet_nemotron_transcriber_t t);
const char *parakeet_nemotron_transcriber_get_text(
    parakeet_nemotron_transcriber_t t);

/* ── Diarized Transcriber ────────────────────────────────────────────────── */

/**
 * Create diarized transcriber combining ASR + Sortformer.
 * config and sf_config may be NULL for defaults.
 */
parakeet_diarized_transcriber_t parakeet_diarized_transcriber_create(
    const char *asr_weights, const char *sortformer_weights,
    const char *vocab, parakeet_config_t config,
    parakeet_config_t sf_config);
void             parakeet_diarized_transcriber_free(
    parakeet_diarized_transcriber_t t);
parakeet_error_t parakeet_diarized_transcriber_to_gpu(
    parakeet_diarized_transcriber_t t);
parakeet_error_t parakeet_diarized_transcriber_to_half(
    parakeet_diarized_transcriber_t t);
parakeet_diarized_result_t parakeet_diarized_transcriber_transcribe_file(
    parakeet_diarized_transcriber_t t, const char *path,
    parakeet_decoder_t decoder);
parakeet_diarized_result_t parakeet_diarized_transcriber_transcribe_pcm(
    parakeet_diarized_transcriber_t t, const float *samples, size_t n,
    parakeet_decoder_t decoder);

/* ── Result Accessors ────────────────────────────────────────────────────── */

/** Free a batch of results returned by transcribe_batch. */
void        parakeet_result_batch_free(parakeet_result_t *results, size_t count);

void        parakeet_result_free(parakeet_result_t r);
const char *parakeet_result_text(parakeet_result_t r);
size_t      parakeet_result_token_count(parakeet_result_t r);
const int  *parakeet_result_token_ids(parakeet_result_t r);
size_t      parakeet_result_timestamped_token_count(parakeet_result_t r);
parakeet_timestamped_token_t parakeet_result_timestamped_token_at(
    parakeet_result_t r, size_t index);
size_t      parakeet_result_word_count(parakeet_result_t r);
parakeet_word_timestamp_t parakeet_result_word_at(parakeet_result_t r,
                                                  size_t index);

/* ── Diarized Result Accessors ───────────────────────────────────────────── */

void        parakeet_diarized_result_free(parakeet_diarized_result_t r);
const char *parakeet_diarized_result_text(parakeet_diarized_result_t r);
size_t      parakeet_diarized_result_word_count(parakeet_diarized_result_t r);
parakeet_diarized_word_t parakeet_diarized_result_word_at(
    parakeet_diarized_result_t r, size_t index);
size_t      parakeet_diarized_result_segment_count(parakeet_diarized_result_t r);
parakeet_diarization_segment_t parakeet_diarized_result_segment_at(
    parakeet_diarized_result_t r, size_t index);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PARAKEET_C_H */
