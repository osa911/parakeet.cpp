#include <parakeet/api/parakeet_c.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int pass = 0, fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { pass++; } \
    else { fail++; printf("FAIL: %s\n", msg); } \
} while(0)

    /* Version */
    CHECK(parakeet_version() != NULL, "version not null");
    CHECK(strlen(parakeet_version()) > 0, "version not empty");

    /* Error handling */
    parakeet_clear_error();
    CHECK(parakeet_last_error() == NULL, "no error initially");

    /* Null handle checks - transcriber */
    CHECK(parakeet_transcriber_to_gpu(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null transcriber to_gpu");
    CHECK(parakeet_transcriber_to_half(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null transcriber to_half");
    CHECK(parakeet_transcriber_transcribe_file(NULL, "x.wav", NULL) == NULL, "null transcriber transcribe_file");
    CHECK(parakeet_transcriber_transcribe_pcm(NULL, NULL, 0, NULL) == NULL, "null transcriber transcribe_pcm");

    /* Null handle checks - tdt transcriber */
    CHECK(parakeet_tdt_transcriber_to_gpu(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null tdt to_gpu");
    CHECK(parakeet_tdt_transcriber_to_half(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null tdt to_half");

    /* Null handle checks - streaming */
    CHECK(parakeet_streaming_transcriber_to_gpu(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null streaming to_gpu");
    CHECK(parakeet_streaming_transcriber_feed_f32(NULL, NULL, 0, NULL) == PARAKEET_ERROR_NULL_HANDLE, "null streaming feed_f32");
    CHECK(parakeet_streaming_transcriber_reset(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null streaming reset");
    CHECK(parakeet_streaming_transcriber_get_text(NULL) == NULL, "null streaming get_text");

    /* Null handle checks - nemotron */
    CHECK(parakeet_nemotron_transcriber_to_gpu(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null nemotron to_gpu");
    CHECK(parakeet_nemotron_transcriber_feed_i16(NULL, NULL, 0, NULL) == PARAKEET_ERROR_NULL_HANDLE, "null nemotron feed_i16");
    CHECK(parakeet_nemotron_transcriber_get_text(NULL) == NULL, "null nemotron get_text");

    /* Null handle checks - diarized */
    CHECK(parakeet_diarized_transcriber_to_gpu(NULL) == PARAKEET_ERROR_NULL_HANDLE, "null diarized to_gpu");

    /* Null handle checks - results */
    CHECK(parakeet_result_text(NULL) == NULL, "null result text");
    CHECK(parakeet_result_token_count(NULL) == 0, "null result token_count");
    CHECK(parakeet_result_token_ids(NULL) == NULL, "null result token_ids");
    CHECK(parakeet_result_word_count(NULL) == 0, "null result word_count");
    CHECK(parakeet_diarized_result_text(NULL) == NULL, "null diarized_result text");
    CHECK(parakeet_diarized_result_word_count(NULL) == 0, "null diarized_result word_count");
    CHECK(parakeet_diarized_result_segment_count(NULL) == 0, "null diarized_result segment_count");

    /* Null handle checks - audio */
    CHECK(parakeet_audio_sample_rate(NULL) == 0, "null audio sample_rate");
    CHECK(parakeet_audio_num_samples(NULL) == 0, "null audio num_samples");
    CHECK(parakeet_audio_duration(NULL) == 0.0f, "null audio duration");
    CHECK(parakeet_audio_format(NULL) == PARAKEET_AUDIO_FORMAT_UNKNOWN, "null audio format");
    size_t cnt = 99;
    CHECK(parakeet_audio_samples(NULL, &cnt) == NULL, "null audio samples");
    CHECK(cnt == 0, "null audio samples out_count zeroed");

    /* Create with null paths should fail */
    CHECK(parakeet_transcriber_create(NULL, "v.txt", NULL) == NULL, "null weights path");
    CHECK(parakeet_last_error() != NULL, "error set after null weights");
    CHECK(parakeet_tdt_transcriber_create("w.st", NULL, NULL) == NULL, "null vocab path");
    CHECK(parakeet_streaming_transcriber_create(NULL, NULL, NULL) == NULL, "null both paths streaming");
    CHECK(parakeet_diarized_transcriber_create(NULL, "sf.st", "v.txt", NULL, NULL) == NULL, "null asr weights diarized");

    /* Config type mismatch */
    parakeet_config_t cfg_eou = parakeet_config_eou_120m();
    CHECK(cfg_eou != NULL, "eou config created");
    CHECK(parakeet_transcriber_create("w.st", "v.txt", cfg_eou) == NULL, "config type mismatch for transcriber");
    CHECK(strstr(parakeet_last_error(), "mismatch") != NULL, "mismatch error message");
    parakeet_config_free(cfg_eou);

    /* Options create/set/free */
    parakeet_options_t opts = parakeet_options_create();
    CHECK(opts != NULL, "options created");
    CHECK(parakeet_options_set_decoder(opts, PARAKEET_DECODER_CTC) == PARAKEET_OK, "set decoder");
    CHECK(parakeet_options_set_timestamps(opts, 1) == PARAKEET_OK, "set timestamps");
    CHECK(parakeet_options_set_boost_score(opts, 3.0f) == PARAKEET_OK, "set boost score");
    CHECK(parakeet_options_add_boost_phrase(opts, "hello") == PARAKEET_OK, "add boost phrase");
    CHECK(parakeet_options_add_boost_phrase(opts, NULL) == PARAKEET_ERROR_INVALID_ARGUMENT, "null boost phrase");
    CHECK(parakeet_options_set_decoder(NULL, PARAKEET_DECODER_CTC) == PARAKEET_ERROR_NULL_HANDLE, "null opts set_decoder");
    parakeet_options_free(opts);

    /* Config create/free cycle (leak check) */
    parakeet_config_free(parakeet_config_110m());
    parakeet_config_free(parakeet_config_tdt_600m());
    parakeet_config_free(parakeet_config_eou_120m());
    parakeet_config_free(parakeet_config_nemotron_600m(0));
    parakeet_config_free(parakeet_config_sortformer_117m());

    /* Free NULL is safe */
    parakeet_config_free(NULL);
    parakeet_options_free(NULL);
    parakeet_transcriber_free(NULL);
    parakeet_tdt_transcriber_free(NULL);
    parakeet_streaming_transcriber_free(NULL);
    parakeet_nemotron_transcriber_free(NULL);
    parakeet_diarized_transcriber_free(NULL);
    parakeet_result_free(NULL);
    parakeet_diarized_result_free(NULL);
    parakeet_audio_free(NULL);

    /* Audio null data */
    CHECK(parakeet_audio_read_file(NULL, 16000) == NULL, "null path audio_read_file");
    CHECK(parakeet_audio_read_f32(NULL, 0, 16000, 16000) == NULL, "null pcm audio_read_f32");
    CHECK(parakeet_audio_read_i16(NULL, 0, 16000, 16000) == NULL, "null pcm audio_read_i16");
    CHECK(parakeet_audio_read_encoded(NULL, 0, 16000) == NULL, "null data audio_read_encoded");

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
