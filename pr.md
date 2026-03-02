## Summary

Adds Silero VAD v5 as a native preprocessing step to strip silence before ASR encoding. The model runs entirely in axiom (no ONNX Runtime dependency), consistent with all other models in parakeet.cpp.

- **Silero VAD v5 model** — native reimplementation (309K params, 14 tensors). STFT → 4-layer Conv encoder → LSTM decoder → sigmoid. Produces exact numerical match with the Python reference (identical probabilities and speech segments).
- **VAD preprocessing pipeline** — `SileroVAD::detect()` implements the full `get_speech_timestamps` algorithm: per-window probabilities → hysteresis thresholding → segment merging with min/max duration constraints → speech padding.
- **Timestamp remapping** — `TimestampRemapper` maps timestamps from compressed (silence-removed) audio back to the original timeline using piecewise linear interpolation via binary search.
- **Integrated into all transcribers** — `enable_vad()` on `Transcriber`, `TDTTranscriber`, and `DiarizedTranscriber`. Transcription text is identical; timestamps refer to the original audio.
- **C API** — standalone `parakeet_vad_t` for detection, plus `parakeet_*_enable_vad()` on all transcriber types.
- **CLI** — `--vad PATH` and `--vad-threshold F` flags. Prints VAD stats (segment count, speech duration, % filtered).
- **Weight converter** — `scripts/convert_silero_vad.py` downloads from torch.hub and exports to safetensors (~1.2MB).

### Performance (Apple M3, tdt-ctc-110m, 24.6s audio)

| Mode | Encoder | Total |
|------|---------|-------|
| CPU baseline | 7,537ms | 7.8s |
| CPU + VAD (24% filtered) | 5,319ms | 5.9s |
| GPU baseline | 1ms | 3.1s |

VAD processing: **~120ms on CPU** for 7.4s audio (0.016x RTF). CPU is preferred over GPU for VAD due to the small sequential model (kernel launch overhead dominates).

## Test plan

- [x] Numerical match: per-window probabilities and detected segments match Python Silero VAD reference exactly
- [x] CLI: `--vad` flag works with tdt-ctc-110m and tdt-600m, with and without `--timestamps`, `--gpu`
- [x] Timestamp remapping: word timestamps refer to original audio positions after VAD filtering
- [x] Transcription quality: text output matches with and without VAD on test audio
- [x] Full build: `cmake --build` succeeds with no errors or warnings (beyond pre-existing ones)
