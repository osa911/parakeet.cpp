# Server

`example-server` keeps a Parakeet transcriber warm inside one process and
accepts newline-delimited JSON requests over a Unix domain socket. It is an
example for consumers who want persistent-process reuse without changing
parakeet.cpp core code.

## Build

This example is opt-in:

```bash
make build SERVER=ON
# Binary: ./build/examples/server/example-server
```

You can also configure CMake directly:

```bash
cmake -B build -DPARAKEET_BUILD_SERVER_EXAMPLE=ON
cmake --build build
```

## Usage

```bash
./build/examples/server/example-server /tmp/parakeet.sock \
  model.safetensors vocab.txt [options]
```

Server startup options:

- `--model TYPE` — `tdt-ctc-110m` (default) or `tdt-600m`
- `--gpu` — move the model to Metal GPU once at startup
- `--fp16` — cast to fp16 before `--gpu`
- `--vad PATH` — load Silero VAD weights once at startup so requests can opt in

This example keeps one loaded model instance warm per process and supports:

- `tdt-ctc-110m` through the high-level `parakeet::Transcriber` API
- `tdt-600m` through the same reusable loaded-state pattern used by the CLI's
  explicit TDT path

## Request protocol

Each request is one JSON object per line:

```json
{"request_id":"1","audio_path":"samples/mm1.wav","decoder":"tdt","timestamps":true}
```

Supported request keys:

- `request_id` — optional string echoed back in the response
- `audio_path` — required path to an audio file readable by `read_audio`
- `decoder` — optional: `tdt`, `ctc`, `tdt-beam`, `ctc-beam`
- `timestamps` — optional boolean
- `use_vad` — optional boolean (requires `--vad PATH` at server startup)
- `beam_width` — optional integer
- `lm_path` — optional ARPA LM path for beam decoders
- `lm_weight` — optional float
- `boost_score` — optional float
- `boost_phrases` — optional array of strings

Responses are also newline-delimited JSON:

```json
{"ok":true,"request_id":"1","text":"...","elapsed_ms":812}
```

With timestamps enabled, the response also includes `word_timestamps`:

```json
{"ok":true,"request_id":"1","text":"...","elapsed_ms":812,"word_timestamps":[{"word":"hello","start":0.0,"end":0.4,"confidence":0.98}]}
```

Errors stay on the socket as JSON and operational logs stay on stderr:

```json
{"ok":false,"request_id":"1","error":"audio_path is required"}
```

## Example session

Start the server:

```bash
./build/examples/server/example-server /tmp/parakeet.sock model.safetensors vocab.txt --model tdt-600m
```

Send a request:

```bash
printf '%s\n' '{"request_id":"demo","audio_path":"samples/mm1.wav","timestamps":true}' \
  | nc -U /tmp/parakeet.sock
```

This is intentionally example-grade:

- one warm model per server process
- line-delimited request/response framing
- no auth, TLS, or concurrency guarantees
- meant to be wrapped or adapted by downstream applications
