# Troly Voice Server

Stateless CPU speech service for Vietnamese-first assistants. It provides VAD, Gipformer STT, and Matcha/Vocos TTS over one WebSocket. It never calls an LLM, executes tools, stores history, or creates sessions.

## Run with Docker

Use the public AMD64 release image, keeping downloaded models in a named volume:

```sh
docker run -d --name voice-server --restart unless-stopped \
  -p 8765:8765 -v voice-models:/models \
  ghcr.io/babi-os/voice-server:0.0.1
docker logs -f voice-server
curl http://127.0.0.1:8765/healthz
```

The service becomes ready after the first model download and warmup. The image
runs as UID/GID 10001; bind-mounted model directories must be writable by that
user. Models are downloaded at runtime, not included in the image.

To build from source instead:

```sh
docker compose up --build
curl http://127.0.0.1:8765/healthz
```

The first start downloads the pinned public Hugging Face models into the `voice-models` named volume. Override revisions in `.env` when deliberately upgrading them.

## Container releases

Pushing a `vMAJOR.MINOR.PATCH` tag runs the host tests and publishes
`ghcr.io/babi-os/voice-server:MAJOR.MINOR.PATCH`, `:latest`, and
`:sha-<full-commit>`. Pull requests run the same tests and build without
publishing. Actions are pinned to commits and GHCR authentication uses the
workflow's `GITHUB_TOKEN`. OCI labels record the source, version and revision.
Container versions are independent of the Python package version.

For reproducible deployments, pin the published image digest instead of
`:latest`. New packages must be made public in GitHub package settings before
unauthenticated pulls work.

## Protocol

Connect to `/v2/voice`, send one `hello` containing independent `audio.input` and `audio.output` profiles, then use `stt.start`/Opus or PCM16 packets/`stt.finish`, or `tts.start`/ordered `tts.text`/`tts.finish`. Binary packets have an 8-byte big-endian `(stream_id, sequence)` header followed by one Opus packet or PCM16 payload. `request_id` and `turn_id` identify an operation only; there is no session identifier. The client owns chat history, LLM calls, tools, retries, and reconnection.

`ready`, `stt.final`, `tts.audio_start`, `tts.audio_end`, `request.end`, `request.cancelled`, and structured `error` events are JSON. State is discarded when the WebSocket closes.

### Speech endpointing

`stt.start` accepts `end_silence_ms`. The default is **1000 ms**, rounded up to
the next 32 ms VAD window (1024 ms). This preserves natural pauses such as the
686 ms pause in “Hôm nay ở Sài Gòn bao nhiêu độ, có nên mang ô không?” The old
512 ms endpoint cut that question off before the umbrella clause. Waiting for
automatic endpointing now costs about 512 ms more silence than the old default;
explicit `stt.finish` does not wait for that silence.

```json
{"type":"stt.start","request_id":"question-1","turn_id":"turn-1","end_silence_ms":1000}
```

Set `end_silence_ms` to **0** when the device owns VAD, the user releases a
push-to-talk button, or a file upload has a known end. Then send `stt.finish`
once the last audio packet has been sent. The command-line WAV client uses
this mode, so pauses within the recording cannot end the request early.
Other accepted values are integers from 256 through 3000 ms. The existing
30-second utterance limit also applies in client-controlled mode.

`stt.finish` preserves the final partial VAD frame and is harmless if automatic
endpointing has already started or completed transcription. Cancelled requests
drain their connection tasks before the connection admits replacement work.
The service remains stateless: `session.create` and `session.resume` are not
part of this contract. Clients send operations directly after `ready`.

## Local development

```sh
uv sync --extra dev
uv run voice-server
uv run python -m voice_server.client --text 'Xin chào' --output reply.wav
```

Run `uv run pytest -q` for bounded protocol, endpointing and cancellation tests.
`uv run python tests/check_endpointing_models.py` additionally downloads/loads
the real public VAD/STT models and replays the checked-in synthetic Vietnamese
fixture through a temporary loopback WebSocket server. It exercises PCM16 and
Opus, reproduces the old cutoff with an explicit 512 ms threshold, and checks
the default and client-controlled modes retain both clauses. No physical
microphone/speaker or production service is used. Results go to `.local/`.

## CPU workers and threads

The service uses separate bounded VAD, STT and TTS pools. STT/TTS can run in
multiple model processes; the next idle process takes the next queued call.
VAD stays in one process so each connection's recurrent state stays together.
Every STT/TTS process loads its model and executes a small inference before the
service becomes ready. Model weights are shared on disk, while runtime model
allocations are per process.

These settings are read at startup:

| Setting | Default | Meaning |
|---|---:|---|
| `VOICE_ASR_WORKERS` | 1 | STT model processes, maximum 8 |
| `VOICE_TTS_WORKERS` | 1 | TTS model processes, maximum 4 |
| `VOICE_ASR_THREADS` | 2 | sherpa inference threads per STT process |
| `VOICE_TTS_THREADS` | 2 | ONNX intra-op threads per Matcha session |
| `VOICE_ORT_SPINNING` | 1 | VAD/Matcha ONNX thread spinning: 0 disables, 1 enables |

STT still admits at most 8 outstanding calls and TTS at most 4, including
running calls. Increasing workers does not increase those admission limits.
Cancellation of native inference retains its worker and admission slot until
the native call finishes; queued work cancelled before execution frees its
slot immediately. The native work itself is not interrupted.

Choose worker/thread combinations against the container CPU quota, then measure
latency and memory under concurrent speech. More workers duplicate model RAM;
more inference threads can compete for the same CPU quota. Kubernetes CPU and
memory requests/limits are set by the deployment, independently of these flags.
The container's `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` settings also apply
to native libraries. Changing these settings requires restarting the service.

The default process/thread counts preserve the previous one-process layout.
The protocol, pinned model selection, 20ms packet contract, and connection-local
state model are unchanged. These options do not add batching or an autoscaler.

## Capacity metrics and shutdown

`GET /metrics` exposes aggregate Prometheus gauges for connected clients and
admitted/running/queued VAD, ASR and TTS work, plus admission-rejection counters.
It contains no audio, transcripts, request IDs or persistent user state. Scrape it
through an internal Service; public gateways can restrict routes to the browser,
health check and voice WebSocket. CPU/memory measurements should include the
whole container, since native inference runs in child processes.

`GET /readyz` returns 503 while draining; `/healthz` remains a liveness check.
SIGTERM/SIGINT stops new WebSocket and speech-operation admission, waits up to
120 seconds for active speech operations, and closes connections with WebSocket
code 1012 so clients can reconnect through the Service. Use a termination grace
period of at least 150 seconds. Model processes remain available until draining
finishes. Existing WebSockets stay on their original replica when scaling up;
new replicas receive new connections. Benchmark that routing behavior separately
from raw model throughput when choosing autoscaling thresholds.
