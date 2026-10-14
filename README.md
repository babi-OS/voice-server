# Troly Voice Server

Stateless CPU speech service for Vietnamese-first assistants. It provides VAD, Gipformer STT, and Matcha/Vocos TTS over one WebSocket. It never calls an LLM, executes tools, stores history, or creates sessions.

## Run with Docker

```sh
docker compose up --build
curl http://127.0.0.1:8765/healthz
```

The first start downloads the pinned public Hugging Face models into the `voice-models` named volume. Override revisions in `.env` when deliberately upgrading them.

## Protocol

Connect to `/v2/voice`, send one `hello` containing independent `audio.input` and `audio.output` profiles, then use `stt.start`/Opus or PCM16 packets/`stt.finish`, or `tts.start`/ordered `tts.text`/`tts.finish`. Binary packets have an 8-byte big-endian `(stream_id, sequence)` header followed by one Opus packet or PCM16 payload. `request_id` and `turn_id` identify an operation only; there is no session identifier. The client owns chat history, LLM calls, tools, retries, and reconnection.

`ready`, `stt.final`, `tts.audio_start`, `tts.audio_end`, `request.end`, `request.cancelled`, and structured `error` events are JSON. State is discarded when the WebSocket closes.

## Local development

```sh
uv sync --extra dev
uv run voice-server
uv run python -m voice_server.client --text 'Xin chào' --output reply.wav
```
