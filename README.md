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
