# Speech endpointing fix

15 October 2026. The server's 16 silent 32 ms VAD windows ended speech after
512 ms, cutting off the second clause of a Vietnamese question with a 686 ms
pause. This is endpointing loss, not an ASR failure on the complete recording.

`stt.start.end_silence_ms` now defaults to 1000 ms, rounded up to 1024 ms.
It accepts 0 for explicit client finish, or integers from 256 through 3000 ms.
The 30-second utterance cap applies to every mode. Automatic endpointing waits
about 512 ms longer than before; explicit finish bypasses that wait. The WAV
client uses explicit finish, and the final partial VAD frame is retained.

A finish racing with automatic endpointing is idempotent. Cancel and disconnect
await connection-task destruction before reusing request state or releasing
VAD state. This does not claim to interrupt an ONNX inference already running
inside a worker process. Failed VAD admission rolls back the active request.

The protocol intentionally remains stateless. `session.create` was not added;
clients send speech operations directly after `ready`.

## Validation

- 23 host tests passed: option validation, natural pauses, client finish,
  frame retention, utterance limits, admission rollback, cancellation and close.
- Real public Silero/Gipformer models replayed the fixture through a loopback
  WebSocket with both PCM16 and 24 kbps Opus input. Six runs covered the old
  threshold, the new default and explicit finish for each codec.

| Input | Old 512 ms threshold | Default | Client-controlled finish |
|---|---|---|---|
| PCM16 | Lost umbrella clause | Full question; 1.06 s after clip | Full question; 0.15 s after clip |
| Opus | Lost umbrella clause | Full question; 1.07 s after clip | Full question; 0.15 s after clip |

[Raw results and fixture identity](endpointing-results.json). Timing is a single
sample per codec/policy on this host, not deployed-server performance or p95.
TTS synthesis/encoding was not modified; its cancellation lifecycle is covered
with controlled workers. Production deployment and physical UI/audio acceptance
are separate from these checks.

Reproduce:

```sh
uv sync --extra dev
uv run pytest -q
HF_HOME="$PWD/.local/huggingface" uv run python tests/check_endpointing_models.py
```

The real-model check downloads public model files on first use and writes its
results to `.local/real-model-results.json`.

The fix was subsequently deployed to 5090 and verified through the public endpoint.
See [deployment and latency results](DEPLOYMENT_20261015.md).
