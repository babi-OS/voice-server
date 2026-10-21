# Bounded gateway ingress and single-pass deployment

The gateway reads WebSocket messages independently of codec/VAD processing.
A connection consumes messages in their original order, with at most 32 waiting
messages and 1,024 waiting messages across the gateway. A shared 25-ms watchdog
rejects a message that has spent more than 500 ms queued or processing. VAD has
256 bounded pending operations; cancelled queued feeds are dropped, while reset
and release commands still run. The watchdog is part of liveness/readiness.

An overloaded stream receives `error` with `code: overloaded`, `stage: ingress`
and `retry_after_ms: 1000`, followed by WebSocket close 1013. The client must
reconnect and begin a new operation. Audio is not replayed. Reporting does not
wait for native cancellation; codec and model occupancy remains held until the
active native operation finishes. Unexpected processing faults use
`internal_error`/close 1011 instead of being classified as capacity rejection.

Server receipt time of `stt.finish` and each `tts.text` follows the message into
inference admission. Time already spent at ingress/chunk scheduling is deducted
from its two-second queue budget. Gateway admission reserves 150 ms for RPC and
reporting, so worker queues receive at most 1.85 seconds. Explicit pre-start
retries share this deadline; uncertain native operations are never replayed.
Inference time is separate from queue wait. Network transit and an unresponsive
client cannot be guaranteed by a server-side deadline.

Limits: `VOICE_INGRESS_ITEMS` (32, max128), `VOICE_INGRESS_GLOBAL_ITEMS` (1024,
max8192), `VOICE_VAD_QUEUE` (256, max1024),
`VOICE_GATEWAY_MAX_CONNECTIONS` (256, max1024), existing
`VOICE_GATEWAY_MAX_MICROPHONES` (default32, max256). Benchmark/deployment profiles
set connection cap512 and microphone cap256 explicitly. These finite capacities
are independent of STT/TTS worker admission, whose default/selected layout is
unchanged. Autoscaling is not enabled by this change.

`/metrics` reports ingress queue/overload-response histograms, pending messages
and fixed-label rejection reasons. The overload histogram starts at receipt of
the oldest offending message. A duplex benchmark client also records overload
received while microphone audio is still being sent, and connection admission
failures count as failed turns rather than aborting the harness.

Build the immutable model image from the selected export, without training
checkpoints, credentials or model-hub access in the serving container:

```sh
.venv/bin/python scripts/build_single_pass.py --model-dir /path/to/exports/single --tag voice-server:single-pass-ingress-release
.venv/bin/python scripts/study_gateway_ingress.py --image voice-server:single-pass-ingress-release --admission 16 --asr-threads 1 --tts-threads 1 --stt-workers 2 --tts-workers 4 --users 208 --turns 40 --label fresh-ingress-confirm
.venv/bin/python scripts/study_gateway_ingress.py --image voice-server:single-pass-ingress-release --admission 16 --asr-threads 1 --tts-threads 1 --stt-workers 2 --tts-workers 4 --users 256 288 320 --turns 2 --label fresh-ingress-stress
```

The harness is local Docker, one aggregate 4 CPU/6 GiB cgroup, swap disabled,
CPUs0-5; the external driver uses CPUs19-20. Adjust these affinity settings on
other hardware. Every case preserves configuration, request records, per-stage
metrics and resource samples. Ten-minute capacity requires all planned turns,
zero failures and p95 speech-end to audio <=5 seconds. Stress levels with
controlled failures do not qualify as supported user capacity.
