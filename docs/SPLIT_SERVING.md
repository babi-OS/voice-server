# Split voice serving contract

`VOICE_ROLE=all` remains the monolithic default. `gateway` loads Silero VAD,
owns the public v2 WebSocket and dispatches independent stage work. `stt` loads
only Gipformer; `tts` loads only Matcha. Split workers require one native process
per Pod (`VOICE_ASR_WORKERS=1`/`VOICE_TTS_WORKERS=1`), use two inference threads by
default, and expose no public WebSocket/static pages. Deployments set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VOICE_ORT_SPINNING=0`.

## Internal endpoints

All inference and `/capacity` calls require `Authorization: Bearer <RPC secret>`.
`POST /internal/v1/stt` takes mono 16-kHz little-endian PCM16, 1..960000 bytes
with even length, and returns JSON `{text}`. `POST /internal/v1/tts` takes
`{text,sample_rate}`, non-empty text <=1500 characters and an existing supported
rate (8000/12000/16000/24000/48000). It returns `application/octet-stream`
PCM16 with `X-Audio-Format: pcm_s16le` and `X-Audio-Sample-Rate`. The gateway
requests native-rate PCM and handles client-rate conversion/Opus off the loop.

`X-Queue-Budget-Ms` is an integer clamped to 0..2000, default 2000, relative to
worker admission. Success timing headers are `X-Queue-Wait-Ms` and
`X-Inference-Ms`. Native operation timeout ceilings are 90s STT / 120s TTS. Each
worker defaults to one running and one queued job. `VOICE_STT_ADMISSION` and
`VOICE_TTS_ADMISSION` configure the total admitted jobs per split worker (1–256);
raising this bound never raises the two-second actual queue deadline. Admission
forecasts include all queued jobs plus estimated remaining running work. EWMA service estimates (alpha .2)
use length buckets initialized from each role’s native warm-up calibration
(one-second fallback before warm-up). Actual queue
deadlines override estimates. A future TTS slot is not reserved during STT or an
external LLM call.

HTTP 429 means overload before inference; HTTP 503 means draining/unavailable.
Both include `{code,stage,retry_after_ms:1000,not_started:true}`. The gateway maps
these to request-end failures with `code:overloaded`, stage and retry hint. An
inference timeout is HTTP 504 with `not_started:false`. Accepted/ambiguous calls
are never replayed. Only explicit pre-start rejection or connection establishment
failure can use the second and final attempt, within the original queue budget.

Gateways watch namespace-scoped Ready/non-terminating EndpointSlices, relist
periodically/after watch expiration, and poll capacity every 250ms plus jitter.
Capacity reports expire after 1s. Selection scores every Ready worker using
reported work plus conservative outstanding dispatch reservations; equal scores
use randomized ties to spread independent gateway decisions. Discovery stale for 30s fails
new operations closed; failed endpoints have a 1s cooldown. A single long-lived
Service-IP HTTP connection is not the gateway's balancing strategy. The
namespace-scoped ServiceAccount can only read EndpointSlices; workers do not
mount Kubernetes credentials. A dedicated RPC Secret is separate from client
access. NetworkPolicy alone is not claimed as isolation on unenforced Flannel.

## Admission, cancellation and drain

Gateways permit 256 WebSockets and default to 32 active microphones
(`VOICE_GATEWAY_MAX_MICROPHONES`, maximum 256), with 64 VAD operations
queued at most. VAD uses one thread, <=16 operations/handoff and <=1ms collection
delay, preserving the exact original stateful per-window function. Codec work
uses two threads plus eight executor-waiting tasks; per-connection codec calls
remain sequential even when a caller cancels. A codec job encodes at most eight
output packets; each is still sent as one negotiated wire packet. Identity
16-kHz input resampling skips the executor. WebSocket flow control supplies
backpressure; no silent frame dropping is introduced.

Queued native cancellation frees admission. Active native cancellation stops
response delivery but retains occupancy until the native job completes. A dead
native process makes readiness and liveness fail so Kubernetes restarts the
container. Busy workers stay Ready. During drain readiness fails, new work is
rejected, admitted native work finishes for <=120s, and gateway sockets close
with 1012. Pod termination grace is 150s. No connection migration, persisted
history, sessions, durable queues or automatic operation replay is introduced.

## Metrics and release boundary

Production telemetry uses bounded fixed-label histograms for queue, inference,
native CPU, RPC, codec and loop delay, plus admitted/running/queued/rejected/
offered counts and active microphones. Native inference CPU excludes model
thread activity between calls; container/process CPU is also measured and must
be included in efficiency comparisons. There are no production benchmark
endpoints or per-request timing arrays.

See [local testing](../local/README.md). The production candidate renderer emits
Deployments/Services/RBAC/ServiceMonitor with two minimum replicas per role and
an existing Secret reference, but no HPA or public route. It requires an image
digest and **does not apply anything**. Production scaling and cutover still
require the one-node-failure reserve, surge/old-service capacity, monitoring and
a later rollout request.

The [larger-queue and STT CPU study](benchmarks/2026-10-21-queue-capacity/README.md)
profiles CPU after inference returns as well as inside each call. Sherpa creates
its own native ONNX sessions; `VOICE_ORT_SPINNING` configures only the Python
VAD/TTS sessions. Use the measured `VOICE_ASR_THREADS=1` candidate for local
multi-user comparisons; compatibility defaults and production manifests remain
unchanged. Rejection metrics distinguish `full`, `predicted_wait`, and `deadline`;
expired waits are recorded separately from waits that reach native inference.

See the [final optimization report](benchmarks/2026-10-21-final-optimization/README.md)
for the worker-layout comparison, TTS stage profile, final CPU-limited capacity
and measured routing/codec changes.

## Resource-bounded single-pod stack

For a fixed aggregate budget, run `python -m voice_server.stack`. This supervises
one gateway, two STT workers and four TTS workers in the **same container cgroup**.
Each worker has one native process and one inference thread; defaults are
admission 16 and the unchanged two-second actual queue deadline. Set
`VOICE_STACK_STT_WORKERS` / `VOICE_STACK_TTS_WORKERS` (1..4) to select the layout.
Workers bind loopback only, share a random in-memory RPC credential, and require
no Kubernetes API token. Explicit internal URLs take precedence over Kubernetes
injected discovery environment variables. The gateway starts after worker
readiness. On shutdown it drains first; any role process failure restarts the
container. Allow at least 150 seconds of termination grace.

This profile preserves the measured 4 CPU/6 GiB aggregate budget. It does not
provide independent Kubernetes scaling; the separate-role manifests remain the
path for that later deployment. Keep autoscaling disabled until that change is
explicitly selected and capacity is reserved.

## Built-in deployment benchmark

Run an explicit benchmark inside the running container:

```sh
/app/.venv/bin/python -m voice_server.benchmark \
  --users 8 --turns 4 --output /tmp/voice-benchmark-8
```

The packaged fixture is the same synthetic Vietnamese weather utterance used by
the external benchmark. It validates the transcript and decodes non-silent Opus
output, uses staggered 15-second turns, and writes `config.json`,
`requests.jsonl`, and `summary.json`. Reports include model revisions, CPU model,
actual cgroup limits/CPU/memory counters, per-stage Prometheus snapshots,
scheduling lag and speech-end-to-first-audio latency. The command connects only
to loopback. Its driver **shares the serving cgroup**, so its CPU is reported
separately; use the external load generator for matched-budget capacity tests.
Only one built-in benchmark can run at a time. Explicit bounds are 32 users,
40 turns and at most 600 seconds of scheduled traffic. There is no public
benchmark endpoint or automatic benchmark at startup. A one-user run is useful
for comparing hardware/model cost before increasing load.
