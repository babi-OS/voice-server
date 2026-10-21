# Local voice development and capacity validation

Everything here targets `k3d-voice-local` using `.local/kubeconfig`. It does not
load production credentials or change the default Kubernetes context. Run from
this repository, with Docker, k3d 5.8.3, Helm and kubectl installed:

```sh
scripts/voice-local create
scripts/voice-local build
scripts/voice-local import
scripts/voice-local deploy
scripts/voice-local monitoring
scripts/voice-local test
scripts/voice-local report
```

The create command requires **26 GiB available RAM and 24 logical CPUs**, creates one server and two
agents with K3s `v1.33.6-k3s1`, and applies Docker limits of 1 CPU/2 GiB to the
server and 6 CPU/10 GiB to each agent. CPU sets match the measured setup:
server 18, agents 6–11 and 12–17. Matched containers use 0–5; run clients on
19–22 with taskset when reproducing the recorded comparison. This restricts
our processes, not unrelated background workloads. Server scheduling is tainted. Metrics-server
comes with that pinned K3s image. Prometheus Operator chart **92.10.1** and
Prometheus adapter chart **5.3.0** match repository production pins. Alertmanager,
Grafana, default alert rules and node exporter are disabled. There is no external
notification integration.

`http://127.0.0.1:18765` exposes the gateway only. Worker services remain internal.
The default deploy has one replica per role and no HPA. Use disposable local RPC
credentials generated in `.local/rpc-token`. This file and kubeconfig are mode
0600 and ignored. EndpointSlice RBAC is namespace scoped; workers have no token.

Model preparation uses **only role-specific serving files** copied from the
existing `voice-server_voice-models` volume mounted read-only into both agents.
Copies live in disposable disk emptyDirs. Pinned VAD/ASR SHA-256 values and Matcha
metadata checksums are validated. Offline native model loading and inference
warm-up precede readiness. The original model volume is never modified/deleted.
Set `VOICE_LOCAL_MODEL_MODE=cold` when deploying to download into new emptyDirs
instead. Init logs separate seed-copy/download preparation from application
warm-up; a rollout with fresh emptyDirs must not be reported as an in-place warm
restart. For fresh Pods with changed code, build/import and run an explicit local
rollout restart; importing the same image tag alone does not restart existing Pods.

## Benchmarks

All fixtures are synthetic. They verify the expected Vietnamese transcript,
packet order/count, decoded audio duration and non-silent Opus output. No LLM is
involved. Run `.venv/bin/python scripts/bench_local.py --help` for bounded inputs.

```sh
.venv/bin/python scripts/bench_matched.py --cpuset 0-5 --users 8 --turns 40
.venv/bin/python scripts/bench_local.py --users 32 --turns 40 \
  --output .local/results/k3d32
.venv/bin/python scripts/bench_local.py --users 32 --turns 3 --mode burst \
  --output .local/results/burst32
```

The matched command runs monolith and split **sequentially**, each inside ONE
Docker cgroup limited to 4 CPU/6 GiB with swap disabled, and with two ASR/two TTS
native workers at two threads each. The host load generator is outside the
container limit. It refuses an existing named matched container, saves service
logs, and removes only containers it creates. Hold other host load stable and
record CPU topology/interference. Shared quotas do not make different machines
performance equivalent. k3d integration is a separate, larger resource budget;
never attribute extra compute to the refactor alone.

## Autoscaling and failure testing

HPA is opt-in **only in this local namespace**:

```sh
python3 local/hpa.py | kubectl --kubeconfig .local/kubeconfig \
  --context k3d-voice-local apply -f -
```

Both workload and absolute CPU metrics are required. Local min/max is 1/2 per
role. Production min/max remains a later capacity-gated 2/4 gateway, 2/4 STT,
2/8 TTS configuration; no production HPA is enabled here. HPA scale-down uses
300 seconds stabilization and one Pod per 60 seconds. Record desired/current/
Ready replicas, metrics, per-worker inference counts, queue deadlines, cold
starts and explicit overloads. Existing WebSockets remain on their gateway;
backend inference routing changes independently as workers become Ready.

After tests, `scripts/voice-local destroy` deletes only `voice-local` and its
local kubeconfig. It preserves the model seed and result files. The standalone
matched command cleans its own container; an interrupted run can leave the
explicitly named `voice-local-matched` container for diagnosis. Do not use global
Docker/Kubernetes cleanup commands.

Additional local validation commands (run after fixed-capacity benchmarks):

```sh
.venv/bin/python local/failures.py
.venv/bin/python local/drain_check.py
.venv/bin/python local/native_failure.py
.venv/bin/python local/scale_check.py
```

`failures.py` requires two ready replicas per role and tests worker replacement,
a synthetically unschedulable canary, and temporary loss/recovery of one named
k3d agent. Its surviving-gateway probe uses a localhost port-forward; it does not
prove external load-balancer failover. `drain_check.py` scales TTS from two to one
while both real native jobs are running, requires both responses to finish, then
restores two replicas. `native_failure.py` kills the model process only inside a
disposable local Pod and verifies readiness/liveness failure and kubelet restart.
`scale_check.py` exercises both real native stages, captures HPA state through
scale-up and the 300-second idle scale-down window, then removes local HPAs.
Its per-service port-forwards intentionally pin stage probes to one worker;
EndpointSlice distribution is measured separately through gateway traffic.

k3d nodes report the host's CPU count (24) even with a Docker quota of 6 cores.
Use the **Docker quotas**, not that advertised count, when assessing local
capacity. This setup does not establish physical host redundancy. CPU/RAM limits
are ceilings; the 22-GiB cluster limit is not its actual working set.

Measured long-input working sets required raising the initial requests to
**896Mi for STT and 3712Mi for TTS** (gateway stays 256Mi), using 1.25x peak sampled
working set rounded up to 128Mi. Final limits are 2Gi STT / 4Gi TTS / 1Gi gateway. The maximum 1500-character
TTS probe reached about 2875Mi working set; its request/limit were raised after
that probe. Historical short-fixture Kubernetes results used the original 3Gi
TTS limit. The matched comparison always retains a shared 4 CPU/6 GiB limit.
These larger requests must be included in a future production reserve check.

`local/maximum_input.py` checks the real 30-second STT / 1500-character TTS
RPC bounds and rejects oversized jobs. The local setup is not a proof that
all maximum-length jobs at all replica maxima fit on one surviving agent.

`local/queue_deadline.py` uses a disposable fresh TTS Pod to measure actual
queue expiration behind a long native job and immediate rejection at saturation.

### Larger queues and the CPU capacity limit

The [queue/STT CPU study](../docs/benchmarks/2026-10-21-queue-capacity/README.md)
uses a single Docker cgroup for every serving process, with 4 CPU/6 GiB and the
client outside. It can push through 256 users while preserving the two-second
stage queue deadline. It does not alter k3d or enable an HPA:

```sh
scripts/voice-local build
docker tag voice-server:local voice-server:queue-study
.venv/bin/python scripts/study_queue_capacity.py --label my-capacity-run --admission 16 --asr-threads 1 --users 32 64 80 96 128 192 --turns 4
.venv/bin/python scripts/summarize_capacity.py
```

Run one study at a time. Labels must be new directory names; results are kept
under `.local/queue-capacity/`. The named study container is removed on exit;
the existing model volume is read-only and preserved. This PC's CPU placement is
fixed in the harness (service 0–5; client 19–20). `--tts-workers 4 --tts-threads 1`
adds the alternate TTS layout without changing aggregate CPU/memory limits.

The [final optimization pass](../docs/benchmarks/2026-10-21-final-optimization/README.md)
uses `--image voice-server:final-candidate --stt-workers 2 --tts-workers 4
--asr-threads 1 --tts-threads 1 --admission 16`. These worker counts are separate
role processes sharing one aggregate quota, not native processes inside each
split worker. For a ten-minute confirmation use `--turns 40`; the harness stops
such future confirmations on the first failed turn (a failed partial run never
passes). Successful runs must complete every scheduled turn. Old fixed-duration
profiles remain recorded with their original configurations.
