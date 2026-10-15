# 5090 deployment and public speech latency

15 October 2026. Deployed application commit `f8609cc` to the existing `voice` Kubernetes Deployment on the 5090 host. The obsolete Docker Compose container was stopped before this change; the Kubernetes service was the live backend.

The new image is `localhost/voice-server:sha-f8609cc`, imported into the node runtime. The rollout completed with one ready pod and zero restarts. Running server source SHA256 matches the tested checkout: `6211714ba2d3d49e3a848529af081caa95fd7c255d7f5671bd081f85195af32f`. Existing service routing and model PVC were retained. The preceding image `localhost/voice-server:sha-c61b8c365c70` remains available as a rollback target.

## Three public-endpoint rounds

Same 4.85-second synthesized Vietnamese recording as before: “Hôm nay ở Sài Gòn bao nhiêu độ, có nên mang ô không?” Same Vietnamese TTS text as the preceding comparison. Host uses verified TLS to `voice.troly.ai.vn`, a persistent negotiated WSS connection, and app-profile 16 kHz/20 ms/24 kbps Opus input. Gemini receives the identical completed source WAV through Key4U. Provider order alternates by round. No LLM answer generation or app tool calls are included.

| Stage | Median seconds | Observed range | Completed |
|---|---:|---:|---:|
| Troly STT, automatic endpointing | 0.99 | 0.98–1.07 | 3/3 |
| Troly STT, client-controlled finish | 0.09 | 0.08–0.10 | 3/3 |
| Troly TTS, first decodable audio | 0.55 | 0.55–0.76 | 3/3 |
| Key4U Gemini 3.5 Flash-Lite STT | 6.56 | 6.54–7.12 | 3/3 |
| Key4U Gemini 3.8 Flash STT | 3.40 | 3.15–4.06 | 3/3 |
| Key4U Gemini 3.8 Flash TTS | No valid first-audio measurement | Three approximately 35 s timeouts | 0/3 |

Both Troly endpointing modes retained the full umbrella clause in every round. This confirms the premature-endpoint fix at the public endpoint, rather than counting a truncated transcript as fast recognition. All Gemini STT requests also returned the full question.

STT timing starts at the end of the supplied recording and ends at stt.final or the full Gemini transcription response. Client-controlled mode sends stt.finish after the last packet; any extra device VAD decision delay is excluded. Automatic mode includes server silence detection. Gemini STT requests are transcription-only, with low thinking for Flash and minimal for Flash-Lite; internal model inference cannot be decomposed into an independent reasoning time.

TTS timing starts when the known complete text is submitted and ends at the first successfully decoded Opus frame for Troly. TTS admission takes an additional approximately 0.05 seconds. Device playback buffering, acoustic output and full synthesized playout duration are excluded. Voices and wire formats differ.

A sum of the measured medians gives about 0.69 seconds for client-finish STT + TTS admission + first audio, or about 1.58 seconds with automatic endpointing. These are stage-budget estimates, not measured complete user turns; add device VAD wait, actual LLM work and playback buffering as applicable. No complete Gemini speech budget can be established while TTS times out.

## Availability and remaining boundaries

The first pre-test HTTPS health request encountered TLSV1_ALERT_INTERNAL_ERROR. Subsequent verified HTTPS health and WSS connections succeeded; TLS verification was never disabled. Speech rounds themselves completed without Troly request failure. Automatic endpointing produced expected late_packet diagnostics for in-flight packets after speech_end; client-controlled mode produced none. The service stayed ready with zero restarts during the measurement window. This is not a long-duration reliability gate.

Samples come from one host/network route and one clean synthetic voice. These are small observed distributions, not p95 or physical device acceptance. The longer automatic silence default intentionally costs more endpointing time than the previous 512 ms cutoff; device-owned VAD plus explicit finish avoids that wait. Actual device VAD and physical listening/UI remain unqualified here.

Raw stage observations, deployment identity, hashes and initial setup failure are in [deployed-latency-20261015.json](deployed-latency-20261015.json). Private deployment/probe scripts and full provider responses remain in `.local/deploy-20261015/`.
