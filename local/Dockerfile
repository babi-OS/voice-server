# Same locked native dependencies as the released serving image.
FROM ghcr.io/babi-os/voice-server:0.0.2@sha256:fe9623ebd5a37dbaef6019c465fedda87804992dd141b23baa5653030f8e7a71
COPY --chown=10001:10001 voice_server /app/voice_server
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VOICE_ORT_SPINNING=0
