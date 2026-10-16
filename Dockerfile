FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/huggingface OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libopus0 libsndfile1 espeak-ng && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY voice_server ./voice_server
RUN uv sync --no-dev --frozen
RUN groupadd --gid 10001 voice-server && useradd --create-home --uid 10001 --gid voice-server voice && mkdir -p /models && chown -R voice:voice-server /app /models
USER voice
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz')"
CMD ["/app/.venv/bin/python","-m","voice_server","--host","0.0.0.0","--port","8765"]
