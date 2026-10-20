"""Prepare serving files once, before starting offline inference workers.

An optional Kubernetes init-container entrypoint. A complete retained cache is
validated locally and needs no model-hub request on subsequent scale-ups.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from .workers import MATCHA_REPO, MATCHA_REVISION, _asr_snapshot, _find, _tokens


def prepare(*, local):
    vad = Path(hf_hub_download('onnx-community/silero-vad', 'onnx/model.onnx',
               cache_dir=os.getenv('HF_HOME', '/models/huggingface'), local_files_only=local))
    asr = _asr_snapshot(local_files_only=local)
    files = [vad, _tokens(asr), *(_find(asr, role) for role in ('encoder', 'decoder', 'joiner'))]
    root = Path(snapshot_download(repo_id=MATCHA_REPO, revision=MATCHA_REVISION,
                allow_patterns=['weights/cpu-onnx/*', 'scripts/matcha_hybrid_frontend.py'],
                local_files_only=local)) / 'weights/cpu-onnx'
    metadata = json.loads((root / 'model.json').read_text())
    files += [root / 'model.json', root / 'phoneme_vocab.json']
    files += [root / item['name'] for item in metadata['files'].values()]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError('serving model cache is incomplete')
    for name, expected in [('phoneme_vocab.json', metadata['vocab_sha256']),
                           *((item['name'], item['sha256']) for item in metadata['files'].values())]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('serving model checksum mismatch: ' + name)
    return sum(path.stat().st_size for path in set(files))


def main():
    start = time.monotonic()
    downloaded = False
    try:
        size = prepare(local=True)
    except (LocalEntryNotFoundError, FileNotFoundError):
        downloaded = True
        size = prepare(local=False)
    print(json.dumps({'models_ready': True, 'downloaded': downloaded,
                      'serving_bytes': size, 'duration_s': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
