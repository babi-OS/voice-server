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
    role=os.getenv('VOICE_ROLE','all')
    if role not in ('all','gateway','stt','tts'): raise ValueError('invalid VOICE_ROLE')
    files=[]; root=None; metadata=None
    if role in ('all','gateway'):
        files.append(Path(hf_hub_download('onnx-community/silero-vad','onnx/model.onnx',
            revision=os.getenv('VAD_REVISION','main'),cache_dir=os.getenv('HF_HOME','/models/huggingface'),local_files_only=local)))
    if role in ('all','stt'):
        asr=_asr_snapshot(local_files_only=local)
        files += [_tokens(asr), *(_find(asr,kind) for kind in ('encoder','decoder','joiner'))]
    if role in ('all','tts'):
        bundle=os.getenv('MATCHA_CPU_DIR')
        root=Path(bundle) if bundle else Path(snapshot_download(repo_id=MATCHA_REPO,revision=MATCHA_REVISION,
            allow_patterns=['weights/cpu-onnx/*','scripts/matcha_hybrid_frontend.py'],local_files_only=local))/'weights/cpu-onnx'
        metadata=json.loads((root/'model.json').read_text())
        files += [root/'model.json',root/'phoneme_vocab.json']
        files += [root/item['name'] for item in metadata['files'].values()]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError('serving model cache is incomplete')
    for name, expected in ([] if metadata is None else [('phoneme_vocab.json', metadata['vocab_sha256']),
                           *((item['name'], item['sha256']) for item in metadata['files'].values())]):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('serving model checksum mismatch: ' + name)
    known={
        'model.onnx':'a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808',
        'encoder.int8.onnx':'b528768939c7711a889be81a718ea7f2ee50d0d2d384d53f399e15b44bd9408c',
        'decoder.int8.onnx':'e0a156b5454722a524230f9e35d5d928cfcdd3723437b418e5bd5e04f1c3a101',
        'joiner.int8.onnx':'12636559d135315f002a1e1b477077d415e888477378db0fd450aee5b21ac551',
        'tokens.txt':'f536d03c2e95ebd2930cf0abec88e823bd17d3c1933da7ae6a82db3b80605e15'}
    for path in files:
        if path.name in known and ('e71cae966052b992a7eca6b17738916ce0eca4ec' in str(path) or 'dd9227dcd8705c13f33bdbe59728d546ab94480f' in str(path)):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=known[path.name]:
                raise RuntimeError('serving model checksum mismatch: '+path.name)
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
