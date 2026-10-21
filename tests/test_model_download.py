from voice_server import workers


def test_default_asr_download_keeps_selected_graphs_and_excludes_training_files(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        for name in ('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'):
            (tmp_path / name).touch()
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot_download)
    assert workers._asr_snapshot() == tmp_path
    assert len(calls) == 1
    assert calls[0]['revision'] == workers.ASR_REVISION
    assert calls[0]['allow_patterns'] == ['encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt']
    assert workers._find(tmp_path, 'encoder').name == 'encoder.int8.onnx'


def test_custom_asr_repository_preserves_full_snapshot_for_external_graph_data(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot_download)
    monkeypatch.setattr(workers,'ASR_REPO','custom/external-data-model')
    assert workers._asr_snapshot() == tmp_path
    assert len(calls) == 1
    assert 'allow_patterns' not in calls[0]


def test_prepare_checks_explicit_tts_bundle_without_hub_download(tmp_path,monkeypatch):
    import hashlib,json
    from voice_server import prepare_models
    monkeypatch.setenv('VOICE_ROLE','tts');monkeypatch.setenv('MATCHA_CPU_DIR',str(tmp_path))
    monkeypatch.setattr(prepare_models,'snapshot_download',lambda **kw:(_ for _ in ()).throw(AssertionError('unexpected download')))
    files={}
    for name in ['matcha_encoder.onnx','matcha_decoder.onnx','vocos_spectrum.onnx','phoneme_vocab.json']:
        (tmp_path/name).write_bytes(name.encode())
        files[name]={'name':name,'sha256':hashlib.sha256(name.encode()).hexdigest()}
    metadata={'vocab_sha256':files.pop('phoneme_vocab.json')['sha256'],'files':files}
    (tmp_path/'model.json').write_text(json.dumps(metadata))
    assert prepare_models.prepare(local=True)>0
    (tmp_path/'matcha_decoder.onnx').write_bytes(b'corrupt')
    import pytest
    with pytest.raises(RuntimeError,match='checksum mismatch'):prepare_models.prepare(local=True)
