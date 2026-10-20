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
