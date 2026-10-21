import signal
from unittest.mock import Mock, patch
import pytest
from voice_server.stack import Stack, worker_layout


def test_explicit_loopback_urls_override_injected_kubernetes_environment(monkeypatch):
    from voice_server.gateway import Discovery
    monkeypatch.setenv('KUBERNETES_SERVICE_HOST','10.0.0.1')
    monkeypatch.setenv('VOICE_STT_URLS','http://127.0.0.1:19001')
    monkeypatch.setenv('VOICE_TTS_URLS','http://127.0.0.1:19101')
    discovery = Discovery(Mock(),'ephemeral')
    assert not discovery.kube
    assert discovery.static['stt'] == ['http://127.0.0.1:19001']


def test_layout_is_bounded_and_private_workers_are_started():
    assert worker_layout({}) == [('stt',19001),('stt',19002),('tts',19101),('tts',19102),('tts',19103),('tts',19104)]
    with pytest.raises(ValueError):
        worker_layout({'VOICE_STACK_TTS_WORKERS':'5'})
    stack = Stack({})
    with patch('voice_server.stack.subprocess.Popen') as popen:
        stack.start('tts', 19101)
    args, kwargs = popen.call_args
    assert args[0][-4:] == ['--host','127.0.0.1','--port','19101']
    assert kwargs['env']['VOICE_TTS_THREADS'] == '1'
    assert kwargs['env']['VOICE_TTS_WORKERS'] == '1'
    assert len(kwargs['env']['VOICE_RPC_TOKEN']) >= 32


def test_drains_gateway_before_workers():
    stack = Stack({})
    stack.gateway = Mock()
    stack.workers = [Mock(), Mock()]
    with patch.object(stack, 'terminate') as terminate:
        stack.stop()
    assert terminate.call_args_list[0].args == ([stack.gateway],125)
    assert terminate.call_args_list[1].args == (stack.workers,20)


def test_dead_worker_fails_whole_stack():
    stack = Stack({})
    child = Mock(); child.poll.return_value = 1
    stack.workers = [child]
    assert not stack.running()


def test_native_worker_health_failure_is_not_hidden_by_living_http_owner():
    stack = Stack({'VOICE_STACK_STT_WORKERS':'1','VOICE_STACK_TTS_WORKERS':'1'})
    with patch('voice_server.stack.urllib.request.urlopen',side_effect=OSError('native child failed')):
        stack.check_workers()
        stack.check_workers()
        with pytest.raises(RuntimeError,match='failed health checks'):
            stack.check_workers()


def test_terminate_escalates_and_reaps_child():
    import subprocess
    child = Mock(); child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired('voice',0),0]
    Stack.terminate([child],0)
    child.send_signal.assert_called_once_with(signal.SIGTERM)
    child.kill.assert_called_once()
    assert child.wait.call_count == 2
