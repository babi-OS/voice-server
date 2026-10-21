import pytest
from voice_server.benchmark import arguments


def test_builtin_benchmark_is_loopback_only_and_bounded():
    args = arguments(['--output','/tmp/report','--users','8'])
    assert args.url == 'ws://127.0.0.1:8765/v2/voice'
    for extra in (['--users','33'],['--turns','100'],['--period','1'],
                  ['--turns','40','--period','30'],['--url','ws://example.com']):
        with pytest.raises(SystemExit):
            arguments(['--output','/tmp/report']+extra)
