"""Run the optimized roles in a single resource-bounded container.

Workers listen only on loopback. Kubernetes should probe the public gateway;
it is started after all workers are ready and stopped before worker draining.
The parent exits on any child failure so Kubernetes can restart the whole stack.
"""
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request


def worker_layout(env):
    result = []
    for stage, base, default in [('stt', 19000, '2'), ('tts', 19100, '4')]:
        count = int(env.get('VOICE_STACK_' + stage.upper() + '_WORKERS', default))
        if not 1 <= count <= 4:
            raise ValueError('stack worker counts must be 1..4')
        result.extend((stage, base + i) for i in range(1, count + 1))
    return result


class Stack:
    def __init__(self, env=None):
        self.env = dict(os.environ if env is None else env)
        self.layout = worker_layout(self.env)
        # Shared only by this container's children, never logged or persisted.
        self.env['VOICE_RPC_TOKEN'] = secrets.token_urlsafe(32)
        self.env.update(VOICE_ASR_WORKERS='1', VOICE_TTS_WORKERS='1')
        for key, value in {'VOICE_ASR_THREADS':'1', 'VOICE_TTS_THREADS':'1',
                           'VOICE_STT_ADMISSION':'16', 'VOICE_TTS_ADMISSION':'16',
                           'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                           'VOICE_ORT_SPINNING':'0'}.items():
            self.env.setdefault(key, value)
        self.workers = []
        self.gateway = None
        self.stopping = False

    def start(self, role, port, host='127.0.0.1', **extra):
        return subprocess.Popen([sys.executable, '-m', 'voice_server', '--host',
                                 host, '--port', str(port)],
                                env=self.env | {'VOICE_ROLE': role} | extra)

    def running(self):
        return all(p.poll() is None for p in self.workers) and (
            self.gateway is None or self.gateway.poll() is None)

    def request_stop(self, *_):
        self.stopping = True

    @staticmethod
    def terminate(processes, timeout):
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def stop(self):
        # Active gateway calls retain their worker until completion/cancellation.
        if self.gateway is not None:
            self.terminate([self.gateway], 125)
        self.terminate(self.workers, 20)

    def run(self):
        try:
            for stage, port in self.layout:
                self.workers.append(self.start(stage, port))
            deadline = time.monotonic() + 900
            pending = {port for _, port in self.layout}
            while pending and not self.stopping:
                if not self.running():
                    raise RuntimeError('a voice worker exited during startup')
                if time.monotonic() >= deadline:
                    raise TimeoutError('voice worker readiness deadline exceeded')
                for port in list(pending):
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/readyz', timeout=.5) as response:
                            if response.status == 200:
                                pending.remove(port)
                    except OSError:
                        pass
                time.sleep(.2)
            if self.stopping:
                return
            urls = {'VOICE_' + stage.upper() + '_URLS': ','.join(
                f'http://127.0.0.1:{p}' for s, p in self.layout if s == stage)
                for stage in ('stt', 'tts')}
            self.gateway = self.start('gateway', 8765, '0.0.0.0', **urls)
            while not self.stopping:
                if not self.running():
                    raise RuntimeError('a voice role exited; restarting the whole stack')
                time.sleep(.2)
        finally:
            self.stop()


def main():
    stack = Stack()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stack.request_stop)
    stack.run()


if __name__ == '__main__':
    main()
