"""TEST ONLY: isolate gateway costs with fixed STT and real precomputed TTS PCM.

Never enabled by the application environment or public API.
"""
import asyncio,os,signal
from pathlib import Path
from aiohttp import web
from voice_server.server import Service,create_app,drain
class FixedPool:
    admitted=running=rejected=offered=0
    def __init__(self,result): self.result=result
    async def call(self,*args,**kwargs): await asyncio.sleep(0); return self.result
    def close(self): pass
async def main():
    os.environ['VOICE_ROLE']='gateway'; os.environ['VOICE_RPC_TOKEN']='disposable-test-only'
    svc=Service(); await svc.warm()
    svc.asr=FixedPool('Hôm nay ở Sài Gòn bao nhiêu độ, có nên mang ô không?')
    svc.tts=FixedPool((24000,Path('/fixtures/reply.pcm').read_bytes()))
    runner=web.AppRunner(create_app(svc),handler_cancellation=True); await runner.setup(); await web.TCPSite(runner,'0.0.0.0',8765).start()
    stop=asyncio.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): asyncio.get_running_loop().add_signal_handler(sig,stop.set)
    await stop.wait(); await drain(svc); await runner.cleanup(); await svc.aclose()
if __name__=='__main__': asyncio.run(main())
