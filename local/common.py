import asyncio,time
import aiohttp
async def wait_ready_port(session,port,process,timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None: raise RuntimeError(f'local port-forward {port} exited with {process.returncode}')
        try:
            async with session.get(f'http://127.0.0.1:{port}/readyz',timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status==200: return
        except (aiohttp.ClientError,TimeoutError): pass
        await asyncio.sleep(.25)
    raise RuntimeError(f'local port-forward {port} did not become ready')
