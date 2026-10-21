"""Private binary-PCM role APIs. Admission belongs to the native worker owner."""
import asyncio
import hmac
import os
import time
from aiohttp import web
from .workers import WorkerBusy,CALL_TIMING
from .protocol import RATES

def authenticated(req):
    expected='Bearer '+os.environ['VOICE_RPC_TOKEN']
    if not hmac.compare_digest(req.headers.get('Authorization','').encode(),expected.encode()): raise web.HTTPUnauthorized()

def busy(stage,code='overloaded',status=429):
    return web.json_response({'code':code,'stage':stage,'retry_after_ms':1000,'not_started':True},status=status,headers={'Retry-After':'1'})

async def capacity(req):
    authenticated(req); svc=req.app['service']; pool=svc.asr if svc.role=='stt' else svc.tts
    return web.json_response({'stage':svc.role,'admitted':pool.admitted,'running':pool.running,'queued':pool.admitted-pool.running,'slots':1,'estimated_service_s':pool.estimate})

async def infer(req):
    authenticated(req); svc=req.app['service']; stage=svc.role
    if svc.draining: return busy(stage,'unavailable',503)
    try:
        budget=min(2000,max(0,int(req.headers.get('X-Queue-Budget-Ms','2000'))))/1000
        if stage=='stt':
            payload=await req.read()
            if not payload or len(payload)%2 or len(payload)>960000: raise ValueError('invalid PCM16 body')
            pool=svc.asr; command='decode'; timeout=90
        else:
            data=await req.json(); payload=data['text']; rate=data.get('sample_rate',24000)
            if not isinstance(payload,str) or not payload.strip() or len(payload)>1500: raise ValueError('invalid text')
            if type(rate) is not int or rate not in RATES: raise ValueError('unsupported sample rate')
            pool=svc.tts; command='synthesize'; timeout=120
    except (ValueError,KeyError,TypeError): raise web.HTTPBadRequest(text='invalid inference request')
    if not getattr(pool,'healthy',True): return busy(stage,'unavailable',503)
    entered=time.monotonic()
    try: result=await pool.call(command,payload,timeout=timeout,queue_budget=budget)
    except WorkerBusy: return busy(stage)
    except TimeoutError: return web.json_response({'code':'inference_timeout','stage':stage,'not_started':False},status=504)
    # Per-request values: do not read mutable "last duration" metrics under concurrency.
    queue_wait,inference=CALL_TIMING.get() or (0.,time.monotonic()-entered)
    headers={'X-Queue-Wait-Ms':str(queue_wait*1000),'X-Inference-Ms':str(inference*1000),'X-Voice-Replica':os.getenv('HOSTNAME','local')}
    if stage=='stt': return web.json_response({'text':result},headers=headers)
    from .server import samples_pcm
    pcm=await asyncio.to_thread(samples_pcm,result[1],result[0],rate)
    headers.update({'X-Audio-Format':'pcm_s16le','X-Audio-Sample-Rate':str(rate)})
    return web.Response(body=pcm,content_type='application/octet-stream',headers=headers)
