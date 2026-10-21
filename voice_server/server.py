from __future__ import annotations
import asyncio, json, logging, math, os, re, secrets, signal, time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import opuslib_next
from aiohttp import web
from .protocol import *
from .codec import encode_batch, MAX_ENCODE_BATCH, make_encoder, validate_input_packet
from .workers import WorkerPool, WorkerBusy
from .batching import make_worker_pool
from .runtime import integer_setting
LOG=logging.getLogger('voice_server')
MAX_UTTERANCE_BYTES=30*INPUT_RATE*2; MAX_TTS_TEXT=1500; ID_RE=re.compile(r'^[A-Za-z0-9_-]{1,64}$')
VAD_WINDOW_MS=32
VAD_WINDOW_BYTES=INPUT_RATE*VAD_WINDOW_MS//1000*2

def ident(v,label):
    if not isinstance(v,str) or not ID_RE.fullmatch(v): raise ProtocolError(f'{label} must be 1-64 ASCII characters')
    return v
def pcm_resample(pcm,rate,out):
    if rate==out:return pcm
    a=np.frombuffer(pcm,dtype='<i2').astype(np.float32); d=math.gcd(rate,out)
    return np.clip(np.rint(resample_poly(a,out//d,rate//d)),-32768,32767).astype('<i2').tobytes()
def samples_pcm(samples,rate,out):
    a=np.asarray(samples,dtype=np.float32)
    if rate!=out:
        d=math.gcd(rate,out); a=resample_poly(a,out//d,rate//d)
    return (np.clip(a,-1,1)*32767).astype('<i2').tobytes()
class Service:
    def __init__(self):
        self.role=os.getenv('VOICE_ROLE','all')
        if self.role not in ('all','gateway','stt','tts'): raise ValueError('invalid VOICE_ROLE')
        self.vad=self.asr=self.tts=None; self.codec=None; self.session=None; self.discovery=None
        self.connections=set(); self.draining=False; self.loop_task=None
        from .telemetry import Telemetry
        self.telemetry=Telemetry()
        if self.role=='all':
            self.vad=WorkerPool('vad',64); self.asr=make_worker_pool('asr',8); self.tts=make_worker_pool('tts',4)
        elif self.role=='gateway':
            from .gateway import VadThread,CodecExecutor
            self.vad=VadThread(); self.codec=CodecExecutor()
        else:
            kind='asr' if self.role=='stt' else 'tts'
            if int(os.getenv('VOICE_'+kind.upper()+'_WORKERS','1'))!=1: raise ValueError('split workers require one native process')
            setattr(self,kind,make_worker_pool(kind,integer_setting('VOICE_'+self.role.upper()+'_ADMISSION',2,maximum=256)))
        if self.role!='all' and not os.getenv('VOICE_RPC_TOKEN'): raise ValueError('VOICE_RPC_TOKEN required for split roles')
    async def warm(self):
        if self.role=='gateway':
            import aiohttp
            from .gateway import Discovery,RemotePool
            self.session=aiohttp.ClientSession()
            self.discovery=Discovery(self.session,os.environ['VOICE_RPC_TOKEN'])
            await self.discovery.start()
            self.asr=RemotePool('stt',self.discovery); self.tts=RemotePool('tts',self.discovery)
        await asyncio.gather(*(p.warm() for p in (self.vad,self.asr,self.tts) if p))
        self.loop_task=asyncio.create_task(self.monitor_loop())
    async def monitor_loop(self):
        while True:
            started=time.monotonic(); await asyncio.sleep(.02)
            self.telemetry.observe('event_loop_delay',max(0.,time.monotonic()-started-.02),self.role)
    async def aclose(self):
        if self.loop_task:
            self.loop_task.cancel(); await asyncio.gather(self.loop_task,return_exceptions=True)
        if self.discovery: await self.discovery.close()
        if self.session: await self.session.close()
        if self.role=='gateway': await self.vad.aclose()
        self.close()
    def close(self):
        for pool in (self.vad,self.asr,self.tts):
            if pool: pool.close()
        if self.codec: self.codec.close()
    async def convert(self,fn,*args):
        if self.codec: return await self.codec.call(fn,*args)
        return fn(*args)
class Connection:
    def __init__(self,ws,svc):
        self.ws=ws; self.svc=svc; self.audio=None; self.decoder=None; self.encoder=None; self.send_lock=asyncio.Lock(); self.key=secrets.token_hex(8)
        self.codec_lock=asyncio.Lock()
        self.last_activity=None
        self.stt=None; self.stt_stream=None; self.stt_seq=0; self.stt_task=None; self.stt_processing=None; self.pending=bytearray(); self.preroll=deque(maxlen=16); self.utterance=bytearray(); self.speaking=False; self.voice=0; self.silence=0
        self.tts=None; self.tts_stream=None; self.tts_task=None; self.tts_q=None; self.tts_next=0; self.tts_len=0; self.tts_finished=False; self.completed={}
    async def convert(self,fn,*args):
        if getattr(self.svc,'codec',None):
            async with self.codec_lock:
                task=asyncio.create_task(self.svc.convert(fn,*args))
                try: return await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
        return fn(*args)
    async def send(self,k,**f):
        async with self.send_lock: await self.ws.send_str(encode_control(k,**f))
    async def send_audio(self,s,q,p):
        async with self.send_lock: await self.ws.send_bytes(pack_audio(s,q,p))
    async def end(self,r,status,result):
        self.completed[r]={'status':status,'result':result}; await self.send('request.end',request_id=r,status=status,result=result)
    def clear_stt(self): self.stt=self.stt_stream=None; self.stt_seq=0; self.pending.clear(); self.preroll.clear(); self.utterance.clear(); self.speaking=False; self.voice=self.silence=0; self.end_silence_windows=math.ceil(DEFAULT_END_SILENCE_MS/VAD_WINDOW_MS)
    def clear_tts(self): self.tts=self.tts_stream=None; self.tts_q=None; self.tts_next=self.tts_len=0; self.tts_finished=False
    async def control(self,text,received_at=None):
        d=parse_control(text); k=d['type']
        if k!='hello': self.last_activity=time.monotonic()
        if self.audio is None:
            if k!='hello': raise ProtocolError('hello required first')
            self.audio=validate_hello(d); self.decoder=opuslib_next.Decoder(self.audio.input.sample_rate,1) if self.audio.input.codec=='opus' else None; self.encoder=make_encoder(self.audio.output) if self.audio.output.codec=='opus' else None
            await self.send('ready',version=2,audio=self.audio.wire(),capabilities={'stt':True,'tts':True,'model':False}); return
        if k=='request.cancel': await self.cancel(ident(d.get('request_id'),'request_id')); return
        if k in ('stt.start','tts.start') and getattr(self.svc,'draining',False):
            raise ProtocolError('replica draining; reconnect','draining')
        if k=='stt.start': await self.start_stt(d); return
        if k=='stt.finish':
            r=ident(d.get('request_id'),'request_id')
            if r==self.stt_processing or r in self.completed: return
            if r!=self.stt: raise ProtocolError('STT request is not active','not_active')
            await self.finish_stt(received_at=received_at); return
        if k=='tts.start': await self.start_tts(d); return
        if k=='tts.text': await self.tts_text(d,received_at=received_at); return
        if k=='tts.finish':
            if d.get('request_id')!=self.tts or self.tts_finished: raise ProtocolError('TTS request is not active','not_active')
            self.tts_finished=True; self.tts_q.put_nowait(None); await self.send('tts.accepted',request_id=self.tts,chunks=self.tts_next); return
        raise ProtocolError(f'unsupported control type {k}')
    async def start_stt(self,d):
        if self.stt or self.stt_task or self.tts: raise ProtocolError('another operation is active','busy')
        if getattr(self.svc,'role','all')=='gateway' and sum(bool(c.stt) for c in self.svc.connections)>=integer_setting('VOICE_GATEWAY_MAX_MICROPHONES',32,maximum=256): raise ProtocolError('microphone capacity full','overloaded')
        r=ident(d.get('request_id'),'request_id'); t=ident(d.get('turn_id'),'turn_id')
        silence_ms=end_silence_ms(d)
        if r in self.completed: await self.send('request.end',request_id=r,**self.completed[r]); return
        self.clear_stt(); self.end_silence_windows=math.ceil(silence_ms/VAD_WINDOW_MS); self.stt=r; self.stt_stream=secrets.randbelow(0xffffffff)+1
        try: await self.svc.vad.call('reset',self.key,timeout=3)
        except Exception:
            self.clear_stt(); raise
        await self.send('stt.ready',request_id=r,turn_id=t,stream_id=self.stt_stream)
    async def packet(self,raw):
        self.last_activity=time.monotonic()
        sid,seq,payload=unpack_audio(raw)
        if sid!=self.stt_stream or not self.stt: raise ProtocolError('audio stream is not active','late_packet')
        if seq!=self.stt_seq: raise ProtocolError('audio sequence mismatch','sequence_mismatch')
        a=self.audio.input
        if a.codec=='opus': validate_input_packet(payload,a); pcm=await self.convert(self.decoder.decode,payload,a.sample_rate*a.packet_ms//1000); validate_pcm16(pcm,a.packet_bytes)
        else: validate_pcm16(payload,a.packet_bytes); pcm=payload
        self.stt_seq+=1
        # The common negotiated 16-kHz path needs no resampling or executor handoff.
        self.pending.extend(pcm if a.sample_rate==INPUT_RATE else await self.convert(pcm_resample,pcm,a.sample_rate,INPUT_RATE))
        while len(self.pending)>=VAD_WINDOW_BYTES and self.stt:
            w=bytes(self.pending[:VAD_WINDOW_BYTES]); del self.pending[:VAD_WINDOW_BYTES]; p=await self.svc.vad.call('feed',(self.key,w),timeout=5); await self.vad_window(w,p)
    async def vad_window(self,w,p):
        voiced=p>=.5
        if not self.speaking:
            self.preroll.append(w); self.voice=self.voice+1 if voiced else 0
            if self.voice>=2: self.speaking=True; self.utterance.extend(b''.join(self.preroll)); self.preroll.clear(); await self.send('stt.speech_start',request_id=self.stt)
            return
        self.utterance.extend(w); self.silence=0 if voiced else self.silence+1
        if len(self.utterance)>MAX_UTTERANCE_BYTES: r=self.stt; self.clear_stt(); await self.end(r,'failed',{'code':'utterance_too_long'}); return
        if self.end_silence_windows and self.silence>=self.end_silence_windows: await self.finish_stt()
    async def finish_stt(self,received_at=None):
        r=self.stt
        if not r:return
        pcm=bytes(self.utterance+self.pending) if self.speaking else b''; self.clear_stt()
        if len(pcm)>MAX_UTTERANCE_BYTES: await self.end(r,'failed',{'code':'utterance_too_long'}); return
        if pcm: await self.send('stt.speech_end',request_id=r)
        if len(pcm)<INPUT_RATE//4*2: await self.send('stt.final',request_id=r,text=''); await self.end(r,'completed',{'transcript':''}); return
        async def run():
            try:
                text=await self.stage_call(self.svc.asr,'decode',pcm,timeout=90,received_at=received_at); await self.send('stt.final',request_id=r,text=text); await self.end(r,'completed',{'transcript':text})
            except asyncio.CancelledError: raise
            except WorkerBusy as e: await self.end(r,'failed',{'code':'overloaded','stage':'stt','retry_after_ms':1000})
            except Exception as e: LOG.exception('STT failed'); await self.end(r,'failed',{'code':'stt_failed','message':str(e)})
            finally: self.stt_task=None; self.stt_processing=None
        self.stt_processing=r; self.stt_task=asyncio.create_task(run())
    async def stage_call(self,pool,command,payload,*,timeout,received_at=None):
        if getattr(self.svc,'role','all')!='gateway':
            return await pool.call(command,payload,timeout=timeout)
        # Reserve 150 ms for RPC/overload reporting within the two-second budget.
        budget=1.85-(time.monotonic()-received_at) if received_at is not None else 1.85
        if budget<=0: raise WorkerBusy('submitted stage deadline exceeded',pool.kind)
        return await pool.call(command,payload,timeout=timeout,queue_budget=budget)
    async def start_tts(self,d):
        if self.tts or self.tts_task or self.stt or self.stt_task: raise ProtocolError('another operation is active','busy')
        r=ident(d.get('request_id'),'request_id'); t=ident(d.get('turn_id'),'turn_id')
        if r in self.completed: await self.send('request.end',request_id=r,**self.completed[r]); return
        self.tts=r; self.tts_stream=secrets.randbelow(0xffffffff)+1; self.tts_q=asyncio.Queue(maxsize=3); self.tts_task=asyncio.create_task(self.run_tts(r,t,self.tts_stream,self.tts_q)); await self.send('tts.ready',request_id=r,turn_id=t,stream_id=self.tts_stream)
    async def tts_text(self,d,received_at=None):
        if d.get('request_id')!=self.tts or self.tts_finished: raise ProtocolError('TTS request is not accepting text','not_active')
        i=d.get('index'); text=d.get('text')
        if type(i)is not int or i!=self.tts_next: raise ProtocolError('TTS chunk index must be next','sequence_mismatch')
        if not isinstance(text,str) or not text.strip() or len(text)>500 or self.tts_len+len(text)>MAX_TTS_TEXT: raise ProtocolError('invalid TTS text','too_large')
        if self.tts_q.full(): raise ProtocolError('TTS queue full','busy')
        self.tts_q.put_nowait((i,text.strip(),received_at)); self.tts_next+=1; self.tts_len+=len(text); await self.send('tts.accepted',request_id=self.tts,index=i)
    async def run_tts(self,r,t,s,q):
        seq=count=0
        try:
            while True:
                item=await asyncio.wait_for(q.get(),45)
                if item is None: break
                i,text,received_at=item; rate,samples=await self.stage_call(self.svc.tts,'synthesize',text,timeout=120,received_at=received_at); out=self.audio.output; pcm=await self.convert(pcm_resample,samples,rate,out.sample_rate) if isinstance(samples,bytes) else await self.convert(samples_pcm,samples,rate,out.sample_rate); await self.send('tts.audio_start',request_id=r,turn_id=t,stream_id=s,index=i,sample_rate=out.sample_rate)
                batch_packets=MAX_ENCODE_BATCH if out.codec=='opus' and getattr(self.svc,'codec',None) else 1
                for off in range(0,len(pcm),out.packet_bytes*batch_packets):
                    if out.codec=='opus':
                        payloads=await self.convert(encode_batch,pcm[off:off+out.packet_bytes*batch_packets],out,self.encoder)
                    else: payloads=[pcm[off:off+out.packet_bytes].ljust(out.packet_bytes,b'\0')]
                    for payload in payloads:
                        await self.send_audio(s,seq,payload); seq+=1
                await self.send('tts.audio_end',request_id=r,stream_id=s,index=i); count+=1
            await self.end(r,'completed',{'chunks':count,'packets':seq})
        except asyncio.CancelledError: raise
        except WorkerBusy as e: await self.end(r,'failed',{'code':'overloaded','stage':'tts','retry_after_ms':1000})
        except Exception as e: LOG.exception('TTS failed'); await self.end(r,'failed',{'code':'tts_failed','message':str(e)})
        finally: self.clear_tts(); self.tts_task=None
    async def cancel(self,r):
        active=r in (self.stt,self.tts) or self.stt_processing==r
        if self.stt==r:self.clear_stt()
        tasks=[]
        if self.stt_task and self.stt_processing==r: tasks.append(self.stt_task)
        if self.tts==r and self.tts_task: tasks.append(self.tts_task)
        for task in tasks: task.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)
        if self.stt_processing==r: self.stt_task=None; self.stt_processing=None
        if self.tts==r: self.clear_tts(); self.tts_task=None
        if not active and r not in self.completed: raise ProtocolError('request not found','not_found')
        if active: await self.send('request.cancelled',request_id=r); await self.end(r,'cancelled',{'code':'client_cancelled'})
        else: await self.send('request.end',request_id=r,**self.completed[r])
    async def close(self):
        tasks=[task for task in (self.stt_task,self.tts_task) if task]
        for task in tasks:
            if not task.done(): task.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)
        self.clear_stt(); self.clear_tts(); self.stt_task=self.stt_processing=self.tts_task=None
        try: await self.svc.vad.call('release',self.key,timeout=2)
        except Exception: pass
async def handle_voice(req):
    if req.app['service'].draining:
        raise web.HTTPServiceUnavailable(text='voice replica is draining',headers={'Retry-After':'1'})
    if getattr(req.app['service'],'role','all')=='gateway' and len(req.app['service'].connections)>=integer_setting('VOICE_GATEWAY_MAX_CONNECTIONS',256,maximum=1024): raise web.HTTPTooManyRequests(text='connection capacity full')
    ws=web.WebSocketResponse(max_msg_size=8192+8)
    ws.headers['X-Voice-Replica']=os.getenv('HOSTNAME','voice-server')
    await ws.prepare(req); c=Connection(ws,req.app['service'])
    c.svc.connections.add(c)
    ingress=None
    if getattr(c.svc,'role','all')=='gateway':
        from .ingress import Ingress
        ingress=Ingress(c,req.app['ingress'])
    try:
        async for msg in ws:
            try:
                if msg.type in (web.WSMsgType.TEXT,web.WSMsgType.BINARY):
                    if ingress: ingress.offer(msg.type,msg.data)
                    elif msg.type==web.WSMsgType.TEXT: await c.control(msg.data)
                    else: await c.packet(msg.data)
                elif msg.type==web.WSMsgType.ERROR: break
            except (ProtocolError,WorkerBusy) as e: await c.send('error',code=getattr(e,'code','busy'),message=str(e))
            except Exception: LOG.exception('voice request failed'); await c.send('error',code='internal_error',message='voice operation failed')
    finally:
        try:
            if ingress: await ingress.close()
            await c.close()
        finally: c.svc.connections.discard(c)
    return ws
def healthy(svc):
    ingress=getattr(svc,'ingress',None)
    if ingress and ingress.task and ingress.task.done() and not svc.draining: return False
    return all(getattr(getattr(svc,k,None),'healthy',True) and not any(getattr(e,'_broken',False) for e in getattr(getattr(svc,k,None),'executors',[])) for k in ('vad','asr','tts')) and not (getattr(getattr(svc,'vad',None),'task',None) and svc.vad.task.done() and not svc.draining)
async def health(req):
    ok=bool(healthy(req.app['service']))
    return web.json_response({'ok':ok,'service':'voice-server','protocol':2,'stateless':True},status=200 if ok else 503)
async def ready(req):
    svc=req.app['service']; ok=healthy(svc) and not svc.draining
    if getattr(svc,'role','all')=='gateway':
        discovery=getattr(svc,'discovery',None)
        ok=ok and discovery is not None and time.monotonic()-discovery.last<=30
    return web.json_response({'ok':bool(ok)},status=200 if ok else 503)
async def metrics(req):
    svc=req.app['service']
    now=time.monotonic()
    active=sum(bool(c.stt or c.stt_task or c.tts or c.tts_task or
                    (c.last_activity is not None and now-c.last_activity<30)) for c in svc.connections)
    lines=['# TYPE voice_connections gauge',f'voice_connections {sum(c.audio is not None for c in svc.connections)}',
           '# TYPE voice_active_connections gauge',f'voice_active_connections {active}',
           '# TYPE voice_draining gauge',f'voice_draining {int(svc.draining)}',
           '# TYPE voice_worker_admitted gauge','# TYPE voice_worker_running gauge',
           '# TYPE voice_worker_queued gauge','# TYPE voice_worker_rejected_total counter']
    for kind in ('vad','asr','tts'):
        pool=getattr(svc,kind)
        if pool is None: continue
        stage='stt' if kind=='asr' else kind
        if hasattr(pool,'telemetry'): lines.extend(pool.telemetry.render())
        for size,count in getattr(pool,'batch_sizes',{}).items():
            lines.append(f'voice_native_batches_total{{stage="{stage}",size="{size}"}} {count}')
        for (model,size),count in getattr(pool,'model_batch_sizes',{}).items():
            lines.append(f'voice_model_batches_total{{stage="{stage}",model="{model}",size="{size}"}} {count}')
        lines.append(f'voice_stage_admitted{{stage="{stage}"}} {pool.admitted}')
        for reason,count in getattr(pool,'rejection_reasons',{}).items():
            lines.append(f'voice_admission_rejected_total{{stage="{stage}",reason="{reason}"}} {count}')
        lines.append(f'voice_stage_rejected_total{{stage="{stage}"}} {pool.rejected}')
        lines.append(f'voice_stage_offered_total{{stage="{stage}"}} {getattr(pool,"offered",0)}')
        for metric,value in [('admitted',pool.admitted),('running',pool.running),
                             ('queued',pool.admitted-pool.running),('rejected_total',pool.rejected)]:
            lines.append(f'voice_worker_{metric}{{kind="{kind}"}} {value}')
    lines.append(f'voice_active_microphones {sum(bool(c.stt) for c in svc.connections)}')
    if hasattr(svc,'telemetry'): lines.extend(svc.telemetry.render())
    if getattr(svc,'codec',None): lines.extend(svc.codec.telemetry.render())
    ingress=req.app.get('ingress')
    if ingress:
        lines.extend(ingress.telemetry.render())
        lines.append(f'voice_ingress_pending {ingress.pending}')
        for reason,n in ingress.rejections.items():
            lines.append(f'voice_ingress_rejected_total{{reason="{reason}"}} {n}')
    return web.Response(text='\n'.join(lines)+'\n',content_type='text/plain')

async def drain(service,timeout=120):
    """Stop admitting connections, finish active speech operations, close idle sockets."""
    service.draining=True
    deadline=asyncio.get_running_loop().time()+timeout
    while service.connections or any(p and p.admitted for p in (service.asr,service.tts)):
        idle=[c for c in tuple(service.connections) if not any((c.stt,c.stt_task,c.tts,c.tts_task))]
        if idle:
            await asyncio.gather(*(c.ws.close(code=1012,message=b'replica draining; reconnect') for c in idle))
        if asyncio.get_running_loop().time()>=deadline:
            break
        await asyncio.sleep(.05)
    await asyncio.gather(*(c.ws.close(code=1012,message=b'replica draining; reconnect') for c in tuple(service.connections)))

def create_app(service):
    if not hasattr(service,'connections'): service.connections=set()
    if not hasattr(service,'draining'): service.draining=False
    app=web.Application(client_max_size=MAX_UTTERANCE_BYTES); app['service']=service
    if getattr(service,'role','all')=='gateway':
        from .ingress import IngressLimiter
        app['ingress']=service.ingress=IngressLimiter()
        async def ingress_lifetime(app):
            await app['ingress'].start()
            yield
            await app['ingress'].close()
        app.cleanup_ctx.append(ingress_lifetime)
    app.router.add_get('/healthz',health); app.router.add_get('/readyz',ready); app.router.add_get('/metrics',metrics)
    if getattr(service,'role','all') in ('all','gateway'):
        root=Path(__file__).parent/'static'
        app.router.add_get('/v2/voice',handle_voice); app.router.add_get('/',lambda r:web.FileResponse(root/'index.html')); app.router.add_get('/mic-worklet.js',lambda r:web.FileResponse(root/'mic-worklet.js'))
    else:
        from .rpc import capacity,infer
        app.router.add_get('/capacity',capacity); app.router.add_post('/internal/v1/'+service.role,infer)
    return app
async def run(host='0.0.0.0',port=8765):
    svc=Service(); stop=asyncio.Event(); loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(sig,stop.set)
    runner=None
    try:
        await svc.warm(); runner=web.AppRunner(create_app(svc),handle_signals=False,shutdown_timeout=5,handler_cancellation=True)
        await runner.setup(); site=web.TCPSite(runner,host,port); await site.start(); LOG.info('listening on %s:%s',host,port)
        await stop.wait()
        await drain(svc)
    finally:
        if runner: await runner.cleanup()
        await svc.aclose()
        for sig in (signal.SIGINT,signal.SIGTERM): loop.remove_signal_handler(sig)
