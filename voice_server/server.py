from __future__ import annotations
import asyncio, json, logging, math, os, re, secrets, signal, time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import opuslib_next
from aiohttp import web
from .protocol import *
from .codec import encode_packet, make_encoder, validate_input_packet
from .workers import WorkerPool, WorkerBusy
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
        self.vad=WorkerPool('vad',64); self.asr=WorkerPool('asr',8); self.tts=WorkerPool('tts',4)
        self.connections=set(); self.draining=False
    async def warm(self): await asyncio.gather(self.vad.warm(),self.asr.warm(),self.tts.warm())
    def close(self): self.vad.close(); self.asr.close(); self.tts.close()
class Connection:
    def __init__(self,ws,svc):
        self.ws=ws; self.svc=svc; self.audio=None; self.decoder=None; self.encoder=None; self.send_lock=asyncio.Lock(); self.key=secrets.token_hex(8)
        self.stt=None; self.stt_stream=None; self.stt_seq=0; self.stt_task=None; self.stt_processing=None; self.pending=bytearray(); self.preroll=deque(maxlen=16); self.utterance=bytearray(); self.speaking=False; self.voice=0; self.silence=0
        self.tts=None; self.tts_stream=None; self.tts_task=None; self.tts_q=None; self.tts_next=0; self.tts_len=0; self.tts_finished=False; self.completed={}
    async def send(self,k,**f):
        async with self.send_lock: await self.ws.send_str(encode_control(k,**f))
    async def send_audio(self,s,q,p):
        async with self.send_lock: await self.ws.send_bytes(pack_audio(s,q,p))
    async def end(self,r,status,result):
        self.completed[r]={'status':status,'result':result}; await self.send('request.end',request_id=r,status=status,result=result)
    def clear_stt(self): self.stt=self.stt_stream=None; self.stt_seq=0; self.pending.clear(); self.preroll.clear(); self.utterance.clear(); self.speaking=False; self.voice=self.silence=0; self.end_silence_windows=math.ceil(DEFAULT_END_SILENCE_MS/VAD_WINDOW_MS)
    def clear_tts(self): self.tts=self.tts_stream=None; self.tts_q=None; self.tts_next=self.tts_len=0; self.tts_finished=False
    async def control(self,text):
        d=parse_control(text); k=d['type']
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
            await self.finish_stt(); return
        if k=='tts.start': await self.start_tts(d); return
        if k=='tts.text': await self.tts_text(d); return
        if k=='tts.finish':
            if d.get('request_id')!=self.tts or self.tts_finished: raise ProtocolError('TTS request is not active','not_active')
            self.tts_finished=True; self.tts_q.put_nowait(None); await self.send('tts.accepted',request_id=self.tts,chunks=self.tts_next); return
        raise ProtocolError(f'unsupported control type {k}')
    async def start_stt(self,d):
        if self.stt or self.stt_task or self.tts: raise ProtocolError('another operation is active','busy')
        r=ident(d.get('request_id'),'request_id'); t=ident(d.get('turn_id'),'turn_id')
        silence_ms=end_silence_ms(d)
        if r in self.completed: await self.send('request.end',request_id=r,**self.completed[r]); return
        self.clear_stt(); self.end_silence_windows=math.ceil(silence_ms/VAD_WINDOW_MS); self.stt=r; self.stt_stream=secrets.randbelow(0xffffffff)+1
        try: await self.svc.vad.call('reset',self.key,timeout=3)
        except Exception:
            self.clear_stt(); raise
        await self.send('stt.ready',request_id=r,turn_id=t,stream_id=self.stt_stream)
    async def packet(self,raw):
        sid,seq,payload=unpack_audio(raw)
        if sid!=self.stt_stream or not self.stt: raise ProtocolError('audio stream is not active','late_packet')
        if seq!=self.stt_seq: raise ProtocolError('audio sequence mismatch','sequence_mismatch')
        a=self.audio.input
        if a.codec=='opus': validate_input_packet(payload,a); pcm=self.decoder.decode(payload,a.sample_rate*a.packet_ms//1000); validate_pcm16(pcm,a.packet_bytes)
        else: validate_pcm16(payload,a.packet_bytes); pcm=payload
        self.stt_seq+=1; self.pending.extend(pcm_resample(pcm,a.sample_rate,INPUT_RATE))
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
    async def finish_stt(self):
        r=self.stt
        if not r:return
        pcm=bytes(self.utterance+self.pending) if self.speaking else b''; self.clear_stt()
        if len(pcm)>MAX_UTTERANCE_BYTES: await self.end(r,'failed',{'code':'utterance_too_long'}); return
        if pcm: await self.send('stt.speech_end',request_id=r)
        if len(pcm)<INPUT_RATE//4*2: await self.send('stt.final',request_id=r,text=''); await self.end(r,'completed',{'transcript':''}); return
        async def run():
            try:
                text=await self.svc.asr.call('decode',pcm,timeout=90); await self.send('stt.final',request_id=r,text=text); await self.end(r,'completed',{'transcript':text})
            except asyncio.CancelledError: raise
            except Exception as e: LOG.exception('STT failed'); await self.end(r,'failed',{'code':'stt_failed','message':str(e)})
            finally: self.stt_task=None; self.stt_processing=None
        self.stt_processing=r; self.stt_task=asyncio.create_task(run())
    async def start_tts(self,d):
        if self.tts or self.tts_task or self.stt or self.stt_task: raise ProtocolError('another operation is active','busy')
        r=ident(d.get('request_id'),'request_id'); t=ident(d.get('turn_id'),'turn_id')
        if r in self.completed: await self.send('request.end',request_id=r,**self.completed[r]); return
        self.tts=r; self.tts_stream=secrets.randbelow(0xffffffff)+1; self.tts_q=asyncio.Queue(maxsize=3); self.tts_task=asyncio.create_task(self.run_tts(r,t,self.tts_stream,self.tts_q)); await self.send('tts.ready',request_id=r,turn_id=t,stream_id=self.tts_stream)
    async def tts_text(self,d):
        if d.get('request_id')!=self.tts or self.tts_finished: raise ProtocolError('TTS request is not accepting text','not_active')
        i=d.get('index'); text=d.get('text')
        if type(i)is not int or i!=self.tts_next: raise ProtocolError('TTS chunk index must be next','sequence_mismatch')
        if not isinstance(text,str) or not text.strip() or len(text)>500 or self.tts_len+len(text)>MAX_TTS_TEXT: raise ProtocolError('invalid TTS text','too_large')
        if self.tts_q.full(): raise ProtocolError('TTS queue full','busy')
        self.tts_q.put_nowait((i,text.strip())); self.tts_next+=1; self.tts_len+=len(text); await self.send('tts.accepted',request_id=self.tts,index=i)
    async def run_tts(self,r,t,s,q):
        seq=count=0
        try:
            while True:
                item=await asyncio.wait_for(q.get(),45)
                if item is None: break
                i,text=item; rate,samples=await self.svc.tts.call('synthesize',text,timeout=120); out=self.audio.output; pcm=samples_pcm(samples,rate,out.sample_rate); await self.send('tts.audio_start',request_id=r,turn_id=t,stream_id=s,index=i,sample_rate=out.sample_rate)
                for off in range(0,len(pcm),out.packet_bytes):
                    frame=pcm[off:off+out.packet_bytes].ljust(out.packet_bytes,b'\0'); payload=encode_packet(frame,out,self.encoder) if out.codec=='opus' else frame; await self.send_audio(s,seq,payload); seq+=1
                await self.send('tts.audio_end',request_id=r,stream_id=s,index=i); count+=1
            await self.end(r,'completed',{'chunks':count,'packets':seq})
        except asyncio.CancelledError: raise
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
    ws=web.WebSocketResponse(max_msg_size=8192+8)
    ws.headers['X-Voice-Replica']=os.getenv('HOSTNAME','voice-server')
    await ws.prepare(req); c=Connection(ws,req.app['service'])
    c.svc.connections.add(c)
    try:
        async for msg in ws:
            try:
                if msg.type==web.WSMsgType.TEXT: await c.control(msg.data)
                elif msg.type==web.WSMsgType.BINARY: await c.packet(msg.data)
                elif msg.type==web.WSMsgType.ERROR: break
            except (ProtocolError,WorkerBusy) as e: await c.send('error',code=getattr(e,'code','busy'),message=str(e))
            except Exception: LOG.exception('voice request failed'); await c.send('error',code='internal_error',message='voice operation failed')
    finally:
        try: await c.close()
        finally: c.svc.connections.discard(c)
    return ws
async def health(req): return web.json_response({'ok':True,'service':'voice-server','protocol':2,'stateless':True})
async def ready(req):
    return web.json_response({'ok':not req.app['service'].draining},status=503 if req.app['service'].draining else 200)
async def metrics(req):
    svc=req.app['service']
    lines=['# TYPE voice_connections gauge',f'voice_connections {sum(c.audio is not None for c in svc.connections)}',
           '# TYPE voice_draining gauge',f'voice_draining {int(svc.draining)}',
           '# TYPE voice_worker_admitted gauge','# TYPE voice_worker_running gauge',
           '# TYPE voice_worker_queued gauge','# TYPE voice_worker_rejected_total counter']
    for kind in ('vad','asr','tts'):
        pool=getattr(svc,kind)
        for metric,value in [('admitted',pool.admitted),('running',pool.running),
                             ('queued',pool.admitted-pool.running),('rejected_total',pool.rejected)]:
            lines.append(f'voice_worker_{metric}{{kind="{kind}"}} {value}')
    return web.Response(text='\n'.join(lines)+'\n',content_type='text/plain')

async def drain(service,timeout=120):
    """Stop admitting connections, finish active speech operations, close idle sockets."""
    service.draining=True
    deadline=asyncio.get_running_loop().time()+timeout
    while service.connections:
        idle=[c for c in tuple(service.connections) if not any((c.stt,c.stt_task,c.tts,c.tts_task))]
        if idle:
            await asyncio.gather(*(c.ws.close(code=1012,message=b'replica draining; reconnect') for c in idle))
        if not service.connections or asyncio.get_running_loop().time()>=deadline:
            break
        await asyncio.sleep(.05)
    await asyncio.gather(*(c.ws.close(code=1012,message=b'replica draining; reconnect') for c in tuple(service.connections)))

def create_app(service):
    if not hasattr(service,'connections'): service.connections=set()
    if not hasattr(service,'draining'): service.draining=False
    app=web.Application(); app['service']=service; root=Path(__file__).parent/'static'; app.router.add_get('/healthz',health); app.router.add_get('/readyz',ready); app.router.add_get('/metrics',metrics); app.router.add_get('/v2/voice',handle_voice); app.router.add_get('/',lambda r:web.FileResponse(root/'index.html')); app.router.add_get('/mic-worklet.js',lambda r:web.FileResponse(root/'mic-worklet.js')); return app
async def run(host='0.0.0.0',port=8765):
    svc=Service(); stop=asyncio.Event(); loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(sig,stop.set)
    runner=None
    try:
        await svc.warm(); runner=web.AppRunner(create_app(svc),handle_signals=False,shutdown_timeout=5)
        await runner.setup(); site=web.TCPSite(runner,host,port); await site.start(); LOG.info('listening on %s:%s',host,port)
        await stop.wait()
        await drain(svc)
    finally:
        if runner: await runner.cleanup()
        svc.close()
        for sig in (signal.SIGINT,signal.SIGTERM): loop.remove_signal_handler(sig)
