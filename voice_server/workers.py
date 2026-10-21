"""Bounded process-isolated CPU speech workers."""
from __future__ import annotations
import asyncio, concurrent.futures, multiprocessing, os, time, contextvars
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any
from .runtime import integer_setting, session_options
CALL_TIMING=contextvars.ContextVar('voice_call_timing',default=None)
_WARM_SECONDS=1.0
_KIND = ''; _MODEL = None; _VAD_STATES = {}
MATCHA_REPO = os.getenv('MATCHA_REPO','duyleekun/matcha-vi-en')
MATCHA_REVISION = os.getenv('MATCHA_REVISION','main')
ASR_REPO = os.getenv('ASR_REPO','g-group-ai-lab/gipformer1.5-68M-rnnt')
ASR_REVISION = os.getenv('ASR_REVISION','dd9227dcd8705c13f33bdbe59728d546ab94480f')
class WorkerBusy(RuntimeError):
    code = 'overloaded'
    def __init__(self, message, stage=None):
        super().__init__(message); self.stage=stage; self.retry_after_ms=1000

def _find(root: Path, role: str):
    hits = [p for p in root.rglob('*.onnx') if role in p.name.lower() and ('int8' in p.name.lower() or role != 'encoder')]
    if not hits: hits=[p for p in root.rglob('*.onnx') if role in p.name.lower()]
    if not hits: raise FileNotFoundError(f'{role} ONNX not found in {root}')
    return sorted(hits)[0]
def _tokens(root):
    hits=list(root.rglob('tokens.txt'))
    if not hits: raise FileNotFoundError(f'tokens.txt not found in {root}')
    return hits[0]
def _asr_snapshot(**download_options):
    from huggingface_hub import snapshot_download
    options=dict(repo_id=ASR_REPO,revision=ASR_REVISION,**download_options)
    # The selected recognizer loads int8 graphs. Avoid downloading unused
    # training checkpoints and float32 copies on every new replica.
    if ASR_REPO=='g-group-ai-lab/gipformer1.5-68M-rnnt' and ASR_REVISION=='dd9227dcd8705c13f33bdbe59728d546ab94480f':
        options['allow_patterns']=['encoder.int8.onnx','decoder.int8.onnx','joiner.int8.onnx','tokens.txt']
    # Other repositories/revisions retain the original full-snapshot behavior,
    # including any external-data files their ONNX graphs may reference.
    return Path(snapshot_download(**options))
def _initialize(kind):
    global _KIND,_MODEL,_WARM_SECONDS
    _KIND=kind
    if kind=='vad':
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        p=hf_hub_download('onnx-community/silero-vad','onnx/model.onnx',revision=os.getenv('VAD_REVISION','main'),cache_dir=os.getenv('HF_HOME','/models/huggingface'))
        o=session_options(1)
        _MODEL=ort.InferenceSession(str(p),o,providers=['CPUExecutionProvider'])
    elif kind=='asr':
        import sherpa_onnx
        root=_asr_snapshot()
        enc=_find(root,'encoder'); dec=_find(root,'decoder'); join=_find(root,'joiner'); tok=_tokens(root)
        _MODEL=sherpa_onnx.OfflineRecognizer.from_transducer(encoder=str(enc),decoder=str(dec),joiner=str(join),tokens=str(tok),num_threads=integer_setting('VOICE_ASR_THREADS',2),sample_rate=16000,feature_dim=80,decoding_method='greedy_search')
        # Initialize native decode allocations before advertising readiness.
        warm_started=time.monotonic(); _work('decode',bytes(16000*2)); _WARM_SECONDS=time.monotonic()-warm_started
    elif kind=='tts':
        from voice_server.matcha_onnx import MatchaOnnx, ModelSpec
        _MODEL=MatchaOnnx(ModelSpec(MATCHA_REPO,MATCHA_REVISION,'selected')); _MODEL.load()
        # Every model process must execute inference, not just load its graphs.
        warm_started=time.monotonic(); _work('synthesize','Xin chào.'); _WARM_SECONDS=time.monotonic()-warm_started
    else: raise ValueError(kind)
def _work(command,payload):
    if command=='ping': return _KIND
    if _KIND=='vad':
        import numpy as np
        if command in ('reset','release'):
            _VAD_STATES.pop(payload,None); return None
        key,pcm=payload; audio=np.frombuffer(pcm,dtype='<i2').astype('float32').reshape(1,-1)/32768
        if audio.shape[1]!=512: raise ValueError('Silero requires 512 samples')
        state,ctx=_VAD_STATES.get(key,(np.zeros((2,1,128),np.float32),np.zeros((1,64),np.float32)))
        features=np.concatenate((ctx,audio),axis=1)
        prob,next_state=_MODEL.run(None,{'input':features,'state':state,'sr':np.asarray(16000,dtype=np.int64)})
        _VAD_STATES[key]=(next_state,features[:,-64:]); return float(prob[0,0])
    if _KIND=='asr' and command=='decode':
        import numpy as np
        stream=_MODEL.create_stream(); stream.accept_waveform(16000,np.frombuffer(payload,dtype='<i2').astype('float32')/32768); _MODEL.decode_stream(stream); return stream.result.text.strip()
    if _KIND=='tts' and command=='synthesize':
        out=_MODEL.synthesize(payload)
        if not out.samples: raise RuntimeError('empty TTS output')
        return out.sample_rate,out.samples
    raise ValueError(command)
def _calibration():
    return _WARM_SECONDS

def _timed_work(command,payload):
    wall=time.monotonic(); cpu=time.process_time()
    result=_work(command,payload)
    return result,time.monotonic()-wall,time.process_time()-cpu

class WorkerPool:
    def __init__(self,kind,pending):
        # VAD carries connection-local recurrent state: keep it in one process.
        # ASR/TTS calls are independent and can use separately warmed processes.
        workers=integer_setting('VOICE_'+kind.upper()+'_WORKERS',1) if kind in ('asr','tts') else 1
        if workers>pending: raise ValueError(f'{kind} workers cannot exceed its admission limit {pending}')
        self.kind=kind; self.pending=asyncio.Semaphore(pending)
        self.admitted=0; self.running=0; self.rejected=0; self.offered=0
        self.healthy=True
        self.queued_estimates={}
        self.rejection_reasons=Counter()
        self.estimates={}; self.estimate=1.0; self.current_estimate=1.0; self.started=None; self.last_queue=0.; self.last_inference=0.
        from .telemetry import Telemetry
        self.telemetry=Telemetry()
        self.executors=[self._new() for _ in range(workers)]
        self.available=asyncio.Queue()
        for slot in range(workers): self.available.put_nowait(slot)
    def _new(self): return concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(self.kind,))
    async def call(self,command,payload=None,timeout=30,queue_budget=None):
        entered=time.monotonic(); self.offered+=1
        bucket=min(3,len(payload)//(16000*2*5 if self.kind=='asr' else 100)) if isinstance(payload,(bytes,str)) else 0
        estimate=self.estimates.get(bucket,self.estimate)
        forecast=(max(0.,self.current_estimate-(entered-(self.started or entered))) if self.running else 0.)+sum(self.queued_estimates.values())
        if queue_budget is not None:
            if self.pending.locked() or forecast>queue_budget:
                self.rejected+=1
                self.rejection_reasons['full' if self.pending.locked() else 'predicted_wait']+=1
                raise WorkerBusy(f'{self.kind} queue budget exceeded',self.kind)
            # No task handoff between checking the bound and reserving capacity.
            await self.pending.acquire()
        else:
            try: await asyncio.wait_for(self.pending.acquire(),.05)
            except TimeoutError as e:
                self.rejected+=1; self.rejection_reasons['full']+=1
                raise WorkerBusy(f'{self.kind} queue is full') from e
        self.admitted+=1
        ticket=object(); self.queued_estimates[ticket]=estimate
        slot=fut=None
        try:
            async with asyncio.timeout(timeout):
                # Wait centrally instead of pre-queueing behind a busy process.
                # The next idle model gets the next request, even for varied lengths.
                try:
                    slot=await asyncio.wait_for(self.available.get(),max(.001,queue_budget-(time.monotonic()-entered))) if queue_budget is not None else await self.available.get()
                except TimeoutError as exc:
                    self.rejected+=1; self.rejection_reasons['deadline']+=1
                    self.telemetry.observe('expired_queue_wait',time.monotonic()-entered,'stt' if self.kind=='asr' else self.kind)
                    raise WorkerBusy(f'{self.kind} queue deadline exceeded',self.kind) from exc
                self.queued_estimates.pop(ticket,None)
                self.last_queue=time.monotonic()-entered
                queue_wait=self.last_queue
                self.telemetry.observe('queue_wait',queue_wait,'stt' if self.kind=='asr' else self.kind)
                started=time.monotonic(); self.started=started; self.current_estimate=estimate
                try: fut=asyncio.get_running_loop().run_in_executor(self.executors[slot],_timed_work,command,payload)
                except BrokenProcessPool:
                    self.healthy=False; raise
                self.running+=1
                def done(_):
                    if not _.cancelled() and isinstance(_.exception(),BrokenProcessPool): self.healthy=False
                    self.last_inference=time.monotonic()-started
                    self.estimate=.2*self.last_inference+.8*self.estimate
                    self.estimates[bucket]=.2*self.last_inference+.8*self.estimates.get(bucket,self.last_inference)
                    self.telemetry.observe('inference',self.last_inference,'stt' if self.kind=='asr' else self.kind)
                    if not _.cancelled() and _.exception() is None:
                        self.telemetry.observe('native_cpu',_.result()[2],'stt' if self.kind=='asr' else self.kind)
                    self.running-=1; self.admitted-=1
                    self.available.put_nowait(slot); self.pending.release()
                # Cancellation/timeout must not free capacity while native inference is running.
                fut.add_done_callback(done)
                result=await asyncio.shield(fut)
                CALL_TIMING.set((queue_wait,time.monotonic()-started))
                return result[0]
        finally:
            if fut is None:
                self.queued_estimates.pop(ticket,None)
                self.admitted-=1
                if slot is not None: self.available.put_nowait(slot)
                self.pending.release()
    async def warm(self):
        loop=asyncio.get_running_loop()
        results=await asyncio.wait_for(asyncio.gather(*(loop.run_in_executor(e,_work,'ping',None) for e in self.executors)),180)
        if any(result!=self.kind for result in results): raise RuntimeError('worker mismatch')
        if self.kind in ('asr','tts'):
            calibration=await asyncio.gather(*(loop.run_in_executor(e,_calibration) for e in self.executors))
            self.estimate=max(.001,sum(calibration)/len(calibration))
            self.estimates[0]=self.estimate
    def close(self):
        for executor in self.executors: executor.shutdown(wait=False,cancel_futures=True)
