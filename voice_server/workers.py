"""Bounded process-isolated CPU speech workers."""
from __future__ import annotations
import asyncio, concurrent.futures, multiprocessing, os
from pathlib import Path
from typing import Any
from .runtime import integer_setting, session_options
_KIND = ''; _MODEL = None; _VAD_STATES = {}
MATCHA_REPO = os.getenv('MATCHA_REPO','duyleekun/matcha-vi-en')
MATCHA_REVISION = os.getenv('MATCHA_REVISION','main')
ASR_REPO = os.getenv('ASR_REPO','g-group-ai-lab/gipformer1.5-68M-rnnt')
ASR_REVISION = os.getenv('ASR_REVISION','dd9227dcd8705c13f33bdbe59728d546ab94480f')
class WorkerBusy(RuntimeError): pass

def _find(root: Path, role: str):
    hits = [p for p in root.rglob('*.onnx') if role in p.name.lower() and ('int8' in p.name.lower() or role != 'encoder')]
    if not hits: hits=[p for p in root.rglob('*.onnx') if role in p.name.lower()]
    if not hits: raise FileNotFoundError(f'{role} ONNX not found in {root}')
    return sorted(hits)[0]
def _tokens(root):
    hits=list(root.rglob('tokens.txt'))
    if not hits: raise FileNotFoundError(f'tokens.txt not found in {root}')
    return hits[0]
def _asr_snapshot():
    from huggingface_hub import snapshot_download
    options=dict(repo_id=ASR_REPO,revision=ASR_REVISION)
    # The selected recognizer loads int8 graphs. Avoid downloading unused
    # training checkpoints and float32 copies on every new replica.
    root=Path(snapshot_download(**options,allow_patterns=['*int8*.onnx','*tokens.txt']))
    try:
        for role in ('encoder','decoder','joiner'): _find(root,role)
        _tokens(root)
    except FileNotFoundError:
        # Preserve support for other configured repositories with float graphs.
        root=Path(snapshot_download(**options,allow_patterns=['*.onnx','*tokens.txt']))
    return root
def _initialize(kind):
    global _KIND,_MODEL
    _KIND=kind
    if kind=='vad':
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        p=hf_hub_download('onnx-community/silero-vad','onnx/model.onnx',cache_dir=os.getenv('HF_HOME','/models/huggingface'))
        o=session_options(1)
        _MODEL=ort.InferenceSession(str(p),o,providers=['CPUExecutionProvider'])
    elif kind=='asr':
        import sherpa_onnx
        root=_asr_snapshot()
        enc=_find(root,'encoder'); dec=_find(root,'decoder'); join=_find(root,'joiner'); tok=_tokens(root)
        _MODEL=sherpa_onnx.OfflineRecognizer.from_transducer(encoder=str(enc),decoder=str(dec),joiner=str(join),tokens=str(tok),num_threads=integer_setting('VOICE_ASR_THREADS',2),sample_rate=16000,feature_dim=80,decoding_method='greedy_search')
        # Initialize native decode allocations before advertising readiness.
        _work('decode',bytes(16000*2))
    elif kind=='tts':
        from voice_server.matcha_onnx import MatchaOnnx, ModelSpec
        _MODEL=MatchaOnnx(ModelSpec(MATCHA_REPO,MATCHA_REVISION,'selected')); _MODEL.load()
        # Every model process must execute inference, not just load its graphs.
        _work('synthesize','Xin chào.')
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
class WorkerPool:
    def __init__(self,kind,pending):
        # VAD carries connection-local recurrent state: keep it in one process.
        # ASR/TTS calls are independent and can use separately warmed processes.
        workers=integer_setting('VOICE_'+kind.upper()+'_WORKERS',1) if kind in ('asr','tts') else 1
        if workers>pending: raise ValueError(f'{kind} workers cannot exceed its admission limit {pending}')
        self.kind=kind; self.pending=asyncio.Semaphore(pending)
        self.admitted=0; self.running=0; self.rejected=0
        self.executors=[self._new() for _ in range(workers)]
        self.available=asyncio.Queue()
        for slot in range(workers): self.available.put_nowait(slot)
    def _new(self): return concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(self.kind,))
    async def call(self,command,payload=None,timeout=30):
        try: await asyncio.wait_for(self.pending.acquire(),.05)
        except TimeoutError as e:
            self.rejected+=1
            raise WorkerBusy(f'{self.kind} queue is full') from e
        self.admitted+=1
        slot=fut=None
        try:
            async with asyncio.timeout(timeout):
                # Wait centrally instead of pre-queueing behind a busy process.
                # The next idle model gets the next request, even for varied lengths.
                slot=await self.available.get()
                fut=asyncio.get_running_loop().run_in_executor(self.executors[slot],_work,command,payload)
                self.running+=1
                def done(_):
                    self.running-=1; self.admitted-=1
                    self.available.put_nowait(slot); self.pending.release()
                # Cancellation/timeout must not free capacity while native inference is running.
                fut.add_done_callback(done)
                return await asyncio.shield(fut)
        finally:
            if fut is None:
                self.admitted-=1
                if slot is not None: self.available.put_nowait(slot)
                self.pending.release()
    async def warm(self):
        loop=asyncio.get_running_loop()
        results=await asyncio.wait_for(asyncio.gather(*(loop.run_in_executor(e,_work,'ping',None) for e in self.executors)),180)
        if any(result!=self.kind for result in results): raise RuntimeError('worker mismatch')
    def close(self):
        for executor in self.executors: executor.shutdown(wait=False,cancel_futures=True)
