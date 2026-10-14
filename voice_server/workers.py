"""Bounded process-isolated CPU speech workers."""
from __future__ import annotations
import asyncio, concurrent.futures, multiprocessing, os
from pathlib import Path
from typing import Any
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
def _initialize(kind):
    global _KIND,_MODEL
    _KIND=kind
    if kind=='vad':
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        p=hf_hub_download('onnx-community/silero-vad','onnx/model.onnx',cache_dir=os.getenv('HF_HOME','/models/huggingface'))
        o=ort.SessionOptions(); o.intra_op_num_threads=1; o.inter_op_num_threads=1
        _MODEL=ort.InferenceSession(str(p),o,providers=['CPUExecutionProvider'])
    elif kind=='asr':
        from huggingface_hub import snapshot_download
        import sherpa_onnx
        root=Path(snapshot_download(ASR_REPO,revision=ASR_REVISION))
        enc=_find(root,'encoder'); dec=_find(root,'decoder'); join=_find(root,'joiner'); tok=_tokens(root)
        _MODEL=sherpa_onnx.OfflineRecognizer.from_transducer(encoder=str(enc),decoder=str(dec),joiner=str(join),tokens=str(tok),num_threads=2,sample_rate=16000,feature_dim=80,decoding_method='greedy_search')
    elif kind=='tts':
        from voice_server.matcha_onnx import MatchaOnnx, ModelSpec
        _MODEL=MatchaOnnx(ModelSpec(MATCHA_REPO,MATCHA_REVISION,'selected')); _MODEL.load()
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
        self.kind=kind; self.pending=asyncio.Semaphore(pending); self.executor=self._new()
    def _new(self): return concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(self.kind,))
    async def call(self,command,payload=None,timeout=30):
        try: await asyncio.wait_for(self.pending.acquire(),.05)
        except TimeoutError as e: raise WorkerBusy(f'{self.kind} queue is full') from e
        try: fut=asyncio.get_running_loop().run_in_executor(self.executor,_work,command,payload)
        except Exception: self.pending.release(); raise
        fut.add_done_callback(lambda _: self.pending.release())
        return await asyncio.wait_for(asyncio.shield(fut),timeout)
    async def warm(self):
        if await self.call('ping',timeout=180)!=self.kind: raise RuntimeError('worker mismatch')
    def close(self): self.executor.shutdown(wait=False,cancel_futures=True)
