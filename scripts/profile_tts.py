"""Real-model TTS stage CPU/wall profile; no serving/runtime changes."""
import json,os,time
from voice_server.matcha_onnx import MatchaOnnx,ModelSpec
from voice_server import vocos_istft
TEXT='Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài.'
metrics={}
def measured(name,fn):
 def call(*args,**kw):
  start=time.monotonic();cpu=time.process_time();result=fn(*args,**kw)
  m=metrics.setdefault(name,{'n':0,'wall_s':0.,'cpu_s':0.});m['n']+=1;m['wall_s']+=time.monotonic()-start;m['cpu_s']+=time.process_time()-cpu
  return result
 return call
class Session:
 def __init__(self,name,wrapped):self.name=name;self.wrapped=wrapped
 def __getattr__(self,name):return getattr(self.wrapped,name)
 def run(self,*args,**kw):return measured(self.name,self.wrapped.run)(*args,**kw)
model=MatchaOnnx(ModelSpec('duyleekun/matcha-vi-en',os.environ['MATCHA_REVISION'],'selected'));model.load()
for _ in range(3):model.synthesize(TEXT)
time.sleep(1)
for name in ('encoder','decoder','vocoder'):setattr(model,name,Session(name,getattr(model,name)))
model._ids=measured('frontend',model._ids);vocos_istft.istft_same=measured('istft',vocos_istft.istft_same)
started=time.monotonic();cpu=time.process_time();lengths=[]
for _ in range(20):
 out=model.synthesize(TEXT);assert out.samples and any(out.samples);lengths.append(len(out.samples))
report={'threads':int(os.environ['VOICE_TTS_THREADS']),'count':20,'wall_s':time.monotonic()-started,'cpu_s':time.process_time()-cpu,'steps':model.steps,'samples_lengths':sorted(set(lengths)),'stages':metrics}
report['unattributed_cpu_s']=report['cpu_s']-sum(v['cpu_s'] for v in metrics.values());print(json.dumps(report,indent=2))
