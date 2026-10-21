"""Real native STT CPU accounting, including work after decode returns."""
import json,os,time,wave
from pathlib import Path
from voice_server import workers

def threads():
    result={}
    for entry in Path('/proc/self/task').iterdir():
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            result[entry.name]={'cpu_s':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),'state':fields[0],'name':(entry/'comm').read_text().strip()}
        except OSError:pass
    return result
fixture=Path(os.getenv('VOICE_PROFILE_FIXTURE',str(Path(__file__).resolve().parents[1]/'tests/fixtures/vi_weather_two_clauses.wav')))
with wave.open(str(fixture)) as f:pcm=f.readframes(f.getnframes())
workers._initialize('asr')
for _ in range(3):workers._work('decode',pcm)
time.sleep(2)
report={'threads_setting':int(os.environ['VOICE_ASR_THREADS']),'spinning_setting':os.environ.get('VOICE_ORT_SPINNING'),'idle_before':threads()}
cpu=time.process_time(); time.sleep(1);report['idle_cpu_s_per_second']=time.process_time()-cpu
before=threads(); start=time.monotonic(); cpu=time.process_time(); text=workers._work('decode',pcm)
report['single_call']={'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'transcript':text}
returned=time.monotonic(); cpu=time.process_time(); tails=[]
for due in (.02,.05,.1,.2,.5,1.):
    time.sleep(max(0,returned+due-time.monotonic()));tails.append({'after_return_s':due,'cpu_s':time.process_time()-cpu,'threads':threads()})
report['post_return']=tails;report['threads_before_call']=before
for cadence,count in [(1.,12),(0.,40)]:
    started=time.monotonic();cpu=time.process_time();calls=[]
    for i in range(count):
        time.sleep(max(0,started+i*cadence-time.monotonic()))
        a=time.monotonic();c=time.process_time();result=workers._work('decode',pcm)
        assert result==text
        calls.append({'wall_s':time.monotonic()-a,'cpu_s':time.process_time()-c})
    # Include native worker CPU after the final returned result.
    time.sleep(1)
    report['paced' if cadence else 'continuous']={'count':count,'elapsed_s':time.monotonic()-started,'cpu_s':time.process_time()-cpu,'calls':calls}
print(json.dumps(report,indent=2),flush=True)
