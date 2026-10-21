"""Real CPU VAD equivalence: original sequential path versus gateway coalescer."""
import asyncio,json,wave
from pathlib import Path
import numpy as np
from voice_server import workers
from voice_server.gateway import VadThread

async def check():
    pool=VadThread(); await pool.warm(); reports=[]
    with wave.open(str(Path(__file__).resolve().parents[1]/'tests/fixtures/vi_weather_two_clauses.wav')) as f: pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')
    try:
        for n in (1,8,16):
            streams=[]
            for i in range(n):
                audio=np.roll(pcm,i*128).copy()
                if i%4==3: audio[:]=0
                elif i%4==2: audio=(audio.astype(float)*.2).astype('<i2')
                streams.append([audio[o:o+512].tobytes() for o in range(0,len(audio)-511,512)])
            baseline=[]
            for j in range(len(streams[0])):
                baseline.append([workers._work('feed',('base-'+str(i),streams[i][j])) for i in range(n)])
            actual=[]
            for j in range(len(streams[0])):
                actual.append(await asyncio.gather(*(pool.call('feed',('batch-'+str(i),streams[i][j])) for i in range(n))))
            a=np.asarray(baseline); b=np.asarray(actual)
            assert np.array_equal(a,b)
            reports.append({'streams':n,'probabilities':a.size,'max_difference':float(np.max(abs(a-b)))})
            for i in range(n):
                workers._work('release','base-'+str(i)); await pool.call('release','batch-'+str(i))
        print(json.dumps(reports,indent=2))
    finally: await pool.aclose()
if __name__=='__main__': asyncio.run(check())
