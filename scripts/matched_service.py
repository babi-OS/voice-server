"""Supervise monolith or split processes under ONE Docker CPU/memory cgroup."""
import os,signal,subprocess,sys,time,urllib.request
children=[]
def start(role,port,**extra):
    env=os.environ|{'VOICE_ROLE':role}|extra
    children.append(subprocess.Popen([sys.executable,'-m','voice_server','--host','0.0.0.0','--port',str(port)],env=env))
def stop(*_):
    for child in children: child.send_signal(signal.SIGTERM)
    for child in children:
        try: child.wait(timeout=150)
        except subprocess.TimeoutExpired: child.kill()
    sys.exit(0)
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
if os.getenv('MATCHED_MODE')=='monolith': start('all',8765,VOICE_ASR_WORKERS='2',VOICE_TTS_WORKERS='2')
else:
    counts={'stt':int(os.getenv('MATCHED_STT_WORKERS','2')),'tts':int(os.getenv('MATCHED_TTS_WORKERS','2'))}
    if any(not 1<=n<=4 for n in counts.values()): raise ValueError('bounded matched worker count: 1..4 per stage')
    ports=[base+i for stage,base in [('stt',19000),('tts',19100)] for i in range(1,counts[stage]+1)]
    for stage,base in [('stt',19000),('tts',19100)]:
        for i in range(1,counts[stage]+1): start(stage,base+i,VOICE_ASR_WORKERS='1',VOICE_TTS_WORKERS='1')
    # Only launch public gateway after all native workers are warmed.
    for port in ports:
        for _ in range(360):
            try: urllib.request.urlopen(f'http://127.0.0.1:{port}/readyz',timeout=1); break
            except OSError: time.sleep(.5)
        else: stop()
    start('gateway',8765,**{'VOICE_'+stage.upper()+'_URLS':','.join('http://127.0.0.1:'+str(base+i) for i in range(1,counts[stage]+1)) for stage,base in [('stt',19000),('tts',19100)]})
while all(c.poll() is None for c in children): time.sleep(.2)
stop()
