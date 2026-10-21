#!/usr/bin/env python3
"""Reproduce paired real-model benchmarks under one shared 4 CPU/6 GiB limit."""
import argparse,json,os,secrets,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAME='voice-local-matched'; IMAGE='voice-server:local'
def run(args,**kw): return subprocess.run(args,check=True,**kw)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--cpuset'); p.add_argument('--turns',type=int,default=40); p.add_argument('--users',type=int,default=8); p.add_argument('--output',default=str(ROOT/'.local/results/matched')); a=p.parse_args()
    if not 1<=a.turns<=100 or not 1<=a.users<=32: p.error('invalid bounded benchmark size')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if subprocess.run(['docker','inspect',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: sys.exit('Named matched container already exists; stop/destroy the previous run explicitly')
    token=secrets.token_urlsafe(24)
    image='voice-server:matched-'+secrets.token_hex(6)
    run(['docker','tag',IMAGE,image])
    metadata={'image_tag':image,'image_id':subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip(),'limits':{'cpu':4,'memory_gib':6},'workers_per_stage':2,'threads_per_worker':2}
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for mode in ('monolith','split'):
        args=['docker','run','-d','--name',NAME,'--cpus','4','--memory','6g','--memory-swap','6g','-p','127.0.0.1:18766:8765','-v','voice-server_voice-models:/models:ro','-v',str(ROOT/'scripts')+':/benchmark:ro']
        if a.cpuset: args+=['--cpuset-cpus',a.cpuset]
        for key,value in {'HF_HOME':'/models/huggingface','HF_HUB_OFFLINE':'1','MATCHA_REVISION':'4f741be47b95f6ee1b23e15ba34f12c302fa64b7','VAD_REVISION':'e71cae966052b992a7eca6b17738916ce0eca4ec','VOICE_ORT_SPINNING':'0','MATCHED_MODE':mode,'VOICE_RPC_TOKEN':token}.items(): args+=['-e',key+'='+value]
        run(args+['--entrypoint','/app/.venv/bin/python',image,'/benchmark/matched_service.py'],stdout=subprocess.DEVNULL)
        try:
            for _ in range(180):
                try: urllib.request.urlopen('http://127.0.0.1:18766/readyz',timeout=1); break
                except OSError: time.sleep(1)
            else: raise RuntimeError('matched service did not become ready')
            run([sys.executable,str(ROOT/'scripts/bench_local.py'),'--url','ws://127.0.0.1:18766/v2/voice','--users',str(a.users),'--turns',str(a.turns),'--container',NAME,'--output',str(out/mode)])
        finally:
            with (out/(mode+'-service.log')).open('w') as f: subprocess.run(['docker','logs',NAME],stdout=f,stderr=f)
            run(['docker','stop','-t','150',NAME],stdout=subprocess.DEVNULL); run(['docker','rm',NAME],stdout=subprocess.DEVNULL)
    run(['docker','image','rm',image],stdout=subprocess.DEVNULL)
if __name__=='__main__': main()
