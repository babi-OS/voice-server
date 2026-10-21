"""Save bounded-duration cgroup and Kubernetes/HPA snapshots, no credentials."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--seconds',type=int,default=1800); p.add_argument('--output',required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; k=['kubectl','--kubeconfig',str(root/'.local/kubeconfig'),'--context','k3d-voice-local']
with open(a.output,'w') as out:
    for _ in range(a.seconds//5):
        row={'timestamp':time.time()}
        for name,args in [('hpa',k+['get','hpa','-n','voice-split-bench','-o','json']),('pods',k+['get','pods','-n','voice-split-bench','-o','json']),('container', ['docker','exec','voice-local-matched','sh','-c','cat /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak'])]:
            result=subprocess.run(args,capture_output=True,text=True,timeout=10)
            if result.returncode==0: row[name]=json.loads(result.stdout) if name!='container' else result.stdout
        out.write(json.dumps(row)+'\n'); out.flush(); time.sleep(5)
