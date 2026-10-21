"""Prepare review-only production objects. Does not apply or enable autoscaling.

Requires a versioned image reference and an existing SOPS-managed RPC Secret.
Use the local renderer for testing instead; this script never creates Secrets.
"""
import argparse,copy,importlib.util,json,os,sys
from pathlib import Path
# The local renderer is a standalone JSON generator. Execute in a subprocess;
# only non-secret application Deployment/Service/RBAC objects are carried over.
import subprocess
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--image',required=True); p.add_argument('--rpc-secret',required=True); a=p.parse_args()
if a.image.endswith(':local') or '@sha256:' not in a.image: p.error('production candidate requires a digest-pinned image')
root=Path(__file__).resolve().parents[1]
items=json.loads(subprocess.check_output([sys.executable,str(root/'local/render.py')],text=True))['items']; result=[]
for item in items:
    if item['kind']=='Secret': continue
    item['metadata']['namespace']='voice'
    if item['kind']=='Namespace': item['metadata']={'name':'voice'}
    elif item['kind']=='RoleBinding': item['subjects'][0]['namespace']='voice'
    elif item['kind']=='Deployment':
        item['spec']['replicas']=2
        pod=item['spec']['template']['spec']; role=item['spec']['template']['metadata']['labels']['voice-role']
        pod['volumes']=[v for v in pod['volumes'] if v['name']!='seed']
        init=pod['initContainers'][0]; init['command']=['/app/.venv/bin/python','-m','voice_server.prepare_models']; init['volumeMounts']=[m for m in init['volumeMounts'] if m['name']!='seed']
        for c in [init,*pod['containers']]:
            c['image']=a.image; c['imagePullPolicy']='IfNotPresent'
            for e in c['env']:
                if e['name']=='POD_NAMESPACE': e['value']='voice'
                if e['name']=='VOICE_RPC_TOKEN': e['valueFrom']['secretKeyRef']['name']=a.rpc_secret
                if e['name']=='HF_HUB_OFFLINE': e['value']='0' if c is init else '1'
    elif item['kind']=='Service':
        item['spec']['type']='ClusterIP'
        for port in item['spec']['ports']: port.pop('nodePort',None)
    result.append(item)
result.append({'apiVersion':'monitoring.coreos.com/v1','kind':'ServiceMonitor','metadata':{'name':'voice-split','namespace':'voice'},'spec':{'selector':{'matchLabels':{'voice-monitor':'true'}},'endpoints':[{'port':'http','interval':'15s'}]}})
print(json.dumps({'apiVersion':'v1','kind':'List','items':result},indent=2))
