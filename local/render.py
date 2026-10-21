"""Render local-only objects without touching the production overlay."""
import json,os,secrets
from pathlib import Path
NS='voice-split-bench'; items=[]
def add(kind,name,spec=None,api='v1',**kw):
    obj={'apiVersion':api,'kind':kind,'metadata':{'name':name,'namespace':NS},**kw}
    if spec is not None: obj['spec']=spec
    items.append(obj); return obj
add('Namespace',NS)['metadata'].pop('namespace')
state=Path(__file__).resolve().parents[1]/'.local'; state.mkdir(exist_ok=True)
tokenfile=state/'rpc-token'
if not tokenfile.exists(): tokenfile.write_text(secrets.token_urlsafe(32)); tokenfile.chmod(0o600)
add('Secret','voice-rpc',stringData={'token':tokenfile.read_text().strip()})
add('ServiceAccount','voice-gateway')
add('Role','voice-discovery',api='rbac.authorization.k8s.io/v1',rules=[{'apiGroups':['discovery.k8s.io'],'resources':['endpointslices'],'verbs':['get','list','watch']}])
add('RoleBinding','voice-discovery',api='rbac.authorization.k8s.io/v1',roleRef={'apiGroup':'rbac.authorization.k8s.io','kind':'Role','name':'voice-discovery'},subjects=[{'kind':'ServiceAccount','name':'voice-gateway','namespace':NS}])
profiles={'gateway':('500m','256Mi','1Gi'),'stt':('500m','896Mi','2Gi'),'tts':('1','3712Mi','4Gi')}
for role,(cpu,memory,limit) in profiles.items():
    labels={'app':'voice-'+role,'voice-role':role}
    env=[{'name':k,'value':v} for k,v in {'VOICE_ROLE':role,'HF_HOME':'/models/huggingface','MATCHA_REVISION':'4f741be47b95f6ee1b23e15ba34f12c302fa64b7','VAD_REVISION':'e71cae966052b992a7eca6b17738916ce0eca4ec','ASR_REVISION':'dd9227dcd8705c13f33bdbe59728d546ab94480f','VOICE_ORT_SPINNING':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','POD_NAMESPACE':NS}.items()]
    env.append({'name':'VOICE_RPC_TOKEN','valueFrom':{'secretKeyRef':{'name':'voice-rpc','key':'token'}}})
    cold=os.getenv('VOICE_LOCAL_MODEL_MODE','seed')=='cold'
    prep_command=('/app/.venv/bin/python -m voice_server.seed_models && ' if not cold else '')+'/app/.venv/bin/python -m voice_server.prepare_models'
    init={'name':'models','image':'voice-server:local','imagePullPolicy':'Never','env':env,'command':['/bin/sh','-c',prep_command],'volumeMounts':[{'name':'models','mountPath':'/models'},{'name':'seed','mountPath':'/seed','readOnly':True}], 'resources':{'requests':{'cpu':'100m','memory':'128Mi'},'limits':{'cpu':'1','memory':'1Gi'}}}
    init['env']=[e if e['name']!='HF_HUB_OFFLINE' else {'name':'HF_HUB_OFFLINE','value':'0' if cold else '1'} for e in env]
    container={'name':'voice','image':'voice-server:local','imagePullPolicy':'Never','env':env,'ports':[{'name':'http','containerPort':8765}],'resources':{'requests':{'cpu':cpu,'memory':memory},'limits':{'cpu':'2','memory':limit}},'volumeMounts':[{'name':'models','mountPath':'/models'},{'name':'tmp','mountPath':'/tmp'}],'readinessProbe':{'httpGet':{'path':'/readyz','port':'http'},'periodSeconds':2},'startupProbe':{'httpGet':{'path':'/readyz','port':'http'},'periodSeconds':2,'failureThreshold':180},'livenessProbe':{'httpGet':{'path':'/healthz','port':'http'},'periodSeconds':15},'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}}}
    pod={'automountServiceAccountToken':role=='gateway','terminationGracePeriodSeconds':150,'securityContext':{'runAsUser':10001,'runAsGroup':10001,'fsGroup':10001},'nodeSelector':{'kubernetes.io/hostname':'k3d-voice-local-agent-0' if role=='gateway' else 'k3d-voice-local-agent-1'},'initContainers':[init],'containers':[container],'volumes':[{'name':'models','emptyDir':{'sizeLimit':'4Gi'}},{'name':'tmp','emptyDir':{'sizeLimit':'256Mi'}},{'name':'seed','hostPath':{'path':'/model-seed','type':'Directory'}}]}
    # Prefer spreading replicas, allowing either worker node; control-plane remains tainted.
    pod.pop('nodeSelector'); pod['affinity']={'podAntiAffinity':{'preferredDuringSchedulingIgnoredDuringExecution':[{'weight':100,'podAffinityTerm':{'labelSelector':{'matchLabels':labels},'topologyKey':'kubernetes.io/hostname'}}]}}
    if role=='gateway': pod['serviceAccountName']='voice-gateway'
    replicas=int(os.getenv('VOICE_LOCAL_REPLICAS','1'))
    if replicas not in (1,2): raise ValueError('local replicas must be 1 or 2')
    add('Deployment','voice-'+role,{'replicas':replicas,'selector':{'matchLabels':labels},'strategy':{'type':'RollingUpdate','rollingUpdate':{'maxSurge':1,'maxUnavailable':0}},'template':{'metadata':{'labels':labels,'annotations':{'prometheus.io/scrape':'true','prometheus.io/port':'8765'}},'spec':pod}},api='apps/v1')
    port={'name':'http','port':8765,'targetPort':'http'}
    if role=='gateway': port['nodePort']=30080
    service=add('Service','voice-'+role,{'selector':labels,'ports':[port],'type':'NodePort' if role=='gateway' else 'ClusterIP'})
    service['metadata']['labels']={'voice-monitor':'true','voice-role':role}
print(json.dumps({'apiVersion':'v1','kind':'List','items':items}))
