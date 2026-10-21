"""Local-only HPA: never rendered or applied by the default deploy command."""
import json
items=[]
for role in ('gateway','stt','tts'):
    metric='voice_active_microphones' if role=='gateway' else 'voice_stage_admitted'
    items.append({'apiVersion':'autoscaling/v2','kind':'HorizontalPodAutoscaler','metadata':{'name':'voice-'+role,'namespace':'voice-split-bench'},'spec':{'scaleTargetRef':{'apiVersion':'apps/v1','kind':'Deployment','name':'voice-'+role},'minReplicas':1,'maxReplicas':2,'metrics':[{'type':'Pods','pods':{'metric':{'name':metric},'target':{'type':'AverageValue','averageValue':'16' if role=='gateway' else '700m'}}},{'type':'Resource','resource':{'name':'cpu','target':{'type':'AverageValue','averageValue':'1000m' if role=='gateway' else '1200m'}}}],'behavior':{'scaleUp':{'stabilizationWindowSeconds':0,'policies':[{'type':'Pods','value':2,'periodSeconds':30}]},'scaleDown':{'stabilizationWindowSeconds':300,'policies':[{'type':'Pods','value':1,'periodSeconds':60}]}}}})
print(json.dumps({'apiVersion':'v1','kind':'List','items':items}))
