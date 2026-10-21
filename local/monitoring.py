"""Local Prometheus + the production-pinned adapter, without alert delivery."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from importlib.machinery import SourceFileLoader
l=SourceFileLoader('l',str(Path(__file__).resolve().parents[1]/'scripts/voice-local')).load_module()
def helm(name,chart,version,values):
    path=l.STATE/(name+'-values.json'); path.write_text(json.dumps(values))
    l.run(['helm','upgrade','--install',name,chart,'--repo','https://prometheus-community.github.io/helm-charts','--version',version,'--namespace','voice-monitoring','--create-namespace','--kubeconfig',str(l.KUBE),'--kube-context','k3d-voice-local','-f',str(path),'--wait','--timeout','300s'])
helm('voice-monitoring','kube-prometheus-stack','92.10.1',{
 'alertmanager':{'enabled':False},'grafana':{'enabled':False},'nodeExporter':{'enabled':False},
 'defaultRules':{'create':False},'kubeEtcd':{'enabled':False},'kubeControllerManager':{'enabled':False},'kubeScheduler':{'enabled':False},
 'prometheus':{'prometheusSpec':{'retention':'2h','scrapeInterval':'5s','serviceMonitorSelectorNilUsesHelmValues':False,'resources':{'requests':{'cpu':'100m','memory':'256Mi'},'limits':{'cpu':'1','memory':'1Gi'}}}}})
l.kubectl('apply','-f','-',input=json.dumps({'apiVersion':'monitoring.coreos.com/v1','kind':'ServiceMonitor','metadata':{'name':'voice','namespace':l.NS},'spec':{'selector':{'matchLabels':{'voice-monitor':'true'}},'endpoints':[{'port':'http','interval':'5s'}]}}),text=True)
for role in ('gateway','stt','tts'): l.kubectl('-n',l.NS,'label','service','voice-'+role,'voice-monitor=true','--overwrite')
helm('voice-adapter','prometheus-adapter','5.3.0',{'prometheus':{'url':'http://voice-monitoring-kube-prom-prometheus.voice-monitoring.svc','port':9090},'metricsRelistInterval':'15s','rules':{'default':False,'custom':[
 {'seriesQuery':'voice_stage_admitted{namespace="voice-split-bench",pod!=""}','resources':{'overrides':{'namespace':{'resource':'namespace'},'pod':{'resource':'pod'}}},'name':{'matches':'^voice_stage_admitted$','as':'voice_stage_admitted'},'metricsQuery':'sum(<<.Series>>{<<.LabelMatchers>>,stage=~"stt|tts"}) by (<<.GroupBy>>)'},
 {'seriesQuery':'voice_active_microphones{namespace="voice-split-bench",pod!=""}','resources':{'overrides':{'namespace':{'resource':'namespace'},'pod':{'resource':'pod'}}},'name':{'matches':'^voice_active_microphones$','as':'voice_active_microphones'},'metricsQuery':'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'}]},'resources':{'requests':{'cpu':'100m','memory':'128Mi'},'limits':{'cpu':'500m','memory':'256Mi'}}})
