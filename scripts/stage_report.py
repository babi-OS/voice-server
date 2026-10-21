"""Query fixed-label stage and Pod telemetry from LOCAL Prometheus only."""
import argparse,json,urllib.parse,urllib.request
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--window',default='5m'); a=p.parse_args()
queries={
 'queue_p95_s':'histogram_quantile(0.95,sum by(stage,le)(rate(voice_stage_queue_wait_seconds_bucket{namespace="voice-split-bench"}[WINDOW])))',
 'inference_mean_s':'sum by(stage)(rate(voice_stage_inference_seconds_sum{namespace="voice-split-bench"}[WINDOW])) / sum by(stage)(rate(voice_stage_inference_seconds_count{namespace="voice-split-bench"}[WINDOW]))',
 'inference_p95_s':'histogram_quantile(0.95,sum by(stage,le)(rate(voice_stage_inference_seconds_bucket{namespace="voice-split-bench"}[WINDOW])))',
 'native_cpu_cores':'sum by(stage)(rate(voice_stage_native_cpu_seconds_sum{namespace="voice-split-bench"}[WINDOW]))',
 'event_loop_p99_s':'histogram_quantile(0.99,sum by(stage,le)(rate(voice_stage_event_loop_delay_seconds_bucket{namespace="voice-split-bench"}[WINDOW])))',
 'worker_distribution':'sum by(pod,stage)(increase(voice_stage_inference_seconds_count{namespace="voice-split-bench",stage=~"stt|tts"}[WINDOW]))',
 'pod_cpu_cores':'sum by(pod)(rate(container_cpu_usage_seconds_total{namespace="voice-split-bench",container="voice"}[WINDOW]))',
 'pod_working_set_bytes':'sum by(pod)(container_memory_working_set_bytes{namespace="voice-split-bench",container="voice"})',
 'pod_throttle_ratio':'sum by(pod)(rate(container_cpu_cfs_throttled_periods_total{namespace="voice-split-bench",container="voice"}[WINDOW])) / sum by(pod)(rate(container_cpu_cfs_periods_total{namespace="voice-split-bench",container="voice"}[WINDOW]))',
}
report={}
for name,query in queries.items():
 query=query.replace('WINDOW',a.window)
 with urllib.request.urlopen('http://127.0.0.1:19090/api/v1/query?'+urllib.parse.urlencode({'query':query}),timeout=10) as response: report[name]={'query':query,'result':json.load(response)['data']['result']}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
