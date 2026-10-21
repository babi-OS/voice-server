"""Fixed-label, bounded Prometheus histograms; never retain individual timings."""
from collections import defaultdict

BUCKETS=(.001,.005,.01,.025,.05,.1,.25,.5,1.,2.,2.05,2.25,5.,10.,30.,120.)
class Telemetry:
    def __init__(self): self.values={}
    def observe(self,name,value,stage):
        key=(name,stage)
        counts,total,n=self.values.setdefault(key,([0]*len(BUCKETS),0.,0))
        for i,bound in enumerate(BUCKETS): counts[i]+=value<=bound
        self.values[key]=(counts,total+value,n+1)
    def render(self):
        lines=[]
        for (name,stage),(counts,total,n) in self.values.items():
            metric='voice_stage_'+name+'_seconds'
            for bound,count in zip(BUCKETS,counts): lines.append(f'{metric}_bucket{{stage="{stage}",le="{bound}"}} {count}')
            lines.extend([f'{metric}_bucket{{stage="{stage}",le="+Inf"}} {n}',f'{metric}_sum{{stage="{stage}"}} {total}',f'{metric}_count{{stage="{stage}"}} {n}'])
        return lines
