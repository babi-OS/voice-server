"""Bounded gateway executors and per-operation HTTP worker routing."""
from __future__ import annotations
import asyncio
import concurrent.futures
import json
import os
import random
import ssl
import time
from pathlib import Path
import aiohttp
from . import workers
from .workers import WorkerBusy
from .telemetry import Telemetry

class CodecExecutor:
    def __init__(self):
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=2,thread_name_prefix='voice-codec')
        self.slots=asyncio.Semaphore(10)
        self.telemetry=Telemetry()
    @staticmethod
    def _run(fn,args):
        started=time.monotonic(); cpu=time.thread_time()
        result=fn(*args)
        return result,time.monotonic()-started,time.thread_time()-cpu
    async def call(self,fn,*args):
        entered=time.monotonic()
        await self.slots.acquire()
        self.telemetry.observe('queue_wait',time.monotonic()-entered,'codec')
        try: future=asyncio.get_running_loop().run_in_executor(self.executor,self._run,fn,args)
        except BaseException:
            self.slots.release(); raise
        future.add_done_callback(lambda _:self.slots.release())
        result,wall,cpu=await asyncio.shield(future)
        self.telemetry.observe('inference',wall,'codec'); self.telemetry.observe('native_cpu',cpu,'codec')
        return result
    def close(self): self.executor.shutdown(wait=False,cancel_futures=True)

class VadThread:
    """One native VAD owner; coalescing changes scheduling, never model math."""
    def __init__(self):
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=1,initializer=workers._initialize,initargs=('vad',),thread_name_prefix='voice-vad')
        self.queue=asyncio.Queue(maxsize=64); self.task=None
        self.admitted=self.running=self.rejected=self.offered=0
        self.telemetry=Telemetry()
    async def warm(self):
        await asyncio.get_running_loop().run_in_executor(self.executor,workers._work,'ping',None)
        self.task=asyncio.create_task(self._serve())
    async def call(self,command,payload=None,timeout=5):
        self.offered+=1
        future=asyncio.get_running_loop().create_future()
        item=(command,payload,future,time.monotonic())
        try:
            if command in ('reset','release'): await asyncio.wait_for(self.queue.put(item),timeout)
            else: self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            self.rejected+=1; raise WorkerBusy('VAD queue full','vad') from exc
        self.admitted+=1
        try: return await asyncio.wait_for(asyncio.shield(future),timeout)
        except (TimeoutError,asyncio.CancelledError):
            future.cancel(); raise
    @staticmethod
    def _batch(batch):
        cpu=time.thread_time(); results=[]
        for command,payload,_,_ in batch:
            try: results.append((True,workers._work(command,payload)))
            except Exception as exc: results.append((False,exc))
        return results,time.thread_time()-cpu
    async def _serve(self):
        while True:
            batch=[await self.queue.get()]
            await asyncio.sleep(.001)
            while len(batch)<16 and not self.queue.empty(): batch.append(self.queue.get_nowait())
            self.running=len(batch)
            for *_,entered in batch: self.telemetry.observe('queue_wait',time.monotonic()-entered,'vad')
            started=time.monotonic()
            results,cpu=await asyncio.get_running_loop().run_in_executor(self.executor,self._batch,batch)
            self.telemetry.observe('native_cpu',cpu,'vad')
            self.telemetry.observe('inference',time.monotonic()-started,'vad')
            for (_,_,future,_),(ok,value) in zip(batch,results):
                self.admitted-=1
                if not future.done():
                    if ok: future.set_result(value)
                    else: future.set_exception(value)
            self.running=0
    async def aclose(self):
        if self.task:
            self.task.cancel(); await asyncio.gather(self.task,return_exceptions=True)
        self.executor.shutdown(wait=True,cancel_futures=True)
    def close(self): self.executor.shutdown(wait=False,cancel_futures=True)

class Discovery:
    def __init__(self,session,token):
        self.session=session; self.token=token; self.endpoints={'stt':[],'tts':[]}; self.capacity={}; self.inflight={}; self.failed={}
        self.last=0.; self.task=None; self.watch_task=None; self.resource_version=None; self.force_relist=True
        self.namespace=os.getenv('POD_NAMESPACE','voice-split-bench')
        self.static={s:[u.rstrip('/') for u in os.getenv('VOICE_'+s.upper()+'_URLS','').split(',') if u] for s in self.endpoints}
        # Explicit URLs also support a supervised, loopback-only stack inside a
        # pod without a service-account token. Kubernetes injects SERVICE_HOST
        # even when automountServiceAccountToken is false.
        self.kube=bool(os.getenv('KUBERNETES_SERVICE_HOST')) and not any(self.static.values())
    async def start(self):
        await self.refresh(); self.task=asyncio.create_task(self._loop())
        if self.kube: self.watch_task=asyncio.create_task(self._watch())
    async def _loop(self):
        while True:
            await asyncio.sleep(.25+random.random()*.05)
            await self.refresh()
    async def refresh(self):
        try:
            if self.kube and (self.force_relist or time.monotonic()-self.last>=5):
                root=Path('/var/run/secrets/kubernetes.io/serviceaccount')
                context=ssl.create_default_context(cafile=str(root/'ca.crt'))
                url=f'https://{os.environ["KUBERNETES_SERVICE_HOST"]}:{os.getenv("KUBERNETES_SERVICE_PORT","443")}/apis/discovery.k8s.io/v1/namespaces/{self.namespace}/endpointslices'
                async with self.session.get(url,headers={'Authorization':'Bearer '+(root/'token').read_text().strip()},ssl=context,timeout=aiohttp.ClientTimeout(total=2)) as response:
                    response.raise_for_status(); body=await response.json(); slices=body['items']; self.resource_version=body['metadata']['resourceVersion']
                found={'stt':[],'tts':[]}
                for item in slices:
                    stage=item['metadata'].get('labels',{}).get('kubernetes.io/service-name','').removeprefix('voice-')
                    if stage not in found: continue
                    port=next((p['port'] for p in item['ports'] if p.get('name')=='http'),8765)
                    for endpoint in item['endpoints']:
                        conditions=endpoint.get('conditions',{})
                        if conditions.get('ready') is not True or conditions.get('terminating'): continue
                        for address in endpoint['addresses']:
                            found[stage].append(f'http://[{address}]:{port}' if ':' in address else f'http://{address}:{port}')
                self.endpoints={s:list(set(urls)) for s,urls in found.items()}; self.last=time.monotonic(); self.force_relist=False
            elif not self.kube:
                self.endpoints=self.static; self.last=time.monotonic()
        except (aiohttp.ClientError,TimeoutError,OSError,KeyError): pass
        active={u for urls in self.endpoints.values() for u in urls}
        retained=active|{u for u,n in self.inflight.items() if n}
        self.capacity={u:v for u,v in self.capacity.items() if u in retained}
        self.failed={u:v for u,v in self.failed.items() if u in retained}
        self.inflight={u:n for u,n in self.inflight.items() if u in retained}
        async def capacity(url):
            try:
                async with self.session.get(url+'/capacity',headers={'Authorization':'Bearer '+self.token},timeout=aiohttp.ClientTimeout(total=.2)) as response:
                    response.raise_for_status(); self.capacity[url]=(time.monotonic(),await response.json())
            except (aiohttp.ClientError,TimeoutError,ValueError): pass
        await asyncio.gather(*(capacity(u) for urls in self.endpoints.values() for u in urls))
    async def _watch(self):
        root=Path('/var/run/secrets/kubernetes.io/serviceaccount')
        context=ssl.create_default_context(cafile=str(root/'ca.crt'))
        url=f'https://{os.environ["KUBERNETES_SERVICE_HOST"]}:{os.getenv("KUBERNETES_SERVICE_PORT","443")}/apis/discovery.k8s.io/v1/namespaces/{self.namespace}/endpointslices'
        while True:
            try:
                params={'watch':'true','timeoutSeconds':'20','allowWatchBookmarks':'true'}
                if self.resource_version: params['resourceVersion']=self.resource_version
                async with self.session.get(url,params=params,headers={'Authorization':'Bearer '+(root/'token').read_text().strip()},ssl=context,timeout=aiohttp.ClientTimeout(total=25)) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        event=json.loads(line)
                        if event.get('type')=='ERROR':
                            self.resource_version=None; self.force_relist=True; await self.refresh(); break
                        self.resource_version=event.get('object',{}).get('metadata',{}).get('resourceVersion',self.resource_version)
                        if event.get('type')!='BOOKMARK':
                            self.force_relist=True; await self.refresh()
            except (aiohttp.ClientError,TimeoutError,ValueError,OSError):
                await asyncio.sleep(1)
                self.force_relist=True; await self.refresh()

    def choose(self,stage,excluded):
        now=time.monotonic()
        if now-self.last>30: raise WorkerBusy('worker discovery stale',stage)
        urls=[u for u in self.endpoints[stage] if u not in excluded and now-self.failed.get(u,0)>1]
        if not urls: raise WorkerBusy('no ready '+stage+' worker',stage)
        def score(url):
            stamp,report=self.capacity.get(url,(0,{}))
            local=self.inflight.get(url,0)
            remote=report.get('admitted',0) if now-stamp<=1 else 0
            return (max(remote,local)+local)*report.get('estimated_service_s',1.)
        # Model fleets are deliberately small (production maximum: eight TTS).
        # Inspect every Ready worker so a sampled busy pair cannot hide an idle one.
        # Random ties keep independent gateways from converging on the first URL.
        return min(urls,key=lambda url:(score(url),random.random()))
    async def close(self):
        tasks=[t for t in (self.task,self.watch_task) if t]
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)

class RemotePool:
    def __init__(self,stage,discovery):
        self.kind=self.stage=stage; self.discovery=discovery
        self.admitted=self.running=self.rejected=self.offered=0; self.telemetry=Telemetry()
    async def call(self,command,payload=None,timeout=120):
        started=time.monotonic(); excluded=set(); self.offered+=1; self.admitted+=1; self.running+=1
        try:
            for attempt in range(2):
                url=self.discovery.choose(self.stage,excluded); excluded.add(url)
                remaining=max(0.,2-(time.monotonic()-started))
                headers={'Authorization':'Bearer '+self.discovery.token,'X-Queue-Budget-Ms':str(int(remaining*1000))}
                kwargs={'data':payload,'headers':headers} if self.stage=='stt' else {'json':{'text':payload,'sample_rate':24000},'headers':headers}
                self.discovery.inflight[url]=self.discovery.inflight.get(url,0)+1
                try:
                    async with self.discovery.session.post(url+'/internal/v1/'+self.stage,**kwargs,timeout=aiohttp.ClientTimeout(total=timeout,sock_connect=.2)) as response:
                        if response.status in (429,503):
                            report=await response.json()
                            if report.get('not_started') and attempt==0 and time.monotonic()-started<2: continue
                            raise WorkerBusy(report.get('code','worker unavailable'),self.stage)
                        response.raise_for_status()
                        if self.stage=='stt': result=(await response.json())['text']
                        else:
                            pcm=await response.read(); result=int(response.headers['X-Audio-Sample-Rate']),pcm
                        return result
                except (aiohttp.ClientConnectorError,aiohttp.ConnectionTimeoutError):
                    self.discovery.failed[url]=time.monotonic()
                    if attempt or time.monotonic()-started>=2: raise WorkerBusy('worker unavailable',self.stage)
                except (aiohttp.ClientError,TimeoutError):
                    # Cool down ambiguous failures, but never replay their operation.
                    self.discovery.failed[url]=time.monotonic(); raise
                finally: self.discovery.inflight[url]-=1
            raise WorkerBusy('worker unavailable',self.stage)
        except WorkerBusy:
            self.rejected+=1; raise
        finally:
            self.admitted-=1; self.running-=1; self.telemetry.observe('rpc',time.monotonic()-started,self.stage)
    async def warm(self): pass
    def close(self): pass
