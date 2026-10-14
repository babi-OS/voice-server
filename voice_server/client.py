from __future__ import annotations
import argparse,asyncio,json,secrets
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from websockets.asyncio.client import connect
from .codec import encode_packet,make_encoder,validate_input_packet
from .protocol import encode_control,pack_audio,unpack_audio,validate_hello,validate_pcm16
AUDIO={'input':{'codec':'opus','sample_rate':16000,'frame_ms':20,'packet_ms':20,'max_packet_bytes':1275,'bitrate':24000,'complexity':3,'application':'restricted_lowdelay'},'output':{'codec':'opus','sample_rate':24000,'frame_ms':20,'packet_ms':20,'max_packet_bytes':1275,'bitrate':24000,'complexity':3,'application':'audio'}}
def read(path,rate):
 a,r=sf.read(path,dtype='float32'); a=a.mean(1) if a.ndim==2 else a
 if r!=rate:
  d=__import__('math').gcd(r,rate); a=resample_poly(a,rate//d,r//d)
 return (np.clip(a,-1,1)*32767).astype('<i2').tobytes()
async def run(url,wav,text,out):
 import opuslib_next
 p=validate_hello({'type':'hello','version':2,'audio':AUDIO}); ia,oa=p.input,p.output; enc=make_encoder(ia); dec=opuslib_next.Decoder(oa.sample_rate,1); rid=secrets.token_hex(8); turn=secrets.token_hex(8); got=bytearray(); seq=0
 async with connect(url,max_size=10000) as ws:
  await ws.send(encode_control('hello',version=2,audio=p.wire())); assert json.loads(await ws.recv())['type']=='ready'
  kind='stt' if wav else 'tts'; await ws.send(encode_control(kind+'.start',request_id=rid,turn_id=turn)); started=json.loads(await ws.recv()); sid=started['stream_id']
  if wav:
   pcm=read(wav,ia.sample_rate)
   for n,o in enumerate(range(0,len(pcm),ia.packet_bytes)):
    frame=pcm[o:o+ia.packet_bytes].ljust(ia.packet_bytes,b'\0'); await ws.send(pack_audio(sid,n,encode_packet(frame,ia,enc))); await asyncio.sleep(ia.packet_ms/1000)
   await ws.send(encode_control('stt.finish',request_id=rid))
  else:
   for n in range(0,len(text),500): await ws.send(encode_control('tts.text',request_id=rid,index=n//500,text=text[n:n+500])); await ws.recv()
   await ws.send(encode_control('tts.finish',request_id=rid))
  while 1:
   m=await ws.recv()
   if isinstance(m,bytes):
    s,n,payload=unpack_audio(m); assert s==sid and n==seq; seq+=1; validate_input_packet(payload,oa); got.extend(dec.decode(payload,oa.sample_rate*oa.packet_ms//1000))
   else:
    e=json.loads(m); print(json.dumps(e,ensure_ascii=False))
    if e.get('type')=='request.end': break
 if got: sf.write(out,np.frombuffer(got,dtype='<i2'),oa.sample_rate); print('saved',out)
def main():
 p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--wav',type=Path); g.add_argument('--text'); p.add_argument('--url',default='ws://127.0.0.1:8765/v2/voice'); p.add_argument('--output',type=Path,default=Path('reply.wav')); a=p.parse_args(); asyncio.run(run(a.url,a.wav,a.text,a.output))
if __name__=='__main__': main()
