import asyncio
from voice_server.protocol import validate_hello,ProtocolError,pack_audio,unpack_audio
from voice_server.server import create_app
from aiohttp.test_utils import TestClient,TestServer
PROFILE={'input':{'codec':'pcm16','sample_rate':16000,'frame_ms':20,'packet_ms':20},'output':{'codec':'pcm16','sample_rate':24000,'frame_ms':20,'packet_ms':20}}
def test_protocol_has_no_session_fields():
 p=validate_hello({'type':'hello','version':2,'audio':PROFILE}); assert p.wire()==PROFILE
 try: validate_hello({'type':'hello','version':2,'audio':PROFILE,'session_id':'x'})
 except ProtocolError: pass
 else: raise AssertionError('session field accepted')
def test_binary_header():
 x=pack_audio(4,7,b'abc'); assert unpack_audio(x)==(4,7,b'abc')
def test_health_stateless():
 async def go():
  class S: pass
  app=create_app(S()); server=TestServer(app); await server.start_server(); c=TestClient(server); await c.start_server(); r=await c.get('/healthz'); assert r.status==200 and (await r.json())['stateless']; await c.close(); await server.close()
 asyncio.run(go())
