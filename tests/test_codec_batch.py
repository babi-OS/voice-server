import numpy as np
import pytest
from voice_server.codec import encode_batch,encode_packet,make_encoder,MAX_ENCODE_BATCH
from voice_server.protocol import AudioDirection

@pytest.mark.parametrize('rate',[8000,16000,24000,48000])
def test_codec_batches_preserve_every_wire_packet_and_partial_tail(rate):
    audio=AudioDirection('opus',rate,20,20,1275,24000,3,'audio')
    raw=(np.sin(np.arange(rate//3)*2*np.pi*440/rate)*12000).astype('<i2').tobytes()
    a,b=make_encoder(audio),make_encoder(audio)
    expected=[encode_packet(raw[o:o+audio.packet_bytes].ljust(audio.packet_bytes,b'\0'),audio,a) for o in range(0,len(raw),audio.packet_bytes)]
    actual=[]
    for off in range(0,len(raw),audio.packet_bytes*MAX_ENCODE_BATCH):
        actual.extend(encode_batch(raw[off:off+audio.packet_bytes*MAX_ENCODE_BATCH],audio,b))
    assert actual==expected
    assert encode_batch(b'',audio,b)==[]
    with pytest.raises(ValueError):encode_batch(bytes(audio.packet_bytes*(MAX_ENCODE_BATCH+1)),audio,b)


def test_cancelled_codec_job_finishes_before_connection_reuses_state():
    import asyncio,threading
    from types import SimpleNamespace
    from voice_server.gateway import CodecExecutor
    from voice_server.server import Connection
    async def check():
        codec=CodecExecutor();entered=threading.Event();release=threading.Event();writes=[]
        async def convert(fn,*args):return await codec.call(fn,*args)
        connection=Connection(None,SimpleNamespace(codec=codec,convert=convert))
        def blocked_batch():entered.set();release.wait(2);writes.append('first')
        first=asyncio.create_task(connection.convert(blocked_batch));second=None
        try:
            assert await asyncio.to_thread(entered.wait,1)
            first.cancel()
            second=asyncio.create_task(connection.convert(lambda:writes.append('second')))
            await asyncio.sleep(.02)
            assert not first.done() and not second.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):await first
            await second
            assert writes==['first','second']
        finally:
            release.set();await asyncio.gather(*(t for t in (first,second) if t),return_exceptions=True);codec.close()
    asyncio.run(check())
