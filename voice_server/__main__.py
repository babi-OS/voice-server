import argparse,asyncio,logging
from .server import run
p=argparse.ArgumentParser(); p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=8765); a=p.parse_args(); logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s'); asyncio.run(run(a.host,a.port))
