"""Copy only pinned serving files from an optional read-only local cache seed."""
import json,os,shutil,time
from pathlib import Path
from .workers import ASR_REPO,ASR_REVISION,MATCHA_REPO,MATCHA_REVISION

def seed(source=Path('/seed')):
    started=time.monotonic(); role=os.environ['VOICE_ROLE']; home=Path(os.getenv('HF_HOME','/models/huggingface'))
    selections=[]
    if role in ('all','gateway'):
        selections.append((Path('models--onnx-community--silero-vad'),os.environ['VAD_REVISION'],['onnx/model.onnx']))
    if role in ('all','stt'):
        selections.append((Path('hub')/('models--'+ASR_REPO.replace('/','--')),ASR_REVISION,['encoder.int8.onnx','decoder.int8.onnx','joiner.int8.onnx','tokens.txt']))
    if role in ('all','tts'):
        repo=Path('hub')/('models--'+MATCHA_REPO.replace('/','--'))
        metadata=json.loads((source/'huggingface'/repo/'snapshots'/MATCHA_REVISION/'weights/cpu-onnx/model.json').read_text())
        files=['weights/cpu-onnx/'+name for name in ['model.json','phoneme_vocab.json',*[v['name'] for v in metadata['files'].values()]]]
        files+=['scripts/matcha_hybrid_frontend.py']
        selections.append((repo,MATCHA_REVISION,files))
    count=0
    for repo,revision,files in selections:
        for name in files:
            src=source/'huggingface'/repo/'snapshots'/revision/name
            dst=home/repo/'snapshots'/revision/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.is_symlink(): dst.unlink()
            shutil.copyfile(src,dst); count+=1
    print(json.dumps({'seed_files':count,'seed_duration_s':time.monotonic()-started}),flush=True)
if __name__=='__main__': seed()
