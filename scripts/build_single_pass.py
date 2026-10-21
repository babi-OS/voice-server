#!/usr/bin/env python3
"""Build only the serving source and a checksummed single-pass ONNX bundle."""
import argparse,hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',required=True,type=Path);p.add_argument('--tag',required=True);a=p.parse_args()
    metadata=json.loads((a.model_dir/'model.json').read_text())
    if metadata.get('steps')!=1 or metadata.get('architecture')!='single':raise SystemExit('expected the selected single-pass export')
    files={'phoneme_vocab.json':metadata['vocab_sha256'],**{item['name']:item['sha256'] for item in metadata['files'].values()}}
    for name,digest in files.items():
        if Path(name).name!=name or hashlib.sha256((a.model_dir/name).read_bytes()).hexdigest()!=digest:raise SystemExit('bundle checksum mismatch: '+name)
    with tempfile.TemporaryDirectory(prefix='voice-single-pass-build-') as d:
        context=Path(d);shutil.copytree(ROOT/'voice_server',context/'voice_server',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (context/'model').mkdir()
        for name in [*files,'model.json']:shutil.copyfile(a.model_dir/name,context/'model'/name)
        shutil.copyfile(ROOT/'deploy/Dockerfile.single-pass',context/'Dockerfile')
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        subprocess.run(['docker','build','--build-arg','SOURCE_REVISION='+revision,'-t',a.tag,str(context)],check=True)
if __name__=='__main__':main()
