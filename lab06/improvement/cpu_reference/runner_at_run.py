"""Run the complete required 10-step reference flow using CPU FP32 only."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8);args=p.parse_args()
    root=Path(__file__).resolve().parent;output=root/'improvement/cpu_reference'
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a clean CPU reference directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'runner_at_run.py').write_bytes(Path(__file__).read_bytes())
    env={**os.environ,'OMP_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),
         'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','PYTHONIOENCODING':'utf-8'}
    rows=[];started=time.perf_counter()
    with (output/'full_run.log').open('w',encoding='utf-8') as log:
        for mode,extra in [('make_data',[]),('make_sft',[]),('infer',[]),('sft',['--steps','10']),('eval',[]),('rsi',['--steps','10','--rounds','2'])]:
            command=[sys.executable,str(root/'hw6_rsi_localize.py'),'--mode',mode,'--root',str(output),
                     '--model',str(args.model.resolve()),'--device','cpu','--no-quant','--offline',*extra]
            log.write('[command] '+' '.join(command)+'\n');log.flush();tick=time.perf_counter()
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            rows.append({'mode':mode,'exit_code':result.returncode,'seconds':time.perf_counter()-tick})
            print(f'[CPU] {mode} exit={result.returncode} seconds={rows[-1]["seconds"]:.1f}',flush=True)
            if result.returncode:break
    import torch,transformers,peft
    metadata={'device':'cpu','dtype':'float32','nf4':False,'threads':args.threads,
              'date':datetime.datetime.now().astimezone().isoformat(),'python':sys.version,
              'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
              'stages':rows,'seconds':time.perf_counter()-started,'success':len(rows)==6 and all(r['exit_code']==0 for r in rows),
              'source_sha256':hashlib.sha256((root/'hw6_rsi_localize.py').read_bytes()).hexdigest()}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not metadata['success']:raise SystemExit(1)
    print('[complete] CPU reference chain',flush=True)


if __name__=='__main__':main()
