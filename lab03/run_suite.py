"""Run CPU experiments with a bounded number of processes inside one allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

from min_llm import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'results')
    a=p.parse_args()
    a.output=a.output.resolve()
    if a.workers<1 or a.threads<1:raise ValueError('workers and threads must be positive')
    allocation=int(os.environ.get('SLURM_CPUS_PER_TASK',os.cpu_count() or 1))
    if a.workers*a.threads>allocation:raise ValueError('Requested threads exceed CPU allocation')
    a.output.mkdir(parents=True,exist_ok=True)
    names=['baseline','exp3_large','exp2_deep','exp1_lr_high','exp1_lr_low',
           'exp2_shallow','exp3_small','exp4_short','exp5_no_pos']
    def run(name):
        with (a.output/f'{name}.log').open('w',encoding='utf-8') as log:
            code=subprocess.call([sys.executable,'-u',str(ROOT/'min_llm.py'),'--experiment',name,
                                  '--threads',str(a.threads),'--output',str(a.output/name)],
                                  stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        return name,code
    failed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,n) for n in names]
        for f in as_completed(futures):
            name,code=f.result();print(name,'exit',code,flush=True)
            if code:failed.append(name)
    if failed:raise RuntimeError(f'Failed experiments: {failed}; see individual logs')
    subprocess.check_call([sys.executable,str(ROOT/'analyze_results.py'),'--results',str(a.output),
                           '--threads',str(a.threads)],cwd=ROOT)


if __name__=='__main__':main()
