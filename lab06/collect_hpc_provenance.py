"""Record Slurm completion and released allocations for the improvement study."""
import hashlib
import subprocess
from pathlib import Path

import hw6_rsi_localize as h

JOBS=[15392,15393,15394,15395,15396,15397]


def ssh(command):
    return subprocess.run(['ssh','-o','BatchMode=yes','-p','28754',
                           'u2024311281@10.251.171.6',command],
                          capture_output=True,text=True,encoding='utf-8',check=True).stdout


def main():
    root=Path(__file__).resolve().parent;folder=root/'provenance'
    raw=ssh('sacct -j '+','.join(map(str,JOBS))+' --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed,ElapsedRaw,Start,End,NodeList,AllocTRES')
    (folder/'sacct_improvement.log').write_bytes(raw.encode('utf-8'))
    records=[]
    keys=['job_id','state','exit_code','elapsed','elapsed_seconds','start','end','node','allocation']
    for line in raw.splitlines():
        values=line.split('|')
        if not values[0].isdigit() or int(values[0]) not in JOBS:continue
        record=dict(zip(keys,values));record['job_id']=int(record['job_id']);record['elapsed_seconds']=int(record['elapsed_seconds'])
        records.append(record)
    assert len(records)==len(JOBS)
    final=next(r for r in records if r['job_id']==15397)
    assert final['state']=='COMPLETED' and final['exit_code']=='0:0'
    queue=ssh('squeue -u u2024311281 --noheader --format="%i|%j|%T|%M"')
    (folder/'queue_after_improvement.log').write_bytes(queue.encode('utf-8'))
    assert not any(line.split('|')[0] in set(map(str,JOBS)) for line in queue.splitlines())
    runtime=root/'improvement/final_v1/source_at_run.py'
    out={**final,'final_job':15397,'jobs':records,'gpu_job_minutes':sum(r['elapsed_seconds'] for r in records)/60,
         'queue_after':[line for line in queue.splitlines() if line.strip()],
         'experiment_allocations_released':True,'runtime_source_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),
         'submitted_source_sha256':hashlib.sha256((root/'finish_grounding.py').read_bytes()).hexdigest(),
         'source_annotation_note':'Only answer_origin provenance categories refined after runtime snapshot; AST comparison and raw pool backups in bootstrap_origins.json',
         'cpu_reference':'improvement/cpu_reference/metadata.json'}
    h.write_json(folder/'improvement_run.json',out)
    print(f'All experiment allocations released; final job completed 0:0; GPU job minutes={out["gpu_job_minutes"]:.2f}')


if __name__=='__main__':main()
