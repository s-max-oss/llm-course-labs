"""Record timed-out training, completed evaluation-only resume, and release."""
import hashlib
from pathlib import Path
import hw6_rsi_localize as h
from collect_hpc_provenance import ssh


def main():
    root = Path(__file__).resolve().parent; out = root/'provenance'
    raw = ssh('sacct -j 15408,15427 --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed,ElapsedRaw,Start,End,NodeList,AllocTRES')
    (out/'sacct_optimization_v2.log').write_bytes(raw.encode('utf-8'))
    keys = ['job_id','state','exit_code','elapsed','elapsed_seconds','start','end','node','allocation']
    jobs = []
    for line in raw.splitlines():
        fields = line.split('|')
        if fields[0] not in ('15408','15427'): continue
        record = dict(zip(keys, fields)); record['job_id'] = int(record['job_id']); record['elapsed_seconds'] = int(record['elapsed_seconds'])
        jobs.append(record)
    assert len(jobs) == 2
    evaluation = next(j for j in jobs if j['job_id'] == 15427)
    assert evaluation['state'] == 'COMPLETED' and evaluation['exit_code'] == '0:0'
    queue = ssh('squeue -u u2024311281 --noheader --format="%i|%j|%T|%M"')
    (out/'queue_after_optimization_v2.log').write_bytes(queue.encode('utf-8'))
    assert not any(line.split('|')[0] in ('15408','15427') for line in queue.splitlines())
    selection = root/'improvement/final_v2/selection.json'
    resume = h.read_json(root/'improvement/final_v2/evaluation_resume.json')
    assert hashlib.sha256(selection.read_bytes()).hexdigest() == resume['selection_sha256']
    result = {'training_job': 15408, 'evaluation_job': 15427, 'jobs': jobs,
              'evaluation_state': evaluation['state'], 'evaluation_exit_code': evaluation['exit_code'],
              'gpu_job_minutes': sum(j['elapsed_seconds'] for j in jobs)/60,
              'training_restarted': False, 'selection_unchanged': True,
              'selection_sha256': resume['selection_sha256'],
              'queue_after': queue.splitlines(), 'allocations_released': True,
              'clock_note': 'Slurm timestamps retained as reported; no timezone assumed.'}
    h.write_json(out/'optimization_v2_run.json', result)
    print(result)


if __name__ == '__main__': main()
