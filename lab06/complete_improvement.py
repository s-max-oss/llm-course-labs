"""Select a development-only recipe, then run a fixed final evaluation protocol."""
import json
import subprocess
import sys
from pathlib import Path

import hw6_rsi_localize as h


def select(root):
    candidates=[]
    for name,compact in [('seed_extended_cn',False),('balanced_cn',False),('balanced_compact',True)]:
        folder=root/'improvement/ablation_v1'/name; path=folder/'validation.json'
        if path.exists():
            s=h.read_json(path);candidates.append({'adapter':str((folder/'adapter').relative_to(root)),
                         'compact':compact,'correct':s['correct'],'n':s['n']})
    for name,checkpoints in [('vision_compact_v1',(400,1200)),('augmented_v1',(1000,2500,4000))]:
        for step in checkpoints:
            folder=root/'improvement'/name/f'step{step}'; path=folder/'validation.json'
            if path.exists():
                s=h.read_json(path);candidates.append({'adapter':str((folder/'adapter').relative_to(root)),
                         'compact':True,'correct':s['correct'],'n':s['n']})
    assert candidates and all(c['n']==128 for c in candidates)
    selected=max(candidates,key=lambda c:c['correct']/c['n'])
    return {'rule':'max correct/128 on fixed development localization; first candidate wins ties',
            'candidates_in_tie_order':candidates,'selected':selected,'final_test_evaluated':False}


def main():
    root=Path(__file__).resolve().parent
    choice=select(root);h.write_json(root/'improvement/recipe_selection.json',choice)
    selected=choice['selected'];print('[recipe] '+json.dumps(selected),flush=True)
    command=[sys.executable,str(root/'finish_grounding.py'),'--model',str(Path.home()/'llm_course_lab05/model'),
             '--adapter',selected['adapter'],'--offline','--augmented-pool']
    if not selected['compact']:command.append('--loc-cn')
    subprocess.run(command,cwd=root,check=True)


if __name__=='__main__':main()
