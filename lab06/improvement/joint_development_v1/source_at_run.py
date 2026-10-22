"""Continue selected grounding recipe with all tasks, two RSI rounds, final test.

Recipe/checkpoint selection uses development validation only. The final-test
images are read for inference only after all stages and selection are finished.
"""
import argparse
import gc
import hashlib
import random
import re
import time
from pathlib import Path

from PIL import Image
import hw6_rsi_localize as h
import improve_localization as exp


def task_prompt(q, compact):
    if q['type']=='loc':
        return exp.prompt(q['target'], compact)
    name=exp.EN_COLORS[q['target'][0]]+' '+('circle' if q['target'][1]=='圆' else 'square')
    return (f'How many {name}s are in the image? Answer with one integer only.' if q['type']=='count'
            else f'Is there a {name} in the image? Answer yes or no only.')


def training_answer(q, compact):
    if q['type']=='loc': return exp.response(q,compact)
    return q['gold'].removesuffix('个') if q['type']=='count' else 'yes' if q['gold']=='有' else 'no'


def make_joint_data(root):
    data=h.read_json(root/'improvement/data/splits.json')
    folder=root/'improvement/joint_data'; folder.mkdir(parents=True,exist_ok=True)
    joint={}
    for split,repetitions,seed in [('train',2,20261025),('validation',1,20261026),('final_test',1,20261027)]:
        rng=random.Random(seed); rows=list(data[split]); images=folder/split; images.mkdir(exist_ok=True)
        for cls in exp.CLASSES:
            for n in range(5):
                for rep in range(repetitions):
                    items=[]; extra=rng.randint(max(0,2-n),4-n)
                    cells=rng.sample(range(16),n+extra)
                    for index,position in enumerate(cells):
                        target=cls if index<n else rng.choice([c for c in exp.CLASSES if c!=cls])
                        row,col=divmod(position,4); side=rng.randint(30,48)
                        x=col*64+rng.randint(6,58-side); y=row*64+rng.randint(6,58-side)
                        items.append({'class':target,'box':[x,y,x+side,y+side]})
                    identity=f'{exp.CLASSES.index(cls)}_{n}_{rep}'
                    image=images/f'{identity}.png'; h.render_scene(items,image)
                    record={'id':f'{split}_count_{identity}','type':'count','target':cls,
                            'image':image.relative_to(root).as_posix(),'gold':f'{n}个','box':None,'objects':items}
                    rows.append(record)
                    if n in (0,2):
                        rows.append({**record,'id':f'{split}_exist_{identity}','type':'exist','gold':'有' if n else '没有'})
        joint[split]=rows
    # Enforce image isolation; sharing an image between two tasks within a split is intended.
    seen={}
    for split,rows in joint.items():
        for relative in {q['image'] for q in rows}:
            digest=hashlib.sha256(Image.open(root/relative).convert('RGB').tobytes()).hexdigest()
            assert digest not in seen or seen[digest]==split
            seen[digest]=split
    h.write_json(folder/'questions.json',joint)
    h.write_json(folder/'pixel_split_hashes.json',seen)
    return joint


def canonical(q,text):
    if q['type']=='count' and not re.fullmatch(r'\s*\d+\s*(?:个)?[。.!！]?\s*',text):
        return None
    if q['type']=='exist' and not re.fullmatch(r'\s*(?:yes|no|有|没有|不存在|无)[。.!！]?\s*',text,re.I):
        return None
    return h.parsed_answer(q,text)


def observations(image,q):
    regions=[o['box'] for o in h.components(image,q['target'][0]) if o['class']==q['target']]
    return f'Detected matching target bounding boxes [left,top,right,bottom] in pixels: {regions}. Image size is 256 by 256; each of the four rows/columns is 64 pixels wide. Derive the requested answer from these observations.'


def check(image,q,text,detail=True):
    expected,evidence=h.pixel_answer(image,q)
    candidate=canonical(q,text); passed=expected is not None and candidate==expected
    if expected is None:
        hint='The visual checker cannot uniquely identify the target. Recheck the image.'
    elif candidate is None:
        hint='Your answer has an invalid format. Recheck the image and use only the requested answer format.'
    elif q['type']=='loc':
        selected=h.cell_box(h.parse_cell(candidate)); box=evidence['box']
        intersection=max(0,min(selected[2],box[2])-max(selected[0],box[0]))*max(0,min(selected[3],box[3])-max(selected[1],box[1]))
        coverage=intersection/((box[2]-box[0])*(box[3]-box[1]))
        hint=f'The selected cell {"passes" if passed else "fails"} the target-center check; it covers {coverage:.0%} of the target box. Recheck row and column.'
    else:
        hint=f'Your {q["type"]} answer {"passes" if passed else "fails"} the independent target-color and shape check. Recheck the image.'
    if detail and expected is not None:
        # Detector-like observations, not the answer string/grid label. This
        # richer evidence condition is reported separately from pass/fail only.
        hint += ' '+observations(image,q)
    return {'passed':passed,'pixel_result':expected,'parsed':candidate,'evidence':evidence,'hint':hint}


def infer(processor,model,device,root,q,compact,folder=None,correction=False,detail=True):
    image=exp.load_image(root,q); question=task_prompt(q,compact)
    raw=h.answer(processor,model,device,image,question,24)
    verification=check(image,q,raw,detail); final=raw; self_check=None; tool_check=None
    if correction and not verification['passed']:
        candidate=h.parse_cell(raw) if q['type']=='loc' else None
        if candidate is not None:
            folder.mkdir(parents=True,exist_ok=True)
            marked=h.overlay(image,candidate,folder/f"{q['id']}.png")
            revised=h.answer(processor,model,device,marked,question+' The purple box marks your previous choice. Check the target, and correct the answer if needed.',24)
            self_check={'prediction':revised,'verification':check(image,q,revised)}
            if self_check['verification']['passed']: final=revised
        if not check(image,q,final)['passed']:
            revised=h.answer(processor,model,device,image,question+' Checker feedback: '+verification['hint'],24)
            tool_check={'prediction':revised,'verification':check(image,q,revised)}; final=revised
    return {'id':q['id'],'type':q['type'],'target':q['target'],'image':q['image'],
            'gold':q['gold'],'raw':raw,'raw_correct':canonical(q,raw)==q['gold'],
            'verification':verification,'self_check':self_check,'tool_check':tool_check,
            'final':final,'correct':canonical(q,final)==q['gold'],'valid':canonical(q,final) is not None,
            'final_verified':check(image,q,final)['passed'],
            'iou':h.loc_iou(q,final) if q['type']=='loc' else None}


def evaluate(processor,model,device,root,questions,compact,name,output,correction=False,detail=True):
    started=time.perf_counter()
    rows=[infer(processor,model,device,root,q,compact,output/'overlays'/name,correction,detail) for q in questions]
    summary={tp:{'correct':sum(r['correct'] for r in rows if r['type']==tp),
                 'n':sum(r['type']==tp for r in rows),'valid':sum(r['valid'] for r in rows if r['type']==tp)}
             for tp in ('loc','count','exist')}
    summary['total']={'correct':sum(r['correct'] for r in rows),'n':len(rows)}
    summary['loc_iou']=sum(r['iou'] for r in rows if r['iou'] is not None)/max(1,summary['loc']['n'])
    result={'name':name,'summary':summary,'seconds':time.perf_counter()-started,'rows':rows}
    h.write_json(output/f'{name}.json',result)
    print(f"[eval] {name} {summary}",flush=True)
    return result


def fit(processor,model,device,root,pool,compact,steps,lr,output,priority=()):
    import torch
    rng=random.Random(2026); groups={tp:[q for q in pool if q['type']==tp] for tp in ('loc','count','exist')}
    order={tp:[] for tp in groups}; optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr)
    logs=[]; types=['loc','loc','loc','count','exist']
    for step in range(steps):
        if step<len(priority): q=priority[step]
        else:
            tp=types[step%len(types)]
            if not order[tp]: order[tp]=list(groups[tp]); rng.shuffle(order[tp])
            q=order[tp].pop()
        target=q['verified_answer'] if 'verified_answer' in q else training_answer(q,compact)
        image=exp.load_image(root,q); question=task_prompt(q,compact)
        tool_conditioned='verified_answer' not in q and step%5==0
        if tool_conditioned: question += ' Visual tool observations: '+observations(image,q)
        batch=h.encode_answer(processor,image,question,target)
        batch={k:v.to(device) for k,v in batch.items()}; model.train(); loss=model(**batch).loss
        loss.backward()
        if step==0:
            gradients={}
            for group in ('vision','connector','language'):
                params=[p for n,p in model.named_parameters() if p.requires_grad and
                        (('vision' in n) if group=='vision' else ('connector' in n) if group=='connector'
                         else ('vision' not in n and 'connector' not in n))]
                if params:
                    magnitude=sum(float(p.grad.detach().abs().sum().cpu()) for p in params if p.grad is not None)
                    assert magnitude>0, f'Inactive adapter gradients: {group}'
                    gradients[group]=magnitude
            output.mkdir(parents=True,exist_ok=True); h.write_json(output/'gradient_check.json',gradients)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        logs.append({'step':step+1,'id':q['id'],'type':q['type'],'tool_conditioned':tool_conditioned,'loss':float(loss.detach().cpu())})
        if (step+1)%50==0: print(f'[train] {output.name} {step+1}/{steps} mean50={sum(x["loss"] for x in logs[-50:])/50:.4f}',flush=True)
    output.mkdir(parents=True,exist_ok=True); model.save_pretrained(output/'adapter'); h.write_json(output/'training.json',logs)


def legacy_questions(root,split):
    return [{**q,'image':h.image_path(root,q).relative_to(root).as_posix(),
             'cell':list(h.parse_cell(q['gold'])) if q['type']=='loc' else None} for q in h.load_questions(root)[split]]


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True); parser.add_argument('--adapter',required=True)
    parser.add_argument('--root',type=Path,default=Path('.')); parser.add_argument('--output',default='improvement/final_v1')
    parser.add_argument('--joint-steps',type=int,default=400); parser.add_argument('--rsi-steps',type=int,default=80)
    parser.add_argument('--development-only',action='store_true',help='Run all development stages without final-test inference')
    parser.add_argument('--loc-cn',action='store_true'); parser.add_argument('--device',default='cuda')
    parser.add_argument('--offline',action='store_true'); parser.add_argument('--no-quant',action='store_true')
    args=parser.parse_args(); root=args.root; output=root/args.output; compact=not args.loc_cn
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    data=make_joint_data(root); pool=list(data['train']); memory=[]; rounds=[]; validation={}
    h.write_json(output/'protocol.json',{'initial_adapter':args.adapter,'joint_steps':args.joint_steps,'rsi_steps':args.rsi_steps,
                 'seed':2026,'compact':compact,'final_selection':'max development localization, then overall accuracy',
                 'tool_training':'Every fifth scheduled non-bootstrap step includes neutral pixel-derived bbox observations; actual flags are logged; no row/column answer string',
                 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/'source_at_run.py').write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(2026)
    processor,model,device,metadata=h.load_model(args,adapter=args.adapter,train=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    h.write_json(output/'model_metadata.json',metadata)
    fit(processor,model,device,root,pool,compact,args.joint_steps,5e-5,output/'sft')
    validation['sft']=evaluate(processor,model,device,root,data['validation'],compact,'sft_validation',output)['summary']
    evaluate(processor,model,device,root,legacy_questions(root,'eval'),compact,'sft_legacy',output)
    evaluate(processor,model,device,root,legacy_questions(root,'eval'),compact,'sft_weak_tool_legacy',output,True,False)
    evaluate(processor,model,device,root,legacy_questions(root,'eval'),compact,'sft_tool_legacy',output,True)
    for number in (1,2):
        feedback=evaluate(processor,model,device,root,legacy_questions(root,'rsi'),compact,f'rsi{number}_feedback',output,True)
        additions=[]
        for q,row in zip(legacy_questions(root,'rsi'),feedback['rows']):
            if row['final_verified'] and not row['verification']['passed'] and not any(p['image']==q['image'] and p['type']==q['type'] and p['target']==q['target'] for p in pool):
                # Gold is unused by fit for these rows: the verified model answer is supervised.
                additions.append({**q,'verified_answer':row['final'],'source':f'rsi{number}'})
            memory.append({'round':number,**row})
        pool+=additions; h.write_json(output/f'pool_round{number}.json',pool)
        fit(processor,model,device,root,pool,compact,args.rsi_steps,1e-5,output/f'rsi{number}',additions)
        validation[f'rsi{number}']=evaluate(processor,model,device,root,data['validation'],compact,f'rsi{number}_validation',output)['summary']
        legacy=evaluate(processor,model,device,root,legacy_questions(root,'eval'),compact,f'rsi{number}_legacy',output)
        rounds.append({'round':number,'pool_size':len(pool),'added':len(additions),
                       'direct_correct':sum(r['raw_correct'] for r in feedback['rows']),
                       'self_attempts':sum(r['self_check'] is not None for r in feedback['rows']),
                       'tool_attempts':sum(r['tool_check'] is not None for r in feedback['rows']),
                       'corrected_correct':sum(r['correct'] for r in feedback['rows']),
                       'legacy':legacy['summary']})
        h.write_json(output/'rsi_memory.json',{'records':memory,'rounds':rounds})
    def key(name):
        s=validation[name]; return (s['loc']['correct']/s['loc']['n'],s['total']['correct']/s['total']['n'])
    selected=max(validation,key=key)
    h.write_json(output/'selection.json',{'selected':selected,'validation':validation,
                 'final_test_evaluated_at_selection':False})
    if args.development_only:
        h.write_json(output/'summary.json',{'selected':selected,'validation':validation,'rounds':rounds,
                     'final_test_evaluated':False})
        print('[complete] development-only multi-task and RSI comparison; test remains sealed',flush=True)
        return
    del model,processor; gc.collect(); torch.cuda.empty_cache()
    processor,model,device,_=h.load_model(args,adapter=output/selected/'adapter')
    final=evaluate(processor,model,device,root,data['final_test'],compact,'selected_final_test',output)
    evaluate(processor,model,device,root,data['final_test'],compact,'selected_weak_tool_final_test',output,True,False)
    evaluate(processor,model,device,root,data['final_test'],compact,'selected_rich_tool_final_test',output,True,True)
    probes=data['final_test'][:128:8]
    exp.evaluate(processor,model,device,root,probes,compact,'selected_blank16',output,True)
    h.write_json(output/'summary.json',{'selected':selected,'validation':validation,'final_test':final['summary'],'rounds':rounds})
    print('[complete] optimized SFT, two RSI rounds and sealed final test',flush=True)


if __name__=='__main__': main()
