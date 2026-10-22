"""Online diverse localization curriculum; fixed development set, sealed test."""
import argparse
import hashlib
import random
import time
from pathlib import Path

import hw6_rsi_localize as h
import improve_localization as exp


def make_examples(root, steps):
    folder=root/'improvement/augmented_data'; folder.mkdir(parents=True,exist_ok=True)
    rng=random.Random(20261028); records=[]; order=[]; hashes=set()
    for step in range(steps):
        if not order:
            order=list(range(128)); rng.shuffle(order)
        identity=order.pop(); cell,cls=divmod(identity,8); target=exp.CLASSES[cls]
        objects=exp.draw_target_scene(rng,cell,target)
        objects=objects[:1] if step<256 else objects[:2] if step<768 else objects
        image=folder/f'{step:05d}.png'; h.render_scene(objects,image)
        digest=hashlib.sha256(exp.Image.open(image).convert('RGB').tobytes()).hexdigest()
        assert digest not in hashes, 'Repeated training image'
        hashes.add(digest); row,col=divmod(cell,4)
        records.append({'id':f'aug_{step:05d}','type':'loc','target':target,
                        'image':image.relative_to(root).as_posix(),'cell':[row,col],
                        'gold':h.cell_text((row,col)),'box':objects[0]['box'],
                        'objects':objects,'pixel_sha256':digest})
    split=h.read_json(root/'improvement/data/splits.json')
    heldout={hashlib.sha256(exp.load_image(root,q).tobytes()).hexdigest()
             for name in ('validation','final_test') for q in split[name]}
    assert not hashes & heldout
    h.write_json(folder/'questions.json',records)
    return records


def main():
    import torch
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--adapter',required=True)
    p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--output',default='improvement/augmented_v1'); p.add_argument('--device',default='cuda')
    p.add_argument('--offline',action='store_true'); p.add_argument('--no-quant',action='store_true')
    args=p.parse_args(); root=args.root; out=root/args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True); train=make_examples(root,args.steps)
    validation=h.read_json(root/'improvement/data/splits.json')['validation']
    torch.manual_seed(2026)
    processor,model,device,metadata=h.load_model(args,adapter=args.adapter,train=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    h.write_json(out/'protocol.json',{'initial_adapter':args.adapter,'model':metadata,'steps':args.steps,
                 'lr':1e-4,'seed':2026,'image_seed':20261028,'new_image_every_step':True,
                 'curriculum':'256 single-object; 512 two-object; remaining 2-4 objects',
                 'final_test_evaluated':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'source_at_run.py').write_bytes(Path(__file__).read_bytes())
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    logs=[]; summaries={}; started=time.perf_counter()
    for step,q in enumerate(train,1):
        batch=h.encode_answer(processor,exp.load_image(root,q),exp.prompt(q['target'],True),exp.response(q,True))
        batch={k:v.to(device) for k,v in batch.items()}; model.train(); loss=model(**batch).loss; loss.backward()
        if step==1:
            gradients={g:sum(float(p.grad.detach().abs().sum().cpu()) for n,p in model.named_parameters()
                       if p.requires_grad and p.grad is not None and
                       (('vision' in n) if g=='vision' else ('connector' in n) if g=='connector'
                        else ('vision' not in n and 'connector' not in n))) for g in ('vision','connector','language')}
            assert min(gradients.values())>0
            h.write_json(out/'gradient_check.json',gradients)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        logs.append({'step':step,'id':q['id'],'loss':float(loss.detach().cpu())})
        if step%100==0: print(f'[train] {step}/{args.steps} mean100={sum(r["loss"] for r in logs[-100:])/100:.4f}',flush=True)
        if step in {1000,2500,args.steps}:
            folder=out/f'step{step}'; folder.mkdir(exist_ok=True); model.save_pretrained(folder/'adapter')
            result=exp.evaluate(processor,model,device,root,validation,True,'validation',folder)
            summaries[str(step)]={k:v for k,v in result.items() if k!='rows'}
            h.write_json(out/'partial_summary.json',summaries); h.write_json(out/'training.json',logs)
    exp.evaluate(processor,model,device,root,exp.original_rows(root,'eval'),True,'legacy',out)
    exp.evaluate(processor,model,device,root,validation[::8],True,'blank16',out,blank=True)
    h.write_json(out/'summary.json',{'validation_checkpoints':summaries,'seconds':time.perf_counter()-started,'final_test_evaluated':False})
    print('[complete] online augmentation development run',flush=True)


if __name__=='__main__': main()
