"""Test whether adapting visual attention and the connector improves grounding.

Same fully crossed training images and compact prompts as the language-only arm.
No final-test predictions are produced here.
"""
import argparse
import hashlib
import random
import time
from pathlib import Path

import hw6_rsi_localize as h
import improve_localization as exp


def load_visual_lora(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    processor, model, device, metadata = h.load_model(args)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache = False
    targets = [name for name, _ in model.named_modules()
               if name.endswith(('q_proj','k_proj','v_proj','o_proj','out_proj'))
               or name.endswith('connector.modality_projection.proj')]
    assert any('vision' in name for name in targets)
    assert any('connector' in name for name in targets)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=targets, task_type='CAUSAL_LM'))
    numerator, denominator = model.get_nb_trainable_parameters()
    metadata.update({'trainable':numerator, 'total':denominator, 'percent':100*numerator/denominator,
                     'target_modules':targets})
    print(f'[model] visual+connector+language LoRA {numerator}/{denominator}',flush=True)
    return processor, model, device, metadata


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', default='improvement/vision_compact_v1')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-quant', action='store_true')
    args = parser.parse_args()
    root, output = args.root, args.root/args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    data = h.read_json(root/'improvement/data/splits.json')
    for split in ('train','validation','final_test'):
        assert len(data[split])==128
        assert len({(tuple(q['cell']),q['target']) for q in data[split]})==128
    train = data['train']
    torch.manual_seed(2026)
    rng = random.Random(2026)
    processor, model, device, metadata = load_visual_lora(args)
    h.write_json(output/'metadata.json', {'model':metadata,'steps':args.steps,'seed':2026,
                 'lr':1e-4,'batch_size':1,'final_test_evaluated':False,
                 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'helper_sha256':hashlib.sha256(Path(exp.__file__).read_bytes()).hexdigest()})
    (output/'source_at_run.py').write_bytes(Path(__file__).read_bytes())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    order, records, summaries = [], [], {}
    started = time.perf_counter()
    checkpoints = {400,args.steps}
    for step in range(1,args.steps+1):
        if not order:
            order = list(range(len(train))); rng.shuffle(order)
        q = train[order.pop()]
        batch = h.encode_answer(processor,exp.load_image(root,q),exp.prompt(q['target'],True),exp.response(q,True))
        batch = {k:v.to(device) for k,v in batch.items()}
        model.train()
        loss = model(**batch).loss
        loss.backward()
        if step==1:
            gradients={}
            for group in ('vision','connector','language'):
                parameters=[p for name,p in model.named_parameters() if p.requires_grad and
                            (('vision' in name) if group=='vision' else ('connector' in name) if group=='connector'
                             else ('vision' not in name and 'connector' not in name))]
                magnitude=sum(float(p.grad.detach().abs().sum().cpu()) for p in parameters if p.grad is not None)
                assert magnitude>0, f'No nonzero {group} adapter gradient'
                gradients[group]=magnitude
            h.write_json(output/'gradient_check.json',gradients)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        records.append({'step':step,'id':q['id'],'loss':float(loss.detach().cpu())})
        if step%50==0:
            print(f'[train] visual step={step}/{args.steps} mean50={sum(r["loss"] for r in records[-50:])/50:.4f}',flush=True)
        if step in checkpoints:
            folder=output/f'step{step}'; folder.mkdir(exist_ok=True)
            model.save_pretrained(folder/'adapter')
            summaries[step]=exp.evaluate(processor,model,device,root,data['validation'],True,'validation',folder)
            h.write_json(output/'partial_summary.json',summaries)
    probes=data['validation'][::8]
    exp.evaluate(processor,model,device,root,train[::8],True,'training_probe',output)
    exp.evaluate(processor,model,device,root,exp.original_rows(root,'eval'),True,'legacy',output)
    exp.evaluate(processor,model,device,root,probes,True,'blank16',output,blank=True)
    exp.rank_candidates(processor,model,device,root,probes,True,'constrained16',output)
    h.write_json(output/'training.json',{'records':records,'seconds':time.perf_counter()-started})
    h.write_json(output/'summary.json',{'validation_checkpoints':summaries,'final_test_evaluated':False})
    print('[complete] visual grounding diagnostics',flush=True)


if __name__=='__main__':
    main()
