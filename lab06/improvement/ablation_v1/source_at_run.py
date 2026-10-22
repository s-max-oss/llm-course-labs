"""Controlled localization diagnostics; never overwrites the required lab run.

Development validation is used for comparing changes. The separately generated
final-test split must not be used until a training recipe has been selected.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image
import hw6_rsi_localize as h

CLASSES = [c+s for c in h.COLORS for s in h.SHAPES]
EN_COLORS = dict(zip(h.COLORS, ('red', 'blue', 'green', 'yellow')))


def prompt(target, compact=False):
    if compact:
        return (f"Locate the {EN_COLORS[target[0]]} {'circle' if target[1]=='圆' else 'square'}. "
                "Divide the image into four rows and four columns. Rows run from top to bottom; "
                "columns run from left to right. Answer only rXcY, where X and Y are numbers from 1 to 4.")
    name = target[0]+'色'+('圆形' if target[1]=='圆' else '方形')
    return f"图中的{name}位于第几行第几列？请用“第X行第Y列”回答，网格为4×4。"


def draw_target_scene(rng, cell, target):
    cells = [cell] + rng.sample([i for i in range(16) if i != cell], rng.randint(1, 3))
    objects = []
    for i, position in enumerate(cells):
        cls = target if i == 0 else rng.choice([c for c in CLASSES if c != target])
        row, col = divmod(position, 4)
        side = rng.randint(30, 48)
        x = col*64 + rng.randint(6, 58-side)
        y = row*64 + rng.randint(6, 58-side)
        objects.append({'class': cls, 'box': [x, y, x+side, y+side]})
    return objects


def make_balanced_data(root):
    folder = root/'improvement'/'data'
    folder.mkdir(parents=True, exist_ok=True)
    splits = {}
    hashes = {}
    for split, repetitions, seed in [('train', 8, 20261022), ('validation', 8, 20261023), ('final_test', 8, 20261024)]:
        rng = random.Random(seed)
        images = folder/split
        images.mkdir(exist_ok=True)
        rows = []
        for cell in range(16):
            for rep in range(repetitions):
                target = CLASSES[rep]
                objects = draw_target_scene(rng, cell, target)
                path = images/f'{cell:02d}_{rep:02d}.png'
                h.render_scene(objects, path)
                digest = hashlib.sha256(Image.open(path).convert('RGB').tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError('Duplicate image across splits')
                hashes[digest] = str(path.relative_to(root))
                row, col = divmod(cell, 4)
                rows.append({'id': f'{split}_{cell:02d}_{rep:02d}', 'type': 'loc', 'target': target,
                             'image': path.relative_to(root).as_posix(), 'cell': [row, col],
                             'gold': h.cell_text((row, col)), 'box': objects[0]['box'], 'objects': objects})
        splits[split] = rows
    h.write_json(folder/'splits.json', splits)
    h.write_json(folder/'pixel_hashes.json', hashes)
    print('[data] fully crossed train=128, development validation=128, sealed final_test=128', flush=True)
    return splits


def load_image(root, q, blank=False):
    return Image.new('RGB', (256, 256), 'white') if blank else Image.open(root/q['image']).convert('RGB')


def original_rows(root, split):
    rows = []
    for q in h.load_questions(root)[split]:
        if q['type'] == 'loc':
            rows.append({**q, 'image': h.image_path(root, q).relative_to(root).as_posix(),
                         'cell': list(h.parse_cell(q['gold']))})
    return rows


def response(q, compact):
    row, col = q['cell']
    return f'r{row+1}c{col+1}' if compact else h.cell_text((row, col))


def evaluate(processor, model, device, root, questions, compact, name, output, blank=False):
    records = []
    tick = time.perf_counter()
    for q in questions:
        text = h.answer(processor, model, device, load_image(root, q, blank), prompt(q['target'], compact), 24)
        canonical = h.parsed_answer(q, text)
        valid = canonical is not None
        correct = canonical == q['gold']
        records.append({'id': q['id'], 'gold': q['gold'], 'prediction': text,
                        'format_valid': valid, 'correct': correct, 'iou': h.loc_iou(q, text)})
    result = {'name': name, 'n': len(records), 'correct': sum(q['correct'] for q in records),
              'valid': sum(q['format_valid'] for q in records),
              'mean_iou': sum(q['iou'] for q in records)/len(records),
              'blank_image': blank, 'rows': records, 'seconds': time.perf_counter()-tick}
    h.write_json(output/f'{name}.json', result)
    print(f"[eval] {name} correct={result['correct']}/{result['n']} format={result['valid']}/{result['n']}", flush=True)
    return result


def rank_candidates(processor, model, device, root, questions, compact, name, output):
    import torch
    model.eval()
    records = []
    with torch.inference_mode():
        for q in questions:
            image = load_image(root, q)
            scores = []
            for row in range(4):
                for col in range(4):
                    answer = f'r{row+1}c{col+1}' if compact else h.cell_text((row, col))
                    batch = h.encode_answer(processor, image, prompt(q['target'], compact), answer)
                    batch = {k:v.to(device) for k,v in batch.items()}
                    labels = batch.pop('labels')[:, 1:]
                    logits = model(**batch).logits[:, :-1].float().log_softmax(-1)
                    chosen = logits.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                    scores.append(float((chosen*labels.ne(-100)).sum().cpu()))
            index = max(range(16), key=lambda i:scores[i])
            selected = h.cell_text(divmod(index, 4))
            records.append({'id':q['id'], 'gold':q['gold'], 'prediction':selected,
                            'correct':selected==q['gold'], 'candidate_logp':scores})
    result = {'name':name, 'n':len(records), 'correct':sum(q['correct'] for q in records),
              'decoding':'select maximum sequence log probability among 16 legal answers', 'rows':records}
    h.write_json(output/f'{name}.json', result)
    print(f"[rank] {name} correct={result['correct']}/{result['n']}", flush=True)
    return result


def fit(processor, model, device, root, questions, compact, steps, seed, output):
    import torch
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    rng = random.Random(seed)
    order = []
    logs = []
    started = time.perf_counter()
    for step in range(steps):
        if not order:
            order = list(range(len(questions)))
            rng.shuffle(order)
        q = questions[order.pop()]
        batch = h.encode_answer(processor, load_image(root, q), prompt(q['target'], compact), response(q, compact))
        batch = {k:v.to(device) for k,v in batch.items()}
        model.train()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logs.append({'step':step+1, 'id':q['id'], 'loss':float(loss.detach().cpu())})
        if (step+1)%50 == 0:
            print(f"[train] {output.name} step={step+1}/{steps} mean50={sum(x['loss'] for x in logs[-50:])/50:.4f}", flush=True)
    model.save_pretrained(output/'adapter')
    h.write_json(output/'training.json', {'steps':steps, 'lr':1e-4, 'batch_size':1,
                 'sampling':'shuffle each epoch, all rows eligible', 'seconds':time.perf_counter()-started, 'records':logs})


def release(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', default='improvement/ablation_v1')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-quant', action='store_true')
    args = parser.parse_args()
    root = args.root
    output = root/args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory to preserve completed experiments')
    output.mkdir(parents=True, exist_ok=True)
    h.write_json(output/'provenance.json', {'seed':2026, 'steps_per_arm':args.steps,
                 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'model_helper_sha256':hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()})
    (output/'source_at_run.py').write_bytes(Path(__file__).read_bytes())
    data_file = root/'improvement/data/splits.json'
    splits = h.read_json(data_file) if data_file.exists() else make_balanced_data(root)
    legacy_eval = original_rows(root, 'eval')
    seed = original_rows(root, 'seed')
    # Same target class in all 16 cells: the prompt cannot predict the cell.
    # These are spatial diagnostics, not a replacement for all-class validation.
    probes = [splits['validation'][i*8] for i in range(16)]
    summaries = {}
    torch.manual_seed(2026)
    processor, model, device, metadata = h.load_model(args)
    for compact in (False, True):
        name = 'base_compact' if compact else 'base_cn'
        folder = output/name
        folder.mkdir(exist_ok=True)
        summaries[name] = {'legacy':evaluate(processor,model,device,root,legacy_eval,compact,'legacy',folder),
                           'validation':evaluate(processor,model,device,root,splits['validation'],compact,'validation',folder)}
        summaries[name]['constrained'] = rank_candidates(processor,model,device,root,probes,compact,'constrained16',folder)
    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    for name, train_set, compact in [('seed_extended_cn', seed, False),
                                     ('balanced_cn', splits['train'], False),
                                     ('balanced_compact', splits['train'], True)]:
        torch.manual_seed(2026)
        folder = output/name
        folder.mkdir(exist_ok=True)
        processor, model, device, metadata = h.load_model(args, train=True)
        h.write_json(folder/'metadata.json', {'model':metadata, 'data_n':len(train_set),
                     'coverage':dict(Counter(q['gold'] for q in train_set)),
                     'compact':compact, 'steps':args.steps, 'seed':2026})
        fit(processor,model,device,root,train_set,compact,args.steps,2026,folder)
        train_probes = train_set if len(train_set) <= 16 else train_set[::8]
        summaries[name] = {'train':evaluate(processor,model,device,root,train_probes,compact,'training_probe',folder),
                           'legacy':evaluate(processor,model,device,root,legacy_eval,compact,'legacy',folder),
                           'validation':evaluate(processor,model,device,root,splits['validation'],compact,'validation',folder),
                           'blank':evaluate(processor,model,device,root,probes,compact,'blank16',folder,blank=True),
                           'constrained':rank_candidates(processor,model,device,root,probes,compact,'constrained16',folder)}
        h.write_json(output/'partial_summary.json', summaries)
        del model, processor
        gc.collect(); torch.cuda.empty_cache()
    h.write_json(output/'summary.json', {'steps_per_arm':args.steps, 'final_test_evaluated':False,
                                       'arms':summaries})
    print('[complete] localization ablation; final test remains sealed', flush=True)


if __name__ == '__main__':
    main()
