"""Bounded continuation study: diverse joint SFT, conservative RSI, fresh test.

Old final-test answers are never used for training or model selection. A new
test is generated only after the development-selected adapter is frozen.
The required default experiment and all v1 evidence remain unchanged.
"""
import argparse
import gc
import hashlib
import math
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image

import hw6_rsi_localize as h
import improve_localization as exp
import finish_grounding as f

PLAN = {
    'seed': 20261030, 'train_image_seed': 20261031,
    'feedback_image_seeds': [20261032, 20261033], 'test_image_seed': 20261036,
    'initial_adapter': 'improvement/final_v1/sft/adapter',
    'sft_checkpoints_optimizer_updates': [400, 800, 1600],
    'gradient_accumulation': 2, 'sft_lr': 3e-5, 'sft_min_lr': 3e-6,
    'warmup_updates': 80, 'max_grad_norm': 1.0,
    'task_cycle': ['loc'] * 6 + ['count'] * 3 + ['exist'],
    'train_questions': {'loc': 2048, 'count': 1280, 'exist': 512},
    'rsi_rounds': 2, 'rsi_optimizer_updates_per_round': 200, 'rsi_lr': 3e-6,
    'bootstrap_replay_microstep_period': 8,
    'feedback_questions_per_round': {'loc': 128, 'count': 40, 'exist': 16},
    'test_questions': {'loc': 256, 'count': 80, 'exist': 32},
    'selection': 'maximum development localization accuracy, then overall accuracy; earlier stage wins ties',
    'test_policy': 'generate fresh test only after stage selection; compare fixed v1 SFT and selected v2 once',
    'training_tools': 'No observations in regular SFT prompts. Only verified model answers supervise RSI replay.',
    'feedback_tools': 'purple overlay self-check, then neutral pixel bounding boxes; no exhaustive candidate fallback',
    'comparison': 'Cumulative improvement, not a matched-compute ablation; identical native decoding and NF4 base.',
}


def digest_image(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def history_hashes(root):
    """Images are inspected only for RGB duplicate exclusion, not labels/errors."""
    folders = [root/'hw6_imgs', root/'improvement/data',
               root/'improvement/augmented_data', root/'improvement/joint_data']
    return {digest_image(p) for folder in folders for p in folder.rglob('*.png')}


def objects_for(rng, tp, target, cell, number, hard):
    if tp == 'loc':
        objects = exp.draw_target_scene(rng, cell, target)
        if hard and len(objects) > 1:
            # Target stays unique. Deliberately distinguish shape from color.
            objects[1]['class'] = target[0] + ('方' if target[1] == '圆' else '圆')
            if len(objects) > 2:
                objects[2]['class'] = rng.choice([c for c in h.COLORS if c != target[0]]) + target[1]
        return objects
    n = number if tp == 'count' else (0 if number == 0 else rng.randint(1, 4))
    extra = rng.randint(max(0, 2-n), 4-n)
    cells = rng.sample(range(16), n+extra)
    objects = []
    for index, position in enumerate(cells):
        cls = target if index < n else rng.choice([c for c in exp.CLASSES if c != target])
        if hard and index == n:
            cls = target[0] + ('方' if target[1] == '圆' else '圆')
        row, col = divmod(position, 4)
        side = rng.randint(30, 48)
        x = col*64 + rng.randint(6, 58-side)
        y = row*64 + rng.randint(6, 58-side)
        objects.append({'class': cls, 'box': [x, y, x+side, y+side]})
    return objects


def make_set(root, folder, counts, seed, excluded, hard=False):
    """Fully crossed labels and fresh pixels, independently checked on creation."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for tp, amount in counts.items():
        labels = ([(cell, target, None) for cell in range(16) for target in exp.CLASSES]
                  if tp == 'loc' else
                  [(None, target, n) for target in exp.CLASSES for n in range(5 if tp == 'count' else 2)])
        assert amount % len(labels) == 0
        schedule = labels * (amount // len(labels))
        rng.shuffle(schedule)
        for index, (cell, target, number) in enumerate(schedule):
            path = folder/f'{tp}_{index:05d}.png'
            for attempt in range(100):
                objects = objects_for(rng, tp, target, cell, number, hard and index % 2 == 0)
                h.render_scene(objects, path)
                digest = digest_image(path)
                if digest not in excluded:
                    break
            else:
                raise ValueError('Unable to generate distinct image')
            excluded.add(digest)
            gold = h.cell_text(divmod(cell, 4)) if tp == 'loc' else (
                f'{number}个' if tp == 'count' else '有' if number else '没有')
            q = {'id': f'{folder.name}_{tp}_{index:05d}', 'type': tp, 'target': target,
                 'image': path.relative_to(root).as_posix(), 'gold': gold,
                 'box': objects[0]['box'] if tp == 'loc' else None,
                 'objects': objects, 'pixel_sha256': digest}
            if tp == 'loc': q['cell'] = list(divmod(cell, 4))
            assert h.pixel_answer(exp.load_image(root, q), q)[0] == gold, q['id']
            rows.append(q)
            if len(rows) % 512 == 0:
                print(f'[data-v2] {folder.name} generated and verified {len(rows)} images', flush=True)
    h.write_json(folder/'questions.json', rows)
    return rows


def key(metrics):
    return (metrics['loc']['correct']/metrics['loc']['n'],
            metrics['total']['correct']/metrics['total']['n'])


def release(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load(args, adapter, train):
    processor, model, device, metadata = h.load_model(args, adapter=adapter, train=train)
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return processor, model, device, metadata


def fit(args, processor, model, device, train, validation, output, checkpoints, lr, seed, replay=(), rsi=False):
    """Two distinct examples per optimizer update; bounded bootstrap replay."""
    import torch
    rng = random.Random(seed)
    groups = {tp: [q for q in train if q['type'] == tp] for tp in ('loc', 'count', 'exist')}
    order = {tp: [] for tp in groups}
    replay_order = []
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    accumulation = PLAN['gradient_accumulation']
    logs = []
    scores = {}
    output.mkdir(parents=True, exist_ok=True)
    for update in range(1, max(checkpoints)+1):
        if rsi:
            rate = lr
        else:
            warm = PLAN['warmup_updates']
            progress = max(0, update-warm)/max(1, max(checkpoints)-warm)
            rate = lr*update/warm if update <= warm else PLAN['sft_min_lr'] + .5*(lr-PLAN['sft_min_lr'])*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group['lr'] = rate
        for micro in range(accumulation):
            index = (update-1)*accumulation + micro
            is_replay = bool(replay) and index % PLAN['bootstrap_replay_microstep_period'] == 0
            if is_replay:
                if not replay_order:
                    replay_order = list(replay); rng.shuffle(replay_order)
                q = replay_order.pop()
                assert 'verified_answer' in q
                target = q['verified_answer']
            else:
                tp = PLAN['task_cycle'][index % len(PLAN['task_cycle'])]
                if not order[tp]:
                    order[tp] = list(groups[tp]); rng.shuffle(order[tp])
                q = order[tp].pop()
                target = f.training_answer(q, True)
            batch = h.encode_answer(processor, exp.load_image(args.root, q), f.task_prompt(q, True), target)
            batch = {k: v.to(device) for k, v in batch.items()}
            model.train()
            loss = model(**batch).loss
            assert torch.isfinite(loss), 'Nonfinite loss'
            (loss/accumulation).backward()
            logs.append({'update': update, 'microstep': index+1, 'id': q['id'], 'type': q['type'],
                         'replay': is_replay, 'answer': target, 'lr': rate, 'loss': float(loss.detach().cpu())})
        if update == 1:
            gradients = {g: sum(float(p.grad.detach().abs().sum().cpu()) for n, p in model.named_parameters()
                               if p.requires_grad and p.grad is not None and
                               (('vision' in n) if g == 'vision' else ('connector' in n) if g == 'connector'
                                else ('vision' not in n and 'connector' not in n)))
                         for g in ('vision', 'connector', 'language')}
            assert min(gradients.values()) > 0
            h.write_json(output/'gradient_check.json', gradients)
        norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), PLAN['max_grad_norm'])
        assert torch.isfinite(norm), 'Nonfinite gradient norm'
        logs[-1]['gradient_norm_before_clip'] = float(norm.cpu())
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if update % 100 == 0:
            print(f'[train-v2] {output.name} update={update}/{max(checkpoints)} mean200={sum(r["loss"] for r in logs[-200:])/len(logs[-200:]):.4f}', flush=True)
        if update in checkpoints:
            folder = output/f'update{update}'
            model.save_pretrained(folder/'adapter')
            score = f.evaluate(processor, model, device, args.root, validation, True, 'validation', folder)['summary']
            scores[str(update)] = score
            h.write_json(output/'training.json', logs)
            h.write_json(output/'validation_checkpoints.json', scores)
    return scores


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-quant', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(); args.root = root
    out = root/'improvement/final_v2'
    if out.exists(): raise FileExistsError('Preserve previous attempts: use a new protocol version')
    out.mkdir(parents=True)
    started = time.perf_counter()
    source_hashes = {}
    for source in ('hw6_rsi_localize.py', 'improve_localization.py', 'finish_grounding.py', Path(__file__).name):
        data = (root/source).read_bytes()
        (out/'sources').mkdir(exist_ok=True)
        (out/'sources'/source).write_bytes(data)
        source_hashes[source] = hashlib.sha256(data).hexdigest()
    h.write_json(out/'protocol.json', {**PLAN, 'source_sha256': source_hashes,
                                      'final_test_created': False})
    excluded = history_hashes(root)
    historical_count = len(excluded)
    train = make_set(root, out/'data/train', PLAN['train_questions'], PLAN['train_image_seed'], excluded, True)
    feedback_sets = [make_set(root, out/f'data/feedback{r}', PLAN['feedback_questions_per_round'], seed, excluded, True)
                     for r, seed in enumerate(PLAN['feedback_image_seeds'], 1)]
    validation = h.read_json(root/'improvement/final_v1/joint_questions_at_run.json')['validation']
    h.write_json(out/'development_questions.json', validation)
    h.write_json(out/'history_exclusion.json', {'unique_historical_images': historical_count,
                                              'hashes_before_fresh_test': sorted(excluded)})
    torch.manual_seed(PLAN['seed'])
    initial = root/PLAN['initial_adapter']
    processor, model, device, metadata = load(args, initial, True)
    h.write_json(out/'model_metadata.json', metadata)
    stages = {'v1_sft': {'adapter': PLAN['initial_adapter'],
                         'summary': f.evaluate(processor, model, device, root, validation, True, 'v1_validation', out)['summary']}}
    scores = fit(args, processor, model, device, train, validation, out/'sft',
                 PLAN['sft_checkpoints_optimizer_updates'], PLAN['sft_lr'], PLAN['seed'])
    for update, metrics in scores.items():
        stages[f'sft{update}'] = {'adapter': f'improvement/final_v2/sft/update{update}/adapter', 'summary': metrics}
    chosen_sft = max(stages, key=lambda name: key(stages[name]['summary']))
    h.write_json(out/'sft_selection.json', {'stages': stages, 'selected': chosen_sft, 'fresh_test_created': False})
    del model; gc.collect(); torch.cuda.empty_cache()
    processor, model, device, _ = load(args, root/stages[chosen_sft]['adapter'], True)
    replay = []
    rounds = []
    original_feedback = f.legacy_questions(root, 'rsi')
    for number, questions in enumerate(feedback_sets, 1):
        questions = original_feedback + questions
        feedback = f.evaluate(processor, model, device, root, questions, True, f'rsi{number}_feedback', out, True, True, False)
        existing = {(q['image'], q['type'], q['target']) for q in replay}
        additions = []
        for q, row in zip(questions, feedback['rows']):
            identity = (q['image'], q['type'], q['target'])
            if row['final_verified'] and not row['verification']['passed'] and identity not in existing:
                additions.append({**q, 'verified_answer': row['final'], 'source': f'v2_rsi{number}',
                                  'answer_origin': f.answer_origin(row)})
                existing.add(identity)
        replay += additions
        h.write_json(out/f'pool_round{number}.json', replay)
        scores = fit(args, processor, model, device, train, validation, out/f'rsi{number}',
                     [PLAN['rsi_optimizer_updates_per_round']], PLAN['rsi_lr'], PLAN['seed']+number, replay, True)
        stage = f'rsi{number}'
        stages[stage] = {'adapter': f'improvement/final_v2/{stage}/update{PLAN["rsi_optimizer_updates_per_round"]}/adapter',
                         'summary': scores[str(PLAN['rsi_optimizer_updates_per_round'])]}
        post = f.evaluate(processor, model, device, root, replay, True, f'{stage}_bootstrap_posttrain', out) if replay else None
        legacy = f.evaluate(processor, model, device, root, f.legacy_questions(root, 'eval'), True, f'{stage}_legacy', out)
        rounds.append({'round': number, 'regular_pool_size': len(train), 'verified_pool_size': len(replay),
                       'added': len(additions),
                       'raw_correct_count': sum(r['raw_correct'] for r in feedback['rows']),
                       'feedback_n': len(questions), 'corrected_correct': sum(r['correct'] for r in feedback['rows']),
                       'self_attempts': sum(r['self_check'] is not None for r in feedback['rows']),
                       'tool_attempts': sum(r['tool_check'] is not None for r in feedback['rows']),
                       'bootstrap_posttrain': post['summary'] if post else None, 'legacy': legacy['summary']})
        h.write_json(out/'rounds.json', rounds)
    selected = max(stages, key=lambda name: key(stages[name]['summary']))
    selection = {'rule': PLAN['selection'], 'stages': stages, 'selected': selected,
                 'sft_selected_before_rsi': chosen_sft, 'fresh_test_created_at_selection': False}
    h.write_json(out/'selection.json', selection)
    del model; gc.collect(); torch.cuda.empty_cache()
    # No training/selection occurs below this boundary. Identical new images for both models.
    test = make_set(root, out/'data/fresh_test', PLAN['test_questions'], PLAN['test_image_seed'], excluded, False)
    finals = {}
    for name, adapter in [('v1', initial), ('v2', root/stages[selected]['adapter'])]:
        processor, model, device, _ = load(args, adapter, False)
        finals[name] = f.evaluate(processor, model, device, root, test, True, f'{name}_fresh_test', out)['summary']
        if name == 'v2':
            f.evaluate(processor, model, device, root, f.legacy_questions(root, 'eval'), True, 'selected_legacy', out)
            f.evaluate(processor, model, device, root, f.legacy_questions(root, 'eval'), True, 'selected_weak_tool_legacy', out, True, False)
            f.evaluate(processor, model, device, root, f.legacy_questions(root, 'eval'), True, 'selected_rich_tool_legacy', out, True, True)
            f.evaluate(processor, model, device, root, test, True, 'v2_rich_tool_fresh_test', out, True, True)
            probes = [next(q for q in test if q['type'] == 'loc' and q['target'] == exp.CLASSES[0]
                           and q['cell'] == list(divmod(cell, 4))) for cell in range(16)]
            exp.evaluate(processor, model, device, root, probes, True, 'v2_blank16', out, blank=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    h.write_json(out/'summary.json', {'selected': selected, 'stages': stages, 'rounds': rounds,
                                     'fresh_test': finals, 'seconds': time.perf_counter()-started})
    print(f'[complete-v2] selected={selected} fresh_test={finals}', flush=True)


if __name__ == '__main__': main()
