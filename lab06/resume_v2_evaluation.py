"""Resume only sealed evaluation after job 15408 reached its wall-time limit.

This program has no training call and cannot change the recorded selection.
Completed evaluation files are checked and reused, never selected by score.
"""
import argparse
import gc
import hashlib
import time
from pathlib import Path

import hw6_rsi_localize as h
import finish_grounding as f
import optimize_grounding_v2 as opt


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-quant', action='store_true')
    args = parser.parse_args(); args.root = args.root.resolve()
    root = args.root; out = root/'improvement/final_v2'
    selection_path = out/'selection.json'
    selection_bytes = selection_path.read_bytes()
    selected = h.read_json(selection_path)
    assert not selected['fresh_test_created_at_selection']
    assert selected['selected'] == max(selected['stages'], key=lambda n: opt.key(selected['stages'][n]['summary']))
    for stage, updates in [('sft', 1600), ('rsi1', 200), ('rsi2', 200)]:
        logs = h.read_json(out/stage/'training.json')
        assert len(logs) == updates*2 and logs[-1]['update'] == updates
    test = h.read_json(out/'data/fresh_test/questions.json')
    assert len(test) == 368
    expected_ids = [q['id'] for q in test]
    protocol = h.read_json(out/'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    (out/'sources'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    h.write_json(out/'evaluation_resume.json', {'reason': 'Job 15408 time limit after all training, selection and fresh-test creation',
                 'training_restarted': False, 'selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
                 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'seed': opt.PLAN['seed']})
    torch.manual_seed(opt.PLAN['seed'])
    finals = {}
    for name, adapter in [('v1', root/opt.PLAN['initial_adapter']),
                          ('v2', root/selected['stages'][selected['selected']]['adapter'])]:
        processor, model, device, _ = opt.load(args, adapter, False)

        def evaluate(questions, title, correction=False, detail=True):
            path = out/f'{title}.json'
            if path.exists():
                result = h.read_json(path)
                assert [r['id'] for r in result['rows']] == [q['id'] for q in questions]
                print(f'[resume-reuse] {title}', flush=True)
                return result
            return f.evaluate(processor, model, device, root, questions, True, title, out, correction, detail)

        result = evaluate(test, f'{name}_fresh_test')
        assert [r['id'] for r in result['rows']] == expected_ids
        finals[name] = result['summary']
        if name == 'v2':
            legacy = f.legacy_questions(root, 'eval')
            evaluate(legacy, 'selected_legacy')
            evaluate(legacy, 'selected_weak_tool_legacy', True, False)
            evaluate(legacy, 'selected_rich_tool_legacy', True, True)
            evaluate(test, 'v2_rich_tool_fresh_test', True, True)
            probes = [next(q for q in test if q['type'] == 'loc' and q['target'] == opt.exp.CLASSES[0]
                           and q['cell'] == list(divmod(cell, 4))) for cell in range(16)]
            if not (out/'v2_blank16.json').exists():
                opt.exp.evaluate(processor, model, device, root, probes, True, 'v2_blank16', out, blank=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    assert selection_path.read_bytes() == selection_bytes
    h.write_json(out/'summary.json', {'selected': selected['selected'], 'stages': selected['stages'],
                                    'rounds': h.read_json(out/'rounds.json'), 'fresh_test': finals,
                                    'evaluation_resume_seconds': time.perf_counter()-started,
                                    'training_job': 15408, 'evaluation_only_resume': True})
    print(f'[complete-v2-resume] selected={selected["selected"]} fresh_test={finals}', flush=True)


if __name__ == '__main__': main()
