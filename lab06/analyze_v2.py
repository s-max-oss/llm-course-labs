"""Audit v2 data, actual optimization/replay, selection and paired fresh test."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import hw6_rsi_localize as h
import finish_grounding as f
import optimize_grounding_v2 as opt
from analyze_improvement import wilson

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'improvement/final_v2'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def audit_links():
    """Bind all summary/selection statements to actual scored artifacts."""
    choice = read(OUT/'selection.json'); summary = read(OUT/'summary.json')
    mapping = {'v1_sft': OUT/'v1_validation.json'}
    mapping.update({f'sft{n}': OUT/f'sft/update{n}/validation.json' for n in (400,800,1600)})
    mapping.update({f'rsi{n}': OUT/f'rsi{n}/update200/validation.json' for n in (1,2)})
    assert list(choice['stages']) == list(mapping)
    for name, path in mapping.items():
        assert choice['stages'][name]['summary'] == read(path)['summary']
    assert summary['selected'] == choice['selected'] and summary['stages'] == choice['stages']
    assert summary['rounds'] == read(OUT/'rounds.json')
    for name in ('v1','v2'):
        assert summary['fresh_test'][name] == read(OUT/f'{name}_fresh_test.json')['summary']
    resume = read(OUT/'evaluation_resume.json')
    assert not resume['training_restarted']
    assert resume['selection_sha256'] == hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    assert resume['source_sha256'] == hashlib.sha256((OUT/'sources/resume_v2_evaluation.py').read_bytes()).hexdigest()
    for stage, updates in [('sft',1600),('rsi1',200),('rsi2',200)]:
        logs = read(OUT/stage/'training.json')
        pool = [] if stage=='sft' else read(OUT/f'pool_round{stage[-1]}.json')
        for index, row in enumerate(logs):
            assert row['replay'] == (bool(pool) and index%8==0)
            assert math.isfinite(row['lr']) and row['lr'] > 0
            if stage != 'sft': assert abs(row['lr']-opt.PLAN['rsi_lr']) < 1e-15
            else:
                update=row['update']; warm=opt.PLAN['warmup_updates']; lr=opt.PLAN['sft_lr']
                expected=lr*update/warm if update<=warm else opt.PLAN['sft_min_lr']+.5*(lr-opt.PLAN['sft_min_lr'])*(1+math.cos(math.pi*(update-warm)/(1600-warm)))
                assert abs(row['lr']-expected) < 1e-15
            if index%2:
                assert math.isfinite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip']>=0
        assert sum(r['replay'] for r in logs)==(50 if pool else 0)
    return {'summary_selection_links':'passed','resume_selection_hash':'passed','learning_rate_replay_schedule':'passed'}


def audit():
    protocol = read(OUT/'protocol.json')
    assert all(protocol[k] == v for k, v in opt.PLAN.items())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((OUT/'sources'/name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    question_sets = {split: read(OUT/f'data/{split}/questions.json')
                     for split in ('train', 'feedback1', 'feedback2', 'fresh_test')}
    question_sets['validation'] = read(OUT/'development_questions.json')
    assert question_sets['validation'] == read(ROOT/'improvement/final_v1/joint_questions_at_run.json')['validation']
    old_hashes = opt.history_hashes(ROOT)
    hashes = {}
    questions = {}
    n = 0
    for split, rows in question_sets.items():
        for q in rows:
            digest = opt.digest_image(ROOT/q['image'])
            if split != 'validation':
                assert digest not in old_hashes
                assert digest == q['pixel_sha256']
            assert digest not in hashes or hashes[digest] == split
            hashes[digest] = split
            assert h.pixel_answer(f.exp.load_image(ROOT, q), q)[0] == q['gold'], q['id']
            questions[q['id']] = q
            n += 1
        if split in ('train', 'fresh_test'):
            loc = Counter((tuple(q['cell']), q['target']) for q in rows if q['type'] == 'loc')
            assert len(loc) == 128 and len(set(loc.values())) == 1
            counts = Counter(q['gold'] for q in rows if q['type'] == 'count')
            assert set(counts) == {f'{i}个' for i in range(5)} and len(set(counts.values())) == 1
            exists = Counter(q['gold'] for q in rows if q['type'] == 'exist')
            assert exists['有'] == exists['没有']
    for split in ('rsi', 'eval'):
        questions.update({q['id']: q for q in f.legacy_questions(ROOT, split)})
    paths = list(OUT.glob('*.json')) + list((OUT/'sft').glob('update*/validation.json'))
    paths += [OUT/f'rsi{i}/update200/validation.json' for i in (1, 2)]
    evaluated = 0
    for path in paths:
        result = read(path)
        if not isinstance(result, dict) or 'summary' not in result or 'rows' not in result: continue
        for row in result['rows']:
            q = questions[row['id']]
            assert row['correct'] == (f.canonical(q, row['final']) == q['gold'])
            assert row['raw_correct'] == (f.canonical(q, row['raw']) == q['gold'])
            assert row['valid'] == (f.canonical(q, row['final']) is not None)
            if q['type'] == 'loc': assert abs(row['iou']-h.loc_iou(q, row['final'])) < 1e-10
            assert row['final_verified'] == f.check(f.exp.load_image(ROOT, q), q, row['final'])['passed']
            evaluated += 1
        assert result['summary']['total']['correct'] == sum(r['correct'] for r in result['rows'])
        assert result['summary']['total']['n'] == len(result['rows'])
        for tp in ('loc', 'count', 'exist'):
            subset = [r for r in result['rows'] if r['type'] == tp]
            assert result['summary'][tp] == {'correct': sum(r['correct'] for r in subset),
                                           'n': len(subset), 'valid': sum(r['valid'] for r in subset)}
    selection = read(OUT/'selection.json')
    assert selection['fresh_test_created_at_selection'] is False
    chosen = max(selection['stages'], key=lambda name: opt.key(selection['stages'][name]['summary']))
    assert chosen == selection['selected']
    assert (OUT/'selection.json').stat().st_mtime <= (OUT/'data/fresh_test/questions.json').stat().st_mtime
    sft_selection = read(OUT/'sft_selection.json')
    assert max(sft_selection['stages'], key=lambda k: opt.key(sft_selection['stages'][k]['summary'])) == sft_selection['selected']
    base = read(OUT/'v1_fresh_test.json'); new = read(OUT/'v2_fresh_test.json')
    assert [r['id'] for r in base['rows']] == [r['id'] for r in new['rows']]
    assert len(new['rows']) == 368
    train = {q['id']: q for q in question_sets['train']}
    replay = []
    trace = {}
    for stage, updates in [('sft', 1600), ('rsi1', 200), ('rsi2', 200)]:
        logs = read(OUT/stage/'training.json')
        assert len(logs) == 2*updates
        if stage.startswith('rsi'):
            number = int(stage[-1]); replay = read(OUT/f'pool_round{number}.json')
            evidence = {}
            for prior in range(1, number+1):
                evidence.update({(prior, r['id']): r for r in read(OUT/f'rsi{prior}_feedback.json')['rows']})
            for q in replay:
                assert q['id'] not in train
                round_number = int(q['source'][-1]); row = evidence[(round_number, q['id'])]
                assert row['final_verified'] and not row['verification']['passed']
                assert q['verified_answer'] == row['final']
                assert q['answer_origin'] == f.answer_origin(row)
                assert f.check(f.exp.load_image(ROOT, q), q, q['verified_answer'])['passed']
            assert len({(q['image'], q['target'], q['type']) for q in replay}) == len(replay)
        lookup = {q['id']: q for q in replay}
        for index, log in enumerate(logs):
            assert log['microstep'] == index+1 and log['update'] == index//2+1
            assert math.isfinite(log['loss']) and log['loss'] >= 0
            if log['replay']:
                assert stage != 'sft' and index % 8 == 0
                assert log['answer'] == lookup[log['id']]['verified_answer']
            else:
                assert log['id'] in train
                assert log['answer'] == f.training_answer(train[log['id']], True)
                assert log['type'] == opt.PLAN['task_cycle'][index % 10]
        gradients = read(OUT/stage/'gradient_check.json')
        assert min(gradients.values()) > 0
        trace[stage] = {'optimizer_updates': updates, 'microsteps': len(logs),
                        'bootstrap_microsteps': sum(r['replay'] for r in logs),
                        'unique_bootstrap_images_trained': len({r['id'] for r in logs if r['replay']}),
                        'unique_regular_images': len({r['id'] for r in logs if not r['replay']}),
                        'task_exposures': dict(Counter(r['type'] for r in logs))}
    return {**audit_links(), 'questions_checked': n, 'unique_images': len(hashes), 'prediction_rows_recomputed': evaluated,
            'data_isolation': 'passed', 'pixel_labels': 'passed', 'strict_scores': 'passed',
            'bootstrap_provenance': 'passed', 'training_trace': trace, 'selection': 'passed',
            'sources_match_runtime': 'passed', 'paired_test_ids': 'passed'}


def paired(base, new, tp):
    assert len(base['rows'])==len(new['rows'])
    assert [(r['id'],r['type']) for r in base['rows']]==[(r['id'],r['type']) for r in new['rows']]
    pairs = [(a['correct'], b['correct']) for a, b in zip(base['rows'], new['rows'])
             if tp == 'total' or a['type'] == tp]
    gained = sum(not a and b for a, b in pairs)
    lost = sum(a and not b for a, b in pairs)
    # Two-sided exact binomial McNemar; no independence assumption between models.
    discordant = gained+lost
    p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(gained, lost)+1))/2**discordant) if discordant else 1.
    correct = sum(b for _, b in pairs)
    return {'n': len(pairs), 'gained': gained, 'lost': lost, 'mcnemar_exact_p': p,
            'new_accuracy_wilson95': wilson(correct, len(pairs))}


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = read(OUT/'summary.json')
    stages = summary['stages']
    labels = ['V1 SFT', 'SFT 400', 'SFT 800', 'SFT 1600', 'RSI 1', 'RSI 2']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for task, label, color in [('loc', 'Localization', '#2876A1'), ('total', 'All tasks', '#138A72')]:
        values = [100*s['summary'][task]['correct']/s['summary'][task]['n'] for s in stages.values()]
        axes[0].plot(range(len(stages)), values, 'o-', label=label, color=color)
    axes[0].set_xticks(range(len(stages)), labels, rotation=25, ha='right')
    axes[0].set_title('Development (184)', fontsize=18)
    axes[0].set_ylabel('Correct (%)', fontsize=17)
    axes[0].legend(fontsize=15, loc='upper center', bbox_to_anchor=(.5, -.42), ncol=2, frameon=False)
    for i, (name, label, color) in enumerate([('v1_fresh_test', 'V1 independent', '#69A9C9'),
                                              ('v2_fresh_test', 'V2 independent', '#2876A1'),
                                              ('v2_rich_tool_fresh_test', 'V2 box tool', '#138A72')]):
        s = read(OUT/f'{name}.json')['summary']
        values = [100*s[t]['correct']/s[t]['n'] for t in ('loc', 'count', 'exist')]
        axes[1].bar([j+(i-1)*.25 for j in range(3)], values, width=.25, label=label, color=color)
    axes[1].set_xticks(range(3), ['Localization\n(256)', 'Counting\n(80)', 'Existence\n(32)'])
    axes[1].set_title('Fresh paired test (368)', fontsize=18)
    axes[1].legend(fontsize=15, loc='upper center', bbox_to_anchor=(.5, -.42), frameon=False, ncol=2)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.set_ylim(0, 105); ax.tick_params(labelsize=14)
    fig.subplots_adjust(bottom=.40, top=.88, wspace=.28, left=.07, right=.98)
    fig.savefig(ROOT/'figures/optimized_v2_accuracy.png', dpi=200)
    fig.savefig(ROOT/'figures/optimized_v2_accuracy.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--audit', action='store_true'); p.add_argument('--plot', action='store_true')
    p.add_argument('--links-only',action='store_true',help='Recheck JSON/source/schedule links against a completed full audit')
    args = p.parse_args()
    if args.links_only:
        path=ROOT/'analysis/optimization_v2.json'; result=read(path)
        assert 'audit' in result
        result['audit'].update(audit_links()); h.write_json(path,result)
        print(json.dumps(result['audit'],ensure_ascii=False)); return
    summary = read(OUT/'summary.json')
    base, new = read(OUT/'v1_fresh_test.json'), read(OUT/'v2_fresh_test.json')
    result = {'summary': summary, 'paired_comparison': {tp: paired(base, new, tp) for tp in ('loc', 'count', 'exist', 'total')}}
    if args.audit: result['audit'] = audit()
    h.write_json(ROOT/'analysis/optimization_v2.json', result)
    if args.plot: draw()
    print(json.dumps(result.get('audit', result['paired_comparison']), ensure_ascii=False))


if __name__ == '__main__': main()
