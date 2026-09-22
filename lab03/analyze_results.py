"""Verify completed CPU experiments, then compare fixed-model sampling choices."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from min_llm import (ROOT, Config, MiniGPT, CharTokenizer, build_corpus, digest, evaluate,
                     experiment_configs, parameter_counts, save_samples, write_json)


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = MiniGPT(len(saved['chars']), Config(**saved['config']))
    model.load_state_dict(saved['state_dict']); model.eval()
    return model, saved


def text_statistics(samples, corpus):
    corpus10 = {corpus[i:i+10] for i in range(len(corpus)-9)}
    rows = []
    for s in samples:
        text = s['text'][len(s['prompt']):]
        triples = [text[i:i+3] for i in range(len(text)-2)]
        tens = [text[i:i+10] for i in range(len(text)-9)]
        rows.append({'prompt': s['prompt'], 'seed': s['seed'], 'new_characters': len(text),
                     'unique_character_ratio': len(set(text))/len(text),
                     'repeated_trigram_fraction': 1-len(set(triples))/len(triples),
                     'corpus_10gram_fraction': sum(x in corpus10 for x in tens)/len(tens)})
    return {'samples': rows, 'mean': {k: sum(r[k] for r in rows)/len(rows)
            for k in ['unique_character_ratio','repeated_trigram_fraction','corpus_10gram_fraction']}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--verification-output', type=Path, help='Separate audit file for another machine')
    args = p.parse_args(); root = args.results
    torch.set_num_threads(args.threads)
    corpus = build_corpus(); tok = CharTokenizer(corpus)
    data = torch.tensor(tok.encode(corpus), dtype=torch.long)
    rows, audit = {}, {}
    for name, config in experiment_configs().items():
        folder = root/name
        r = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert r['signature']['config'] == asdict(config)
        assert r['signature']['source_sha256'] == digest(ROOT/'min_llm.py')
        assert r['signature']['corpus_sha256'] == digest(ROOT/'corpus.txt')
        assert r['signature']['device'] == 'cpu'
        for filename, sha in r['artifacts'].items(): assert digest(folder/filename) == sha, filename
        assert len(r['losses']) == config.iters and all(math.isfinite(v) for v in r['losses'])
        assert r['last_batch_loss'] == r['losses'][-1]
        assert abs(r['last100_mean_loss']-sum(r['losses'][-100:])/100) < 1e-12
        model, saved = load_model(folder/'model.pt')
        assert saved['step'] == config.iters and saved['signature'] == r['signature']
        assert saved['chars'] == tok.chars
        assert model.head.weight is model.tok_emb.weight
        assert sum(p.numel() for p in model.parameters()) == r['parameters']['exact']
        replay = evaluate(model, data)
        assert abs(replay-r['training_window_loss']) < 2e-4, (name, replay, r['training_window_loss'])
        audit[name] = {'passed': True, 'replayed_training_window_loss': replay,
                       'absolute_difference': abs(replay-r['training_window_loss'])}
        rows[name] = r
    base1000 = json.loads((root/'baseline/step1000.json').read_text(encoding='utf-8'))
    model1000, saved1000 = load_model(root/'baseline/model_step1000.pt')
    assert saved1000['step'] == 1000
    assert saved1000['signature'] == rows['baseline']['signature']
    assert abs(evaluate(model1000, data)-base1000['training_window_loss']) < 2e-4
    assert rows['baseline']['losses'][999] == base1000['last_batch_loss']
    assert abs(sum(rows['baseline']['losses'][900:1000])/100-base1000['last100_mean_loss']) < 1e-12
    write_json(args.verification_output or root/'verification.json',
               {'groups': audit, 'baseline1000': 'passed', 'passed': True})
    if args.verify_only:
        print('All 9 trained models plus baseline1000 passed artifact and CPU replay checks.')
        return

    sampling = root/'sampling'; sampling.mkdir(exist_ok=True)
    save_samples(model1000, tok, sampling/'generated_baseline1000.txt')
    model, _ = load_model(root/'baseline/model.pt')
    settings = [('temp05_k20', .5, 20, 0), ('temp10_k20', 1., 20, 0),
                ('temp15_k20', 1.5, 20, 0), ('temp10_k5', 1., 5, 0),
                ('temp10_all', 1., tok.vocab_size, 0), ('bonus_no_repeat3', 1., 20, 3)]
    sampling_rows = {}
    for name, temp, k, repeat in settings:
        samples = save_samples(model, tok, sampling/f'{name}.txt', prompts=('春','月','山'),
                               temperature=temp, top_k=k, no_repeat_ngram=repeat)
        stats = text_statistics(samples, corpus)
        sampling_rows[name] = {'temperature': temp, 'top_k': k, 'no_repeat_ngram': repeat,
                               **stats}
        if repeat:
            for sample in samples:
                text = sample['text']; triples = [text[i:i+3] for i in range(len(text)-2)]
                assert len(set(triples)) == len(triples)
    write_json(sampling/'statistics.json', sampling_rows)
    summary = {'experiments': rows, 'baseline1000': base1000, 'sampling': sampling_rows,
               'question2_parameters': parameter_counts(tok.vocab_size, Config(n_embd=256,n_head=8,n_layer=3))}
    write_json(root/'results.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    pairs=[('Learning rate',['exp1_lr_high','exp1_lr_low']),('Layers',['exp2_shallow','exp2_deep']),
           ('Width (32 dims/head)',['exp3_small','exp3_large']),('Context',['exp4_short']),
           ('Position embedding',['exp5_no_pos'])]
    for ax,(title,names) in zip(axes.flat,pairs):
        for name in ['baseline']+names:
            points=[v for v in rows[name]['evaluations'] if v['step']<=1000]
            ax.plot([v['step'] for v in points],[v['training_window_loss'] for v in points],label=name)
        ax.set(title=title,xlabel='Step',ylabel='Fixed training-window loss');ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    base=rows['baseline']
    axes.flat[5].plot(range(1,2001),base['losses'],linewidth=.6)
    axes.flat[5].set(title='Full baseline (2000 steps)',xlabel='Step',ylabel='Training batch loss')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    lines=['# 实验三完整结果','', '所有loss均来自训练语料，不是测试集指标。参数、配置及原始记录见各组result.json。', '',
           '|配置|步数|参数量|末步训练loss|最后100步均值|固定训练窗口loss|训练秒|',
           '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"|baseline_1000|1000|{base['parameters']['exact']}|{base1000['last_batch_loss']:.6f}|"
              f"{base1000['last100_mean_loss']:.6f}|{base1000['training_window_loss']:.6f}|{base1000['training_seconds']:.2f}|"]
    for name,r in rows.items():
        lines.append(f"|{name}|{len(r['losses'])}|{r['parameters']['exact']}|{r['last_batch_loss']:.6f}|"
                     f"{r['last100_mean_loss']:.6f}|{r['training_window_loss']:.6f}|{r['training_seconds']:.2f}|")
    lines += ['', '![对照曲线](comparison.png)', '',
              '计时仅含取批、前后向、优化器更新及loss读取，不含评估、保存、绘图、采样和排队。各组CPU同型号、同线程数；并行作业和共享节点负载仍会影响耗时。', '',
              '上下文32组每步训练字符为基线的1/4；嵌入维度组保持每头32维，因此同步调整头数。', '',
              '## 采样与选做结果', '',
              '|配置|温度|top-k|重复3-gram比例|语料10-gram匹配率|不同字符比例|',
              '|---|---:|---:|---:|---:|---:|']
    for name,r in sampling_rows.items():
        m=r['mean'];lines.append(f"|[{name}](sampling/{name}.txt)|{r['temperature']}|{r['top_k']}|"
            f"{m['repeated_trigram_fraction']:.2%}|{m['corpus_10gram_fraction']:.2%}|{m['unique_character_ratio']:.2%}|")
    lines += ['', '比例均按生成的200字符计算（不含提示词），保留标点和换行；三段取平均。它们描述重复、多样性与语料重合，不是语义质量或原创性的充分判据。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified all models; sampling and comparison materials saved.')


if __name__ == '__main__':
    main()
