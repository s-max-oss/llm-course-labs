"""Generate from a saved model without retraining it."""
import argparse
import sys
from pathlib import Path
import torch
from analyze_results import load_model
from min_llm import ROOT, CharTokenizer


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/baseline/model.pt')
    p.add_argument('--prompt',default='春')
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--top_k',type=int,default=20,help='0 disables top-k truncation')
    p.add_argument('--max_new_tokens',type=int,default=200)
    p.add_argument('--seed',type=int,default=1001)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--no_repeat_ngram',type=int,choices=[0,3],default=0)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not a.prompt or a.max_new_tokens<1 or a.threads<1 or a.top_k<0:
        p.error('Use a nonempty prompt, positive token/thread counts, and top_k >= 0')
    torch.set_num_threads(a.threads)
    model,saved=load_model(a.checkpoint)
    tok=CharTokenizer(''.join(saved['chars']))
    unknown=sorted(set(a.prompt)-set(tok.chars))
    if unknown:p.error(f'Prompt contains characters outside this corpus: {unknown}')
    ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long)
    generated=model.generate(ids,a.max_new_tokens,a.temperature,a.top_k or None,
                             torch.Generator().manual_seed(a.seed),a.no_repeat_ngram)
    text=tok.decode(generated[0].tolist())
    print(text)
    if a.output:a.output.write_text(text+'\n',encoding='utf-8')


if __name__=='__main__':main()
