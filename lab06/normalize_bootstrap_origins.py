"""Refine run-15397 provenance labels, preserving every answer and raw file.

The running experiment used 'free_generation' for unconstrained decoding after
both bbox feedback and self-check. That name is ambiguous about tool use. This
annotation-only migration derives exact paths from unchanged inference memory.
"""
import ast
import hashlib
from collections import Counter
from pathlib import Path

import hw6_rsi_localize as h
import finish_grounding as final


def semantic_code(source):
    class RemoveAnnotation(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            return None if node.name=='answer_origin' else self.generic_visit(node)
        def visit_Dict(self,node):
            kept=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value=='answer_origin')]
            node.keys=[k for k,v in kept];node.values=[v for k,v in kept]
            return self.generic_visit(node)
    return ast.dump(RemoveAnnotation().visit(ast.parse(source)),include_attributes=False)


def main():
    root=Path(__file__).resolve().parent;folder=root/'improvement/final_v1'
    runtime=(folder/'source_at_run.py').read_text(encoding='utf-8')
    submitted=(root/'finish_grounding.py').read_text(encoding='utf-8')
    assert semantic_code(runtime)==semantic_code(submitted), 'Unexpected training/inference change'
    memory=h.read_json(folder/'rsi_memory.json')['records']
    lookup={(r['round'],r['id']):r for r in memory}
    backup=root/'provenance/origin_labels_raw_15397';backup.mkdir(exist_ok=True)
    summary={}
    for number in (1,2):
        path=folder/f'pool_round{number}.json';pool=h.read_json(path);counts=Counter();changed=0
        for q in pool:
            if 'verified_answer' not in q:continue
            r=lookup[(int(q['source'].removeprefix('rsi')),q['id'])]
            assert q['verified_answer']==r['final'] and r['final_verified']
            origin=final.answer_origin(r);counts[origin]+=1
            if q['answer_origin']!=origin:q['answer_origin']=origin;changed+=1
        if changed:
            raw=path.read_bytes()
            if (backup/path.name).exists():assert (backup/path.name).read_bytes()==raw
            else:(backup/path.name).write_bytes(raw)
            before=h.read_json(path)
            without=lambda rows:[{k:v for k,v in q.items() if k!='answer_origin'} for q in rows]
            assert without(before)==without(pool)
            h.write_json(path,pool)
        summary[str(number)]={'origins':dict(counts),'labels_changed':changed,'answers_and_other_fields_unchanged':True}
    h.write_json(root/'analysis/bootstrap_origins.json',{
        'reason':'Disambiguate bbox-assisted and self-check decoding from raw independent generation; no prediction or training target changed',
        'runtime_source_sha256':hashlib.sha256((folder/'source_at_run.py').read_bytes()).hexdigest(),
        'submitted_source_sha256':hashlib.sha256((root/'finish_grounding.py').read_bytes()).hexdigest(),
        'source_ast_equal_except_origin_annotation':True,'raw_pool_backups':str(backup.relative_to(root)),
        'rounds':summary})
    print(summary)


if __name__=='__main__':main()
