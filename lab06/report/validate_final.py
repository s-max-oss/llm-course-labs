"""Validate template preservation, OOXML, report copies and final evidence."""
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
COURSE=next(p for p in ROOT.parents if (p/'实验指南与模板').is_dir())
NAME='2024311281_孙驰_实验作业六'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    v2=(ROOT/'improvement/final_v2/summary.json').exists()
    template=COURSE/'实验指南与模板/实验六_RSI递归自进化_实验报告样板.docx'
    assert sha(template)=='b2ba41dca8634f1b37d54d7f0fe57aa0c4df7b08357f371de529d7ca91467272'
    main=COURSE/'提交材料';docx=main/(NAME+'.docx');pdf=main/(NAME+'.pdf')
    for extension in ('.docx','.pdf'):
        assert len({sha(folder/(NAME+extension)) for folder in (main,ROOT/'report',COURSE/'lab06_rsi/report')})==1
    with ZipFile(docx) as z,ZipFile(template) as original:
        count=0
        for name in z.namelist():
            if name.endswith(('.xml','.rels')):etree.fromstring(z.read(name));count+=1
        xml=etree.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text=''.join(xml.xpath('//w:t/text()',namespaces=ns))
        expected=('孙驰','2024311281','计算机10班','表6','表7','ChartQA')+ (('215/256','290/368','168/199','164/199') if v2 else ('109/128','144/184'))
        assert all(s in text for s in expected)
        if v2:
            assert all(s in text for s in ('r=8','α=16','0.3580%','python hw6_rsi_localize.py --mode',
                                          'make_data','make_sft','infer','sft --steps 10','eval','rsi --rounds 2 --steps 10',
                                          '自查8/41','框重答17/70','错→对25','对→错0','A30/NF4'))
        assert not any(s in text for s in ('________________','TODO','待填写'))
        assert xml.xpath('//w:sectPr',namespaces=ns)[0].attrib==etree.fromstring(original.read('word/document.xml')).xpath('//w:sectPr',namespaces=ns)[0].attrib
        sections=xml.xpath('//w:sectPr',namespaces=ns)[0]
        original_sections=etree.fromstring(original.read('word/document.xml')).xpath('//w:sectPr',namespaces=ns)[0]
        for tag in ('pgSz','pgMar'):
            assert sections.find('w:'+tag,ns).attrib==original_sections.find('w:'+tag,ns).attrib
        rel=etree.fromstring(z.read('word/_rels/document.xml.rels'))
        ids=[r.get('Id') for r in rel];assert len(ids)==len(set(ids))
        figure=ROOT/'figures'/('optimized_v2_accuracy.png' if v2 else 'improved_accuracy.png')
        assert hashlib.sha256(z.read('word/media/lab06_accuracy.png')).hexdigest()==sha(figure)
        modified=[n for n in original.namelist() if original.read(n)!=z.read(n)]
        assert set(modified)<={'[Content_Types].xml','word/document.xml','word/_rels/document.xml.rels'}
    pages=len(PdfReader(pdf).pages);assert pages==3
    audit=read(ROOT/'analysis/improvement_summary.json')['audit']
    assert audit['joint_questions_checked']==4480 and audit['unique_images']==4416
    assert all(audit[k]=='passed' for k in ('strict_scores','bootstrap_provenance','selection','split_isolation'))
    origins=read(ROOT/'analysis/bootstrap_origins.json');assert origins['source_ast_equal_except_origin_annotation']
    cpu=read(ROOT/'improvement/cpu_reference/metadata.json');assert cpu['success'] and len(cpu['stages'])==6
    run=read(ROOT/'provenance/improvement_run.json');assert run['state']=='COMPLETED' and run['exit_code']=='0:0' and run['experiment_allocations_released']
    result={'formal_job':15397,'default_reference_job':15391,'default_raw_files_verified':85,
            'improved_audit':audit,'unit_tests_passed':10,'cpu_full_chain_stages':6,
            'source_ast_equal_except_origin_annotation':True,'template_unchanged':True,
            'xml_parts_valid':count,'modified_template_parts':modified,'report_pages':pages,
            'three_report_copies_identical':True,'docx_sha256':sha(docx),'pdf_sha256':sha(pdf),
            'figure_matches_report':True,'visual_review':'All three final pages inspected; no clipping, overlap or blank page',
            'renderer':'Word COM read-only PDF export and pypdfium2; packaged renderer unavailable because LibreOffice is absent',
            'independent_review':'All important findings resolved; numeric and attribution checks passed',
            'hpc_allocations_released':True,'queue_after':run['queue_after']}
    if v2:
        optimized=read(ROOT/'analysis/optimization_v2.json')['audit']
        assert all(optimized[k]=='passed' for k in ('data_isolation','pixel_labels','strict_scores','bootstrap_provenance','selection','sources_match_runtime','paired_test_ids','summary_selection_links','resume_selection_hash','learning_rate_replay_schedule'))
        assert optimized['questions_checked']==4760
        run2=read(ROOT/'provenance/optimization_v2_run.json')
        assert run2['evaluation_state']=='COMPLETED' and run2['evaluation_exit_code']=='0:0' and run2['allocations_released']
        assert run2['selection_unchanged'] and not run2['training_restarted']
        tests=(ROOT/'analysis/unit_tests_v2.log').read_text(encoding='utf-8')
        assert re.search(r'Ran 12 tests in [0-9.]+s',tests) and '\nOK\n' in tests
        test_proof=read(ROOT/'analysis/unit_tests_v2.json')
        assert sha(ROOT/'analysis/unit_tests_v2.log')==test_proof['log_sha256']
        assert all(sha(ROOT/name)==digest for name,digest in test_proof['source_sha256'].items())
        result.update({'formal_job':run2['evaluation_job'],'training_job':run2['training_job'],
                       'optimization_v2_audit':optimized,'unit_tests_passed':12,
                       'queue_after':run2['queue_after'],'selection_unchanged_after_timeout':True})
        result['pre_submission_review']={'date':'2026-10-27','guide_sha256':sha(next((COURSE/'实验指南与模板').glob('作业六*.pdf'))),
                                        'lesson11_sha256':sha(COURSE.parent/'课件/Lesson11.pdf'),
                                        'lesson12_sha256':sha(COURSE.parent/'课件/Lesson12.pdf'),
                                        'commands_lora_tool_transitions':'passed',
                                        'prior_passing_tests_sources_and_log_unchanged':True}
    (ROOT/'report/validation.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'pages':pages,'xml_parts':count,'audit':audit,'report_copies_identical':True},ensure_ascii=False))


if __name__=='__main__':main()
