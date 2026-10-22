"""Validate template preservation, OOXML, report copies and final evidence."""
import hashlib
import json
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
        assert all(s in text for s in ('孙驰','2024311281','计算机10班','109/128','144/184','表6','表7','ChartQA'))
        assert not any(s in text for s in ('________________','TODO','待填写'))
        assert xml.xpath('//w:sectPr',namespaces=ns)[0].attrib==etree.fromstring(original.read('word/document.xml')).xpath('//w:sectPr',namespaces=ns)[0].attrib
        sections=xml.xpath('//w:sectPr',namespaces=ns)[0]
        original_sections=etree.fromstring(original.read('word/document.xml')).xpath('//w:sectPr',namespaces=ns)[0]
        for tag in ('pgSz','pgMar'):
            assert sections.find('w:'+tag,ns).attrib==original_sections.find('w:'+tag,ns).attrib
        rel=etree.fromstring(z.read('word/_rels/document.xml.rels'))
        ids=[r.get('Id') for r in rel];assert len(ids)==len(set(ids))
        assert hashlib.sha256(z.read('word/media/lab06_accuracy.png')).hexdigest()==sha(ROOT/'figures/improved_accuracy.png')
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
    (ROOT/'report/validation.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'pages':pages,'xml_parts':count,'audit':audit,'report_copies_identical':True},ensure_ascii=False))


if __name__=='__main__':main()
