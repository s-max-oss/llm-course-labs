from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from docx import Document
from pypdf import PdfReader


PLACEHOLDERS = (
    "请用自己的语言",
    "简述知识蒸馏",
    "此处粘贴",
    "粘贴 runs_distill",
    "对比表 6 中",
    "最优组合是哪一组",
    "对比表 8 中",
    "总结本次实验",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def all_text(document: Document) -> str:
    pieces = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                pieces.append(cell.text)
    return "\n".join(pieces)


def validate(report: Path, template: Path, pdf: Path | None = None) -> dict:
    document = Document(report)
    text = all_text(document)
    checks = {
        "template_sha256_unchanged": sha256(template) == "A2BAF61699F7F8AC11EE6EEAA5EBAE38DE12715657F775A9E987A3AEC6127217",
        "table_count_12": len(document.tables) == 12,
        "single_section": len(document.sections) == 1,
        "identity_present": "孙驰 / 2024311281" in text,
        "model_counts_present": "737,280" in text and "147,456" in text,
        "baseline_metrics_present": "1.2722" in text and "1.2107" in text,
        "all_thought_answers_present": text.count("答：") >= 6,
        "no_placeholders": not any(marker in text for marker in PLACEHOLDERS),
    }
    with zipfile.ZipFile(report) as archive:
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
        checks["three_figures_embedded"] = len(media) >= 3
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"报告校验失败: {failed}")
    result = {
        "report": str(report.resolve()),
        "report_sha256": sha256(report),
        "template": str(template.resolve()),
        "template_sha256": sha256(template),
        "page_geometry_inches": {"width": 8.27, "height": 11.69},
        "checks": checks,
        "passed": True,
    }
    if pdf is not None:
        reader = PdfReader(pdf)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        pdf_checks = {
            "pdf_nonempty": pdf.stat().st_size > 0,
            "pdf_page_count_6": len(reader.pages) == 6,
            "pdf_identity_present": "2024311281" in pdf_text,
            "pdf_key_metrics_present": "1.2722" in pdf_text and "1.2107" in pdf_text,
        }
        if not all(pdf_checks.values()):
            failed = [name for name, passed in pdf_checks.items() if not passed]
            raise AssertionError(f"PDF 校验失败: {failed}")
        result["pdf"] = str(pdf.resolve())
        result["pdf_sha256"] = sha256(pdf)
        result["pdf_page_count"] = len(reader.pages)
        result["pdf_checks"] = pdf_checks
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.report, args.template, args.pdf)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
