"""Render Word-exported PDF when packaged LibreOffice renderer is unavailable."""
import argparse
from pathlib import Path

import pypdfium2 as pdfium

p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
doc=pdfium.PdfDocument(args.pdf)
for index in range(len(doc)):
    page=doc[index];bitmap=page.render(scale=2);bitmap.to_pil().save(args.output/f'page-{index+1}.png')
    bitmap.close();page.close()
print(f'{len(doc)} pages rendered');doc.close()
