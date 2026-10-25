"""Renderer canonico della skill, con conversione PDF Word su Windows.

Il runtime Windows non include LibreOffice. Non si usa LibreOffice desktop.
Le PNG sono prodotte dalla rasterizzazione standard del renderer della skill.
"""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
skill=Path(r'C:\Users\Marco\.codex\plugins\cache\openai-primary-runtime\documents\26.1013.11914\skills\documents')
spec=importlib.util.spec_from_file_location('canonical_docx_renderer',skill/'render_docx.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)


def word_to_pdf(input_path,user_profile,convert_tmp_dir,stem,verbose=False):
    out=Path(convert_tmp_dir)/(stem+'.pdf')
    command=['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/export_docx_pdf.ps1'),'-InputPath',str(Path(input_path).resolve()),'-OutputPath',str(out)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    if result.returncode or not out.is_file():raise RuntimeError(result.stdout+'\n'+result.stderr)
    return str(out),'PDF exported with Microsoft Word COM; PNG rasterization by canonical render_docx.py'


if __name__=='__main__':
    temp=ROOT/'reports/word_qa_20261020/temp'
    temp.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp)
    tempfile.tempdir=str(temp)
    renderer.convert_to_pdf=word_to_pdf
    poppler=Path(r'C:\Users\Marco\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler')
    executable=next(poppler.rglob('pdfinfo.exe'))
    os.environ['PATH']=str(executable.parent)+os.pathsep+os.environ.get('PATH','')
    qa_dir=Path(os.environ.get('DOCX_QA_DIR',str(ROOT/'reports/word_qa_20261020')))
    sys.argv=['render_docx.py',str(ROOT/'exports/Incontro_professore_tracking_basket.docx'),
              '--output_dir',str(qa_dir),'--dpi','130','--emit_pdf']
    renderer.main()
