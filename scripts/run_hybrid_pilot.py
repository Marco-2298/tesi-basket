"""Esegue06 in kernel separato e conserva il notebook anche in caso di errore."""
import argparse
import json
import os
import nbformat
import pandas as pd
from nbclient import NotebookClient
from project_config import ROOT
from thesis_tools import new_run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clip',default='clip_01');a=ap.parse_args()
    m=pd.read_csv(ROOT/'annotations/manifest.csv')
    if a.clip not in set(m.clip_id):ap.error('Clip non presente nel manifest')
    out=new_run(ROOT,'hybrid_execution')
    nb=nbformat.read(ROOT/'06_NBA_tracking_recupero_giocatori.ipynb',as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell("print('HYBRID_RUN='+str(RUN_DIR))"))
    try:
        NotebookClient(nb,timeout=None,resources={'metadata':{'path':str(ROOT)}}).execute(
            env=dict(os.environ,NBA_PROJECT_DIR=str(ROOT),NBA_CLIP_ID=a.clip))
    finally:
        nbformat.write(nb,out/(a.clip+'_executed.ipynb'))
        print('Notebook eseguito/log:',out,flush=True)
    result=None
    for output in nb.cells[-1].get('outputs',[]):
        for line in output.get('text','').splitlines():
            if line.startswith('HYBRID_RUN='):result=line.split('=',1)[1]
    if result is None:raise RuntimeError('Esecuzione senza cartella risultato')
    (out/'result.json').write_text(json.dumps(dict(clip_id=a.clip,run_dir=result),indent=2),encoding='utf-8')
    print('Output variante06:',result,flush=True)
    print('Manifest baseline non modificato.',flush=True)


if __name__=='__main__':main()
