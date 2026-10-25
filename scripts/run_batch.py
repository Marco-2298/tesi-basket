"""Colab: python scripts/run_batch.py annotations/manifest.csv --execute

Default: sola verifica dei percorsi. Kernel separato per clip; nessuna inferenza
se un CSV esplicitamente registrato è già valido. Output notebook senza overwrite.
"""
import argparse
import os
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from thesis_tools import load_tracks, new_run, fingerprint

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--clip',help='Esegue soltanto la clip indicata, utile per il pilot')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=args.manifest.resolve()
    m=pd.read_csv(manifest).fillna('')
    if len(m)!=7 or m.clip_id.duplicated().any():
        raise ValueError('Manifest: richieste sette clip univoche')
    if args.clip and args.clip not in set(m.clip_id):raise ValueError('Clip non presente nel manifest')
    selected=m[m.clip_id.eq(args.clip)] if args.clip else m
    for r in selected.itertuples():
        if not (root/r.video).is_file(): raise FileNotFoundError(root/r.video)
        if r.tracks_csv:
            d=load_tracks(root/r.tracks_csv)
            if d.empty: raise ValueError(f'CSV vuoto: {r.tracks_csv}')
            if set(d.frame)!=set(range(int(r.frames))):raise ValueError(f'{r.clip_id}: tracking non copre i frame della clip')
            missing=[name for name in ['positions_csv','homographies_json'] if not getattr(r,name,'') or not (root/getattr(r,name)).is_file()]
            print(r.clip_id, 'tracking disponibile:',r.tracks_csv)
            if missing:
                print('  Geometria da completare separatamente, senza rilanciare SAM2:',', '.join(missing))
            continue
        print(r.clip_id,'richiede baseline')
        if not args.execute: continue
        import nbformat
        from nbclient import NotebookClient
        out=new_run(root,'batch')
        nb=nbformat.read(root/'01_NBA_tracking_revisionato.ipynb',as_version=4)
        nb.cells.append(nbformat.v4.new_code_cell("print('NBA_RUN_RESULT='+json.dumps({'run_dir':str(RUN_DIR)}))"))
        env=dict(os.environ,NBA_PROJECT_DIR=str(root),NBA_CLIP_ID=str(r.clip_id))
        try:
            NotebookClient(nb,timeout=None,resources={'metadata':{'path':str(root)}}).execute(env=env)
        finally:
            nbformat.write(nb,out/f'{r.clip_id}_executed.ipynb')
        result=None
        for output in nb.cells[-1].get('outputs',[]):
            for line in output.get('text','').splitlines():
                if line.startswith('NBA_RUN_RESULT='):result=json.loads(line.split('=',1)[1])
        if not result:raise RuntimeError('Notebook terminato senza indicazione della cartella output')
        run=Path(result['run_dir']).resolve()
        if not run.is_relative_to(root/'outputs'):raise ValueError('Output fuori dal progetto')
        if (run/'input_video_sha256.txt').read_text().strip()!=fingerprint(root/r.video):
            raise ValueError('Video elaborato diverso dal manifest')
        registered=dict(tracks_csv=run/f'{r.clip_id}_tracks.csv',positions_csv=run/'nba_positions_2d.csv',
                        homographies_json=run/'homographies_image_to_court_ft.json')
        for name,p in registered.items():
            if not p.is_file():raise FileNotFoundError(p)
        d=load_tracks(registered['tracks_csv'])
        if set(d.frame)!=set(range(int(r.frames))):raise ValueError('Tracking incompleto: nessuna registrazione')
        history=manifest.parent/'history';history.mkdir(exist_ok=True)
        shutil.copy2(manifest,history/('manifest_before_batch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.csv'))
        for name,p in registered.items():m.loc[m.clip_id.eq(r.clip_id),name]=p.relative_to(root).as_posix()
        temp=manifest.with_suffix('.tmp');m.to_csv(temp,index=False);os.replace(temp,manifest)
        print('Notebook eseguito:',out,'; percorsi registrati nel manifest')

if __name__=='__main__': main()
