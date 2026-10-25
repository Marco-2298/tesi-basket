"""Pacchetto delle sole inferenze mancanti, senza venv/output/cache duplicate."""
from pathlib import Path
from datetime import datetime,timezone
import json
import zipfile
from project_config import ROOT,load_config
from thesis_tools import fingerprint


def main():
    c=load_config();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ROOT/'exports';destination.mkdir(exist_ok=True)
    name='NBA_Tracking_Run_'+stamp
    files=[ROOT/'01_NBA_tracking_revisionato.ipynb',ROOT/'05_NBA_palla_lavagna.ipynb',ROOT/'06_NBA_tracking_recupero_giocatori.ipynb',ROOT/'AVVIO_COLAB.md',ROOT/'PALLA_E_LAVAGNA.md',ROOT/'RECUPERO_GIOCATORI.md']
    files+=list((ROOT/'scripts').glob('*.py'))
    files+=list((ROOT/'configs').glob('*.json'))
    files+=[ROOT/'annotations'/s for s in ['manifest.csv','annotations.csv','tasks.csv','scene_intervals.csv','player_teams.csv']]
    files+=sorted(c['clips_dir'].glob('*.mov'))
    files+=[c['weights'].parent/s for s in ['weights.onnx','inference_config.json','class_names.txt','model_config.json']]
    files=sorted(set(files))
    records=[dict(path=p.relative_to(ROOT).as_posix(),sha256=fingerprint(p),bytes=p.stat().st_size) for p in files]
    target=destination/(name+'.zip')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,name+'/'+p.relative_to(ROOT).as_posix())
        z.writestr(name+'/BUNDLE_MANIFEST.json',json.dumps(records,indent=2))
    evidence=dict(zip=target.name,sha256=fingerprint(target),bytes=target.stat().st_size,root_folder=name,files=records)
    (destination/(name+'_manifest.json')).write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(target,flush=True);print('MB:',round(target.stat().st_size/1e6,2))


if __name__=='__main__':main()
