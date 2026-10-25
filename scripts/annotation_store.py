"""Salvataggio validato delle sole etichette manuali, con cronologia."""
import csv
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil

FIELDS=['clip_id','frame','player_id','visibility','annotatable','x_px','y_px','sequence_id','note']
VISIBILITY=('visible','partial','occluded','out_of_view')


def required_players(task):
    return [x.strip() for x in task.get('required_player_ids','').split(';') if x.strip()]


def task_complete(task,annotations):
    rows=[r for r in annotations if r['clip_id']==task['clip_id'] and int(r['frame'])==int(task['frame'])]
    ids={r['player_id'] for r in rows}
    if task.get('sequence_id') and 'required_player_ids' in task:
        required=required_players(task)
        return len(required)==2 and set(required)<=ids
    return len(ids)>=int(task.get('target_count') or 1)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def validate_row(row, width, height):
    if not str(row['player_id']).strip(): raise ValueError('Inserire una identità umana, es. p01')
    if row['visibility'] not in VISIBILITY: raise ValueError('Selezionare la visibilità')
    if str(row['annotatable']) not in {'0','1'}: raise ValueError('Annotabilità deve essere 0 oppure 1')
    if str(row['annotatable'])=='1':
        x,y=float(row['x_px']),float(row['y_px'])
        if not all(map(math.isfinite,[x,y])) or not (0<=x<width and 0<=y<height):
            raise ValueError('Coordinate fuori dall’immagine')
        if row['visibility']=='out_of_view': raise ValueError('Un giocatore fuori immagine non ha punto annotabile')
    elif str(row.get('x_px','')).strip() or str(row.get('y_px','')).strip():
        raise ValueError('Non annotabile: lasciare entrambe le coordinate vuote')


def upsert(path,row,width,height):
    path=Path(path)
    validate_row(row,width,height)
    key=lambda r:(r['clip_id'],int(r['frame']),str(r['player_id']).strip())
    rows=read_rows(path) if path.exists() else []
    keys=[key(r) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('CSV esistente con etichette duplicate: correggerlo prima di salvare')
    target=key(row)
    rows=[r for r in rows if key(r)!=target]
    rows.append({k:row.get(k,'') for k in FIELDS})
    rows.sort(key=lambda r:(r['clip_id'],int(r['frame']),r['player_id']))
    if path.exists():
        history=path.parent/'history';history.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copy2(path,history/f'annotations_{stamp}.csv')
    temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    os.replace(temp,path)


def validate_dataset(manifest,tasks,annotations):
    """Controllo finale prima dell'analisi, non misura qualità umana delle etichette."""
    clips={r['clip_id']:r for r in manifest}
    errors=[];keys=set()
    for r in annotations:
        try:
            if r['clip_id'] not in clips:raise ValueError('Clip sconosciuta')
            c=clips[r['clip_id']];frame=int(r['frame'])
            if frame<0 or frame>=int(c['frames']):raise ValueError('Frame fuori intervallo')
            validate_row(r,int(c['width']),int(c['height']))
            key=(r['clip_id'],frame,r['player_id'])
            if key in keys:raise ValueError('Etichetta duplicata')
            keys.add(key)
        except (ValueError,KeyError,TypeError) as e:errors.append(f'{r.get("clip_id")}/{r.get("frame")}: {e}')
    covered={(r['clip_id'],int(r['frame'])) for r in annotations}
    missing=[(r['clip_id'],r['frame']) for r in tasks if not task_complete(r,annotations)]
    if missing:errors.append(f'{len(missing)} frame del piano senza tutte le etichette richieste')
    if not {'validation','test'}<=set(r.get('split','') for r in manifest):errors.append('Assegnare split validation/test per clip')
    if any(r.get('split','') not in {'validation','test'} for r in manifest):errors.append('Ogni clip deve avere uno split valido')
    # Nelle sequenze confrontare identità manuali presenti per ogni frame; annotare
    # anche i punti non localizzabili per distinguere mancata etichetta e occlusione.
    seq={}
    for t in tasks:
        if t['sequence_id']:seq.setdefault((t['clip_id'],t['sequence_id']),set()).add(int(t['frame']))
    for (clip,sid),frames in seq.items():
        planned=[t for t in tasks if t['clip_id']==clip and t['sequence_id']==sid]
        explicit='required_player_ids' in planned[0]
        rr=[r for r in annotations if r['clip_id']==clip and (int(r['frame']) in frames if explicit else r['sequence_id']==sid)]
        players=set(required_players(planned[0])) if explicit else {r['player_id'] for r in rr}
        if explicit and len(players)!=2:
            errors.append(f'{clip}/{sid}: scegliere soltanto 2 identità umane per la sequenza')
            continue
        if not players:continue
        for player in players:
            present={int(r['frame']) for r in rr if r['player_id']==player}
            if not frames<=present:errors.append(f'{clip}/{sid}/{player}: sequenza incompleta')
    return errors
