"""Verifica video reali, metadati YOLO11 e prepara input manuali senza sovrascriverli."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_new_csv(path, fields, rows):
    if path.exists():
        print('Conservato:', path)
        return
    with path.open('x', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    import cv2
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report=ROOT/'reports'/('workspace_check_'+stamp)
    report.mkdir(parents=True,exist_ok=False)
    (ROOT/'configs').mkdir(exist_ok=True)
    annotations=ROOT/'annotations'; annotations.mkdir(exist_ok=True)
    models=list((ROOT/'models').rglob('model_type.json'))
    candidates=[]
    for p in models:
        m=json.loads(p.read_text())
        if m.get('model_type')=='yolov11': candidates.append((p,m))
    if len(candidates)!=1: raise ValueError(f'Atteso un modello YOLO11 identificabile; trovati {len(candidates)}')
    type_path,model_type=candidates[0]
    weights=list((ROOT/'models').rglob('weights.onnx'))
    if len(weights)!=1: raise ValueError('Selezionare esplicitamente il modello ONNX')
    weights=weights[0]
    backup_manifest=next((ROOT/'models').glob('*/manifest.json'))
    backup=backup_manifest.parent
    original=json.loads(backup_manifest.read_text())
    verified=[]
    for entry in original['files']:
        p=backup/entry['path']
        ok=p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
        verified.append(dict(path=entry['path'],valid=ok))
    if not all(e['valid'] for e in verified): raise ValueError('Backup modello alterato: vedere checksum originali')
    config=dict(schema_version=1, clips_dir='videos', detection_test_dir='basketball_player_detection_v18/test',
                weights=weights.relative_to(ROOT).as_posix(), model_variant=model_type['model_variant'],
                model_id=model_type['model_id'],manifest='annotations/manifest.csv',annotations='annotations/annotations.csv',
                tasks='annotations/tasks.csv',confidence=0.4,nms_iou=0.7,max_gap_candidates=[0,2,5,10],match_gate_px=80.,
                player_classes=['player','player-in-possession','player-jump-shot','player-layup-dunk','player-shot-block'])
    config_path=ROOT/'configs/project.json'
    if not config_path.exists(): config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    historical=list(csv.DictReader((ROOT/'outputs/tracking_evaluation_v1/test_set_manifest.csv').open(encoding='utf-8-sig')))
    clips=sorted(p for p in (ROOT/'videos').iterdir() if p.suffix.lower() in {'.mov','.mp4','.mkv','.avi'})
    if len(clips)!=7 or len({p.stem for p in clips})!=7: raise ValueError('Servono sette clip con nomi univoci')
    meta=[]; tasks=[]; targets=[]; scene=[]
    for j,p in enumerate(clips):
        cap=cv2.VideoCapture(str(p))
        if not cap.isOpened(): raise RuntimeError(f'Video non apribile: {p}')
        fps=cap.get(cv2.CAP_PROP_FPS); declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count=0
        while True:
            ok,image=cap.read()
            if not ok: break
            if image.shape[:2]!=(height,width): raise ValueError(f'Risoluzione variabile: {p}')
            count+=1
        cap.release()
        if not count or fps<=0: raise ValueError(f'Video vuoto o FPS invalidi: {p}')
        old=next((r for r in historical if r['file_name']==p.name),None)
        record=dict(clip_id=p.stem,video=p.relative_to(ROOT).as_posix(),fps=fps,frames=count,width=width,height=height,
                    duration_s=count/fps,split='',shot_type='',shot_verified='',tracks_csv='',positions_csv='',homographies_json='')
        meta.append(record)
        targets.append(dict(clip_id=p.stem,sha256=sha(p),declared_frames=declared,decoded_frames=count,
                            metadata_consistent=count==declared,historical_hash_match=bool(old and old['sha256']==sha(p))))
        point_frames={round(k*(count-1)/((15 if j<2 else 14)-1)) for k in range(15 if j<2 else 14)}
        seq_frames={}
        for k,start in enumerate([count//4,count//2],1):
            for f in range(start,min(start+15,count)): seq_frames[f]=f'seq{k}'
        for f in sorted(point_frames|set(seq_frames)):
            tasks.append(dict(clip_id=p.stem,frame=f,point_target=int(f in point_frames),sequence_id=seq_frames.get(f,''),
                              instruction='Un giocatore per punto; nelle sequenze scegli 2-3 identita umane stabili e seguile in tutti i frame'))
        scene.append(dict(clip_id=p.stem,start_frame=0,end_frame=count-1,shot_type='',camera_motion='',verified='',notes=''))
    write_new_csv(annotations/'manifest.csv',list(meta[0]),meta)
    write_new_csv(annotations/'tasks.csv',list(tasks[0]),tasks)
    write_new_csv(annotations/'scene_intervals.csv',list(scene[0]),scene)
    write_new_csv(annotations/'annotations.csv',['clip_id','frame','player_id','visibility','annotatable','x_px','y_px','sequence_id','note'],[])
    write_new_csv(report/'decode_verification.csv',list(targets[0]),targets)
    evidence=dict(model_type_metadata=str(type_path.relative_to(ROOT)),model_type=model_type,weights=str(weights.relative_to(ROOT)),
                  weights_sha256=sha(weights),backup_files_verified=verified,all_backup_hashes_verified=True,
                  runtime_restore_verified=False,point_targets=sum(t['point_target'] for t in tasks),tasks=len(tasks))
    (report/'model_and_preparation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (ROOT/'configs/model_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report),variant=model_type['model_variant'],clips=len(meta),tasks=len(tasks),
                          point_targets=evidence['point_targets']),indent=2))


if __name__=='__main__': main()
