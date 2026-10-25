"""Piano ridotto basato sulle etichette presenti; non riscrive le annotazioni."""
import csv
import hashlib
import json
from pathlib import Path
from annotation_store import read_rows,task_complete
from project_config import ROOT


def main():
    if (ROOT/'reports/annotation_plan_reduction/summary.json').exists():
        raise RuntimeError('Piano già ridotto: non rigenerarlo, conserverai le coppie scelte nella app.')
    directory=ROOT/'annotations';path=directory/'annotations.csv'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=read_rows(path);manifest=read_rows(directory/'manifest.csv')
    tasks={}
    for r in rows:
        key=(r['clip_id'],int(r['frame']))
        tasks[key]=dict(clip_id=key[0],frame=key[1],point_target=1,sequence_id='',target_count=1,required_player_ids='',
                        instruction='Frame già annotato: lavoro conservato, nessun punto aggiuntivo richiesto')
    labeled_clips={r['clip_id'] for r in rows}
    for c in manifest:
        if c['clip_id'] in labeled_clips:continue
        n=int(c['frames'])
        for f in [round((n-1)*.2),round((n-1)*.8)]:
            tasks[(c['clip_id'],f)]=dict(clip_id=c['clip_id'],frame=f,point_target=1,sequence_id='',target_count=5,
                required_player_ids='',instruction='Annota SOLO 5 giocatori; puoi salvare non annotabile senza coordinate')
    for clip,start in [('clip_01',26),('clip_05',100),('clip_08',60)]:
        for f in range(start,start+8):
            tasks[(clip,f)]=dict(clip_id=clip,frame=f,point_target=int((clip,f) in tasks),sequence_id='seq1',target_count=2,
                required_player_ids='',instruction='Segui SOLO gli stessi 2 giocatori per 8 frame; scegli le identità nel pulsante dedicato')
    fields=['clip_id','frame','point_target','sequence_id','target_count','required_player_ids','instruction']
    ordered=[v for k,v in sorted(tasks.items())]
    with (directory/'tasks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(ordered)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    remaining_sparse=sum(max(0,int(t['target_count'])-len([r for r in rows if r['clip_id']==t['clip_id'] and int(r['frame'])==t['frame']]))
                         for t in ordered if not t['sequence_id'])
    summary=dict(previous_tasks=295,new_tasks=len(ordered),existing_labels=len(rows),existing_annotatable=sum(r['annotatable']=='1' for r in rows),
                 annotation_sha256_unchanged=before,remaining_sparse_labels=remaining_sparse,sequence_frames=24,players_per_sequence=2,
                 maximum_additional_sequence_labels=48,notes='Almeno due etichette del frame 26 sono riutilizzabili scegliendo ID già annotati; nessuna annotazione rimossa.')
    (ROOT/'reports/annotation_plan_reduction/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
