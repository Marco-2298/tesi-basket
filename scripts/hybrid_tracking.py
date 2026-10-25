"""YOLO ripetuto e SAM2: nuovi giocatori confermati, reset esplicito del fork.

Il reset perde la memoria SAM2 ma conserva gli ID associati geometricamente.
Non è ReID: incroci e occlusioni prolungate possono cambiare l'identità.
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from yolo_onnx import nms_indices


def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=wh.prod();union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/max(union,1e-9))


def player_detections(model, frame, classes, confidence=.4):
    p=[x for x in model.infer(frame,confidence=confidence)[0].predictions if x.class_name in classes]
    boxes=np.array([[x.x-x.width/2,x.y-x.height/2,x.x+x.width/2,x.y+x.height/2] for x in p]).reshape(-1,4)
    scores=np.array([x.confidence for x in p])
    keep=nms_indices(boxes,scores,.7)
    return boxes[keep],scores[keep]


class Roster:
    def __init__(self, initial, confirmations=2, min_iou=.3, lost_frames=30):
        self.tracks={i+1:dict(box=np.array(b,float),seen=0) for i,b in enumerate(initial)}
        self.next_id=len(initial)+1;self.pending=[]
        self.confirmations=confirmations;self.min_iou=min_iou;self.lost_frames=lost_frames

    def observe_masks(self, rows, frame):
        for r in rows:
            tid=int(r[1])
            if tid in self.tracks and r[2]>0 and np.isfinite(np.array(r[3:7],float)).all():
                self.tracks[tid]=dict(box=np.array(r[3:7],float),seen=frame)

    def refresh(self, boxes, scores, frame):
        """Una sola chiamata per controllo detector; candidature consecutive."""
        ids=[i for i,t in self.tracks.items() if frame-t['seen']<=self.lost_frames]
        assignments={};matched=set()
        if len(boxes) and ids:
            cost=np.array([[1-box_iou(b,self.tracks[i]['box']) for i in ids] for b in boxes])
            # I match impossibili non devono rubare una colonna a un match valido.
            cost[cost>1-self.min_iou]=1e6
            rr,cc=linear_sum_assignment(cost)
            for r,c in zip(rr,cc):
                if cost[r,c]<1e6:
                    assignments[ids[c]]=np.array(boxes[r],float);matched.add(int(r))
        pending=[];used=set();births=[]
        for j,b in enumerate(boxes):
            if j in matched:continue
            if any(box_iou(b,other)>=.7 for other in assignments.values()):continue
            options=[(box_iou(b,p['box']),k) for k,p in enumerate(self.pending) if k not in used]
            score,k=max(options,default=(0,None))
            count=1
            if score>=self.min_iou:
                used.add(k);count=self.pending[k]['count']+1
            if count>=self.confirmations:
                tid=self.next_id;self.next_id+=1
                assignments[tid]=np.array(b,float);births.append(tid)
                self.tracks[tid]=dict(box=np.array(b,float),seen=frame)
            else:pending.append(dict(box=np.array(b,float),count=count))
        self.pending=pending
        # Quando nasce un oggetto ricreiamo il roster con bbox correnti, senza
        # riportare un oggetto invisibile da30frame come se fosse osservato.
        prompts={i:t['box'].copy() for i,t in self.tracks.items() if frame-t['seen']<=2}
        prompts.update(assignments)
        for tid,box in assignments.items():self.tracks[tid]=dict(box=box,seen=frame)
        return prompts,births


def seed(predictor,frame,prompts):
    """API del fork fissato: load_first_frame resetta condition_state."""
    predictor.frame_idx=0
    predictor.load_first_frame(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    ids=logits=None
    for tid,box in sorted(prompts.items()):
        _,ids,logits=predictor.add_new_prompt(frame_idx=0,obj_id=int(tid),bbox=np.asarray(box,np.float32).reshape(1,4))
    if ids is None:raise ValueError('Nessun prompt per SAM2')
    return ids,logits


def run(predictor,model,video,initial_boxes,draw_tracks,run_dir,work_dir,classes,
        interval=5,confirmations=2,confidence=.4):
    import torch
    if interval<1 or confirmations<1:raise ValueError('Intervalli non validi')
    run_dir=Path(run_dir);work_dir=Path(work_dir);video=Path(video)
    cap=cv2.VideoCapture(str(video));ok,frame=cap.read()
    if not ok:raise ValueError('Video vuoto')
    fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));h,w=frame.shape[:2]
    raw=work_dir/(video.stem+'_tracking_raw.mp4');csvpath=run_dir/(video.stem+'_tracks.csv')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():cap.release();raise RuntimeError('Codec non disponibile')
    roster=Roster(initial_boxes,confirmations=confirmations);events=[];audit=[];processed=0
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        with csvpath.open('w',newline='') as stream,torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
            output=csv.writer(stream);output.writerow(['frame','track_id','mask_area_px','x1','y1','x2','y2','foot_x','foot_y'])
            ids,logits=seed(predictor,frame,{i+1:b for i,b in enumerate(initial_boxes)})
            while True:
                if processed:
                    ids,logits=predictor.track(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                annotated,rows=draw_tracks(frame,ids,logits,processed)
                roster.observe_masks(rows,processed)
                if processed and processed%interval==0:
                    boxes,scores=player_detections(model,frame,classes,confidence)
                    prompts,births=roster.refresh(boxes,scores,processed)
                    for b,s in zip(boxes,scores):audit.append(dict(frame=processed,confidence=float(s),box=b.tolist()))
                    if births:
                        events.append(dict(frame=processed,new_ids=births,prompt_ids=sorted(prompts),action='reset_sam2_roster'))
                        ids,logits=seed(predictor,frame,prompts)
                        annotated,rows=draw_tracks(frame,ids,logits,processed)
                        roster.observe_masks(rows,processed)
                        print('Nuovi giocatori al frame',processed,':',births,flush=True)
                output.writerows(rows);writer.write(annotated);processed+=1
                if processed%25==0:print('Tracking ibrido:',processed,'/',total,flush=True)
                ok,frame=cap.read()
                if not ok:break
    finally:
        cap.release();writer.release()
        (run_dir/'roster_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        (run_dir/'periodic_detections.json').write_text(json.dumps(audit),encoding='utf-8')
    if processed!=total:raise RuntimeError('Decodifica incompleta')
    (run_dir/'hybrid_parameters.json').write_text(json.dumps(dict(interval_frames=interval,confirmations=confirmations,
        confidence=confidence,association_iou=.3,initial_ids=len(initial_boxes),total_ids=roster.next_id-1,
        limitations=['geometric association only; not ReID','SAM2 memory reset on roster change','new IDs not retrospectively tracked','parameters not validated']),indent=2),encoding='utf-8')
    return raw,csvpath,processed
