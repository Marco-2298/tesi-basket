"""Estensione sperimentale palla: YOLO11s locale, Kalman pixel e possesso candidato.

Non proietta la palla in aria con l'omografia del pavimento.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from project_config import ROOT, load_config
from thesis_tools import fingerprint, new_run
from yolo_onnx import YoloOnnxAdapter, nms_indices


class BallTracker:
    def __init__(self, gate_px=90., max_missing=4):
        self.gate = gate_px
        self.max_missing = max_missing
        self.k = None
        self.missing = 0
        self.segment = 0

    def step(self, candidates):
        """Un passo per frame; coordinate nello spazio immagine, dt=1 frame."""
        prediction = None
        if self.k is not None:
            prediction = self.k.predict()[:2, 0].copy()
        eligible = candidates
        if prediction is not None:
            eligible = [c for c in candidates if np.linalg.norm(np.array(c[:2])-prediction) <= self.gate]
        if eligible:
            if prediction is None:
                chosen = max(eligible, key=lambda c: c[2])
                self.k = cv2.KalmanFilter(4, 2)
                self.k.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
                self.k.measurementMatrix = np.eye(2,4,dtype=np.float32)
                self.k.processNoiseCov = np.eye(4,dtype=np.float32)*2
                self.k.measurementNoiseCov = np.eye(2,dtype=np.float32)*9
                self.k.errorCovPost = np.eye(4,dtype=np.float32)*100
                self.k.statePost = np.array([[chosen[0]],[chosen[1]],[0],[0]],np.float32)
                self.segment += 1
            else:
                chosen = min(eligible, key=lambda c: np.linalg.norm(np.array(c[:2])-prediction))
                self.k.correct(np.array(chosen[:2],np.float32).reshape(2,1))
            self.missing = 0
            return dict(state='observed', x_px=chosen[0], y_px=chosen[1], confidence=chosen[2], segment=self.segment)
        if self.k is None:
            return dict(state='missing',x_px=np.nan,y_px=np.nan,confidence=np.nan,segment=self.segment)
        self.missing += 1
        if self.missing > self.max_missing:
            self.k = None
            return dict(state='missing',x_px=np.nan,y_px=np.nan,confidence=np.nan,segment=self.segment)
        return dict(state='predicted',x_px=float(prediction[0]),y_px=float(prediction[1]),confidence=np.nan,segment=self.segment)


def owner_candidate(ball, players, max_distance=.35, ambiguity=.15):
    """Prossimità al bbox normalizzata per altezza: non è possesso verificato."""
    if ball['state'] != 'observed':
        return None
    ranked=[]
    for r in players.itertuples():
        if not np.isfinite([r.x1,r.y1,r.x2,r.y2]).all() or r.y2-r.y1 < 10:
            continue
        dx=max(r.x1-ball['x_px'],0,ball['x_px']-r.x2)
        dy=max(r.y1-ball['y_px'],0,ball['y_px']-r.y2)
        ranked.append((np.hypot(dx,dy)/(r.y2-r.y1),str(r.track_id)))
    ranked.sort()
    if not ranked or ranked[0][0]>max_distance:
        return None
    if len(ranked)>1 and ranked[1][0]-ranked[0][0]<ambiguity:
        return None
    return ranked[0][1]


def court_canvas():
    im=np.full((620,1040,3),(195,220,235),np.uint8)
    cv2.rectangle(im,(50,70),(990,570),(255,255,255),2)
    cv2.line(im,(520,70),(520,570),(255,255,255),2)
    cv2.circle(im,(520,320),60,(255,255,255),2)
    for x in [50,800]:
        cv2.rectangle(im,(x,240),(x+190,400),(255,255,255),2)
    return im


def run(clip_id, tracks_path=None, positions_path=None, confidence=.15, gate=90., max_missing=4, limit=None):
    config=load_config()
    manifest=pd.read_csv(config['manifest']).fillna('')
    match=manifest[manifest.clip_id.eq(clip_id)]
    if len(match)!=1: raise ValueError('Clip non univoca nel manifest')
    row=match.iloc[0]
    def resolve(value):
        p=Path(value)
        return p if p.is_absolute() else ROOT/p
    tracks=resolve(tracks_path or row.tracks_csv) if (tracks_path or row.tracks_csv) else None
    positions=resolve(positions_path or row.positions_csv) if (positions_path or row.positions_csv) else None
    if tracks is None or positions is None:
        raise ValueError('Servono tracking e posizioni giocatori: eseguire 01 oppure fornire --tracks e --positions.')
    players=pd.read_csv(tracks); coords=pd.read_csv(positions)
    if not {'frame','track_id','x1','y1','x2','y2'}<=set(players): raise ValueError('Schema tracking giocatori incompleto')
    if not {'frame','track_id','x_ft','y_ft','status'}<=set(coords): raise ValueError('Schema coordinate campo incompleto')
    if players.duplicated(['frame','track_id']).any() or coords.duplicated(['frame','track_id']).any(): raise ValueError('Frame/ID duplicati')
    players['track_id']=players.track_id.astype(str);coords['track_id']=coords.track_id.astype(str)
    video=resolve(row.video)
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(3));h=int(cap.get(4));total=int(cap.get(7))
    if not cap.isOpened() or fps<=0: raise ValueError('Video non leggibile')
    expected=set(range(min(total,limit) if limit else total))
    if not expected<=set(players.frame) or not expected<=set(coords.frame): raise ValueError('Output giocatori non coprono i frame richiesti')
    model=YoloOnnxAdapter(config['weights'])
    if not {'ball','ball-in-basket'}<=set(model.class_names): raise ValueError('Classi palla assenti')
    out=new_run(ROOT,'ball_tracking')
    writers=[cv2.VideoWriter(str(out/'ball_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)),
             cv2.VideoWriter(str(out/'tactical_board.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1040,620))]
    if not all(v.isOpened() for v in writers):
        cap.release()
        for v in writers:v.release()
        raise RuntimeError('Codec video non disponibile')
    tracker=BallTracker(gate,max_missing); history=[];detections=[];trail=[]
    previous=None;streak=0
    try:
        for frame in sorted(expected):
            ok,im=cap.read()
            if not ok:raise RuntimeError(f'Decodifica interrotta al frame {frame}')
            preds=[p for p in model.infer(im,confidence=confidence)[0].predictions if p.class_name in {'ball','ball-in-basket'}]
            boxes=np.array([[p.x-p.width/2,p.y-p.height/2,p.x+p.width/2,p.y+p.height/2] for p in preds]).reshape(-1,4)
            kept=nms_indices(boxes,np.array([p.confidence for p in preds]),.5)
            candidates=[]
            for i in kept:
                p=preds[i];candidates.append((p.x,p.y,p.confidence))
                detections.append(dict(frame=frame,x_px=p.x,y_px=p.y,confidence=p.confidence,class_name=p.class_name,width=p.width,height=p.height))
            ball=tracker.step(candidates)
            pp=players[players.frame.eq(frame)]; cc=coords[coords.frame.eq(frame)]
            candidate=owner_candidate(ball,pp)
            streak=streak+1 if candidate is not None and candidate==previous else (1 if candidate else 0)
            previous=candidate
            owner=candidate if streak>=3 else None
            history.append(dict(frame=frame,**ball,owner_candidate=owner or '',owner_state='candidate' if owner else 'unknown'))
            if ball['state']!='missing':
                point=(round(ball['x_px']),round(ball['y_px']))
                if not (0<=point[0]<w and 0<=point[1]<h):trail=[]
                else:
                    if trail and trail[-1][0]!=ball['segment']:trail=[]
                    trail.append((ball['segment'],point));trail=trail[-8:]
                    color=(0,165,255) if ball['state']=='observed' else (0,255,255)
                    cv2.circle(im,point,8,color,2)
                    for a,b in zip(trail,trail[1:]):cv2.line(im,a[1],b[1],color,1)
            else:trail=[]
            cv2.putText(im,f"Ball: {ball['state']} | owner candidate: {owner or '?'}",(10,25),0,.55,(0,165,255),2)
            board=court_canvas()
            for r in cc.itertuples():
                if r.status!='projected' or not np.isfinite([r.x_ft,r.y_ft]).all():continue
                if not (0<=r.x_ft<=94 and 0<=r.y_ft<=50):continue
                pt=(round(50+r.x_ft*10),round(70+r.y_ft*10))
                cv2.circle(board,pt,10,(65,65,65),-1);cv2.putText(board,str(r.track_id),(pt[0]+12,pt[1]),0,.5,(0,0,0),1)
                if owner==str(r.track_id):cv2.circle(board,(pt[0]-14,pt[1]-14),7,(0,140,255),-1)
            cv2.putText(board,f"Frame {frame} | ball: {ball['state']} | owner candidate: {owner or '?'}",(20,25),0,.6,(0,0,0),1)
            cv2.putText(board,'Orange ball = candidate owner; free-flight ground position unavailable',(20,600),0,.55,(0,0,0),1)
            writers[0].write(im);writers[1].write(board)
            if frame%25==0:print(f'{clip_id}: {frame+1}/{len(expected)} frame',flush=True)
    finally:
        cap.release()
        for v in writers:v.release()
        pd.DataFrame(history).to_csv(out/'ball_tracks.csv',index=False)
        pd.DataFrame(detections,columns=['frame','x_px','y_px','confidence','class_name','width','height']).to_csv(out/'ball_detections.csv',index=False)
    summary=dict(clip_id=clip_id,frames=len(history),state_counts=pd.Series([r['state'] for r in history]).value_counts().to_dict(),
                 confidence=confidence,gate_px=gate,max_missing=max_missing,owner_stable_frames=3,
                 video_sha256=fingerprint(video),weights_sha256=fingerprint(config['weights']),
                 tracks_sha256=fingerprint(tracks),positions_sha256=fingerprint(positions),
                 parameters_status='experimental defaults; not selected or evaluated against ball annotations',
                 limitations=['owner is heuristic, not verified possession','no ball 3D or airborne ground projection','camera motion and cuts may break pixel Kalman','no automatic pass events'],
                 complete=len(history)==total)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # H.264 per la riproduzione nei browser Colab, se ffmpeg è disponibile.
    # Gli originali mp4v restano disponibili se la conversione non riesce.
    if shutil.which('ffmpeg'):
        for name in ['ball_tracking','tactical_board']:
            target=out/(name+'_browser.mp4')
            result=subprocess.run(['ffmpeg','-v','error','-n','-i',str(out/(name+'.mp4')),
                                   '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],
                                  capture_output=True,text=True)
            if result.returncode:
                target.unlink(missing_ok=True)
                print('Conversione browser non disponibile:',result.stderr[-300:],flush=True)
    print('Output:',out,flush=True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--clip',default='clip_01');ap.add_argument('--tracks');ap.add_argument('--positions')
    ap.add_argument('--confidence',type=float,default=.15);ap.add_argument('--gate',type=float,default=90.)
    ap.add_argument('--max-missing',type=int,default=4);ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    if args.gate<=0 or args.max_missing<0 or (args.limit is not None and args.limit<=0):ap.error('Parametri temporali non validi')
    run(args.clip,args.tracks,args.positions,args.confidence,args.gate,args.max_missing,args.limit)
