"""Eventi candidati da palla e tracking già esportati; nessuna nuova inferenza."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ball_tracking import owner_candidate, court_canvas
from project_config import ROOT
from thesis_tools import new_run, fingerprint


class PossessionMachine:
    def __init__(self, teams=None, stable=3, hold=6, timeout=30):
        self.teams=teams or {};self.stable=stable;self.hold=hold;self.timeout=timeout
        self.owner=None;self.last_seen=None;self.segment=None
        self.pending=None;self.streak=0;self.release=None;self.events=[]
        self.last_ball_observed=None

    def reset(self):
        self.owner=None;self.last_seen=None;self.pending=None
        self.streak=0;self.release=None

    def step(self, frame, candidate, observed, segment, free=False, basket=False):
        if self.segment is not None and segment!=self.segment:
            self.reset()
        self.segment=segment
        if basket or (self.last_seen is not None and frame-self.last_seen>self.timeout):
            self.reset()
        # Un vero buco di rilevamento lungo invalida l'evento, anche se l'ID palla è rimasto.
        if not observed:
            self.pending=None;self.streak=0
            if self.last_ball_observed is not None and frame-self.last_ball_observed>self.hold:
                self.reset()
        elif basket:
            candidate=None
        elif candidate is not None:
            candidate=str(candidate)
            if candidate==self.owner:
                self.last_seen=frame;self.release=None;self.pending=None;self.streak=0
            else:
                self.streak=self.streak+1 if self.pending==candidate else 1
                self.pending=candidate
                if self.streak>=self.stable:
                    if self.owner is not None:
                        a=self.teams.get(self.owner);b=self.teams.get(candidate)
                        transit=self.release is not None
                        if a and b and transit:
                            kind='pass_candidate' if a==b else 'interception_candidate'
                        else:
                            kind='uncertain_change'
                        self.events.append(dict(event_id=len(self.events)+1,event_frame=frame,
                            release_frame=self.release if transit else self.last_seen,
                            reception_start_frame=frame-self.streak+1,from_id=self.owner,to_id=candidate,
                            from_team=a or '',to_team=b or '',kind=kind,transit_observed=transit,
                            review_status='unreviewed',note='Interception may also be rebound/loose ball/steal; review video' if kind=='interception_candidate' else 'Heuristic event; review video'))
                    self.owner=candidate;self.last_seen=frame;self.pending=None;self.streak=0;self.release=None
        else:
            self.pending=None;self.streak=0
            if free and self.owner is not None and self.release is None:
                self.release=frame
        if observed:
            self.last_ball_observed=frame
        if self.owner is None:
            return dict(owner_id='',possession_state='unknown')
        age=frame-self.last_seen
        if self.release is not None:
            return dict(owner_id='',possession_state='in_transit')
        if age==0:
            return dict(owner_id=self.owner,possession_state='stable_candidate')
        if age<=self.hold:
            return dict(owner_id=self.owner,possession_state='held_uncertain')
        return dict(owner_id='',possession_state='unknown')


def load_teams(path, clip):
    if not Path(path).is_file():return {}
    d=pd.read_csv(path,dtype=str).fillna('')
    if not {'clip_id','track_id','team','verified'}<=set(d):raise ValueError('Schema squadre incompleto')
    d=d[d.clip_id.eq(clip)]
    if d.track_id.duplicated().any():raise ValueError('ID squadra duplicati nella clip')
    valid=d.verified.str.lower().isin(['true','1','yes']) & d.team.str.strip().ne('')
    return dict(zip(d.loc[valid,'track_id'],d.loc[valid,'team'].str.strip()))


def free_ball(ball, players):
    if ball['state']!='observed':return False
    for r in players.itertuples():
        if not np.isfinite([r.x1,r.y1,r.x2,r.y2]).all():continue
        pad=max(0,r.y2-r.y1)*.1
        if r.x1-pad<=ball['x_px']<=r.x2+pad and r.y1-pad<=ball['y_px']<=r.y2+pad:
            return False
    return True


def run(clip, ball_dir, tracks, positions, teams='annotations/player_teams.csv', stable=3, hold=6, timeout=30):
    resolve=lambda p:Path(p) if Path(p).is_absolute() else ROOT/Path(p)
    ball_dir=resolve(ball_dir);tracks=resolve(tracks);positions=resolve(positions);teams=resolve(teams)
    ball=pd.read_csv(ball_dir/'ball_tracks.csv');players=pd.read_csv(tracks);coords=pd.read_csv(positions)
    summary=json.loads((ball_dir/'summary.json').read_text())
    if summary['clip_id']!=clip or summary['tracks_sha256']!=fingerprint(tracks) or summary['positions_sha256']!=fingerprint(positions):
        raise ValueError('Clip o CSV diversi dagli input del tracking palla')
    if ball.frame.duplicated().any() or list(ball.frame)!=list(range(len(ball))):raise ValueError('Frame palla non consecutivi')
    players['track_id']=players.track_id.astype(str);coords['track_id']=coords.track_id.astype(str)
    manifest=pd.read_csv(ROOT/'annotations/manifest.csv').fillna('');row=manifest[manifest.clip_id.eq(clip)].iloc[0]
    video=resolve(row.video)
    if fingerprint(video)!=summary['video_sha256']:raise ValueError('Video diverso dagli input della palla')
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(3));h=int(cap.get(4))
    out=new_run(ROOT,'possession_events')
    writers=[cv2.VideoWriter(str(out/'possession_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)),
             cv2.VideoWriter(str(out/'tactical_events.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1040,620))]
    if not cap.isOpened() or not all(v.isOpened() for v in writers):raise RuntimeError('Video/codec non disponibile')
    mapping=load_teams(teams,clip);machine=PossessionMachine(mapping,stable,hold,timeout);states=[]
    det_path=ball_dir/'ball_detections.csv';dets=pd.read_csv(det_path) if det_path.exists() else pd.DataFrame()
    def location(frame,tid):
        d=coords[(coords.frame.eq(frame)) & coords.track_id.eq(str(tid))]
        if len(d)!=1:return None
        r=d.iloc[0]
        if r.status!='projected' or not np.isfinite([r.x_ft,r.y_ft]).all() or not (0<=r.x_ft<=94 and 0<=r.y_ft<=50):return None
        return (round(50+r.x_ft*10),round(70+r.y_ft*10))
    try:
        for r in ball.itertuples(index=False):
            frame=int(r.frame);ok,im=cap.read()
            if not ok:raise RuntimeError('Video interrotto')
            b=r._asdict();pp=players[players.frame.eq(frame)]
            candidate=owner_candidate(b,pp)
            basket=False
            if b['state']=='observed' and not dets.empty:
                dd=dets[(dets.frame.eq(frame)) & dets.class_name.eq('ball-in-basket')]
                basket=any(np.hypot(dd.x_px-b['x_px'],dd.y_px-b['y_px'])<1)
            state=machine.step(frame,candidate,b['state']=='observed',int(b['segment']),free_ball(b,pp),basket)
            states.append(dict(frame=frame,ball_state=b['state'],raw_owner_candidate=candidate or '',**state))
            counts={k:sum(e['kind']==k for e in machine.events) for k in ['pass_candidate','interception_candidate','uncertain_change']}
            board=court_canvas()
            for player in pp.itertuples():
                pt=location(frame,player.track_id)
                if pt is not None:
                    cv2.circle(board,pt,10,(65,65,65),-1);cv2.putText(board,str(player.track_id),(pt[0]+12,pt[1]),0,.5,(0,0,0),1)
                    if str(player.track_id)==state['owner_id']:
                        thickness=2 if state['possession_state']=='held_uncertain' else -1
                        cv2.circle(board,(pt[0]-14,pt[1]-14),8,(0,140,255),thickness)
                if str(player.track_id)==state['owner_id']:
                    cv2.rectangle(im,(int(player.x1),int(player.y1)),(int(player.x2),int(player.y2)),(0,165,255),2)
            for e in machine.events:
                if frame-e['event_frame']>round(2*fps):continue
                a=location(e['release_frame'],e['from_id']);z=location(e['reception_start_frame'],e['to_id'])
                if a is not None and z is not None:
                    color=(0,160,0) if e['kind']=='pass_candidate' else ((0,0,220) if e['kind']=='interception_candidate' else (100,100,100))
                    cv2.arrowedLine(board,a,z,color,3,tipLength=.15)
                    cv2.putText(board,f"{e['event_id']}?",a,0,.6,color,2)
            if b['state']!='missing' and np.isfinite([b['x_px'],b['y_px']]).all():
                cv2.circle(im,(round(b['x_px']),round(b['y_px'])),7,(0,165,255),2)
            title=f"Frame {frame} | {state['possession_state']} | owner {state['owner_id'] or '?'}"
            count_text=f"CANDIDATES passes {counts['pass_candidate']} | interceptions {counts['interception_candidate']} | uncertain {counts['uncertain_change']}"
            cv2.putText(im,title,(10,25),0,.5,(0,165,255),2)
            cv2.putText(board,title,(20,25),0,.55,(0,0,0),1)
            cv2.putText(board,count_text,(20,50),0,.5,(0,0,0),1)
            cv2.putText(board,'Heuristic events: video review required. Hollow ball = uncertain memory.',(20,600),0,.5,(0,0,0),1)
            writers[0].write(im);writers[1].write(board)
    finally:
        cap.release()
        for v in writers:v.release()
    pd.DataFrame(states).to_csv(out/'possession_states.csv',index=False)
    columns=['event_id','event_frame','release_frame','reception_start_frame','from_id','to_id','from_team','to_team','kind','transit_observed','review_status','note']
    pd.DataFrame(machine.events,columns=columns).to_csv(out/'events_to_review.csv',index=False)
    result=dict(clip_id=clip,frames=len(states),verified_team_ids=len(mapping),events=len(machine.events),
        candidate_counts={k:sum(e['kind']==k for e in machine.events) for k in ['pass_candidate','interception_candidate','uncertain_change']},
        stable_frames=stable,hold_frames=hold,event_timeout_frames=timeout,ball_tracks_sha256=fingerprint(ball_dir/'ball_tracks.csv'),
        teams_sha256=fingerprint(teams) if teams.exists() else None,source_sha256=fingerprint(Path(__file__)),
        note='Candidate counts only; interception alternatives include rebounds and steals. Long gaps/segment changes break event linkage.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));print('Output:',out)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--clip',default='clip_01');ap.add_argument('--ball-dir',required=True)
    ap.add_argument('--tracks',required=True);ap.add_argument('--positions',required=True)
    ap.add_argument('--teams',default='annotations/player_teams.csv');ap.add_argument('--stable',type=int,default=3)
    ap.add_argument('--hold',type=int,default=6);ap.add_argument('--timeout',type=int,default=30)
    a=ap.parse_args()
    if a.stable<1 or a.hold<0 or a.timeout<1:ap.error('Durate non valide')
    run(a.clip,a.ball_dir,a.tracks,a.positions,a.teams,a.stable,a.hold,a.timeout)
