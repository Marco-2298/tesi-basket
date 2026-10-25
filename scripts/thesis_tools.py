"""Valutazione offline: nessuna inferenza, nessuna cancellazione di punti."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def new_run(root, name):
    import uuid
    p = Path(root) / 'outputs' / name / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    p.mkdir(parents=True, exist_ok=False)
    return p


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_tracks(path):
    d = pd.read_csv(path)
    required = {'frame', 'track_id', 'foot_x', 'foot_y'}
    if not required <= set(d):
        raise ValueError(f'{path}: colonne richieste {sorted(required)}')
    if d.duplicated(['frame', 'track_id']).any():
        raise ValueError(f'{path}: frame/ID duplicati')
    if d.frame.isna().any() or (d.frame < 0).any() or (d.frame % 1 != 0).any() or d.track_id.isna().any():
        raise ValueError('Frame non interi/negativi o ID mancanti')
    d['track_id'] = d.track_id.astype(str)
    d[['foot_x', 'foot_y']] = d[['foot_x', 'foot_y']].apply(pd.to_numeric, errors='raise')
    if 'mask_area_px' in d:
        d.loc[d.mask_area_px.eq(0), ['foot_x', 'foot_y']] = np.nan
    d[['foot_x', 'foot_y']] = d[['foot_x', 'foot_y']].replace([np.inf, -np.inf], np.nan)
    return d.sort_values(['track_id', 'frame']).reset_index(drop=True)


def correct(d, max_gap):
    """Nuovo baseline di interpolazione; non sostituisce un correttore non fornito.

    Interpola solo buchi interni con entrambi gli estremi finiti dello stesso ID.
    MAX_GAP conta i frame mancanti; non cambia i punti osservati o gli ID.
    """
    if not isinstance(max_gap, int) or max_gap < 0:
        raise ValueError('MAX_GAP deve essere intero >= 0')
    parts = []
    for tid, g in d.groupby('track_id'):
        g = g.set_index('frame').reindex(range(int(g.frame.min()), int(g.frame.max()) + 1))
        g['track_id'] = tid
        g['interpolated'] = False
        valid = np.flatnonzero(np.isfinite(g[['foot_x', 'foot_y']]).all(axis=1))
        for a, b in zip(valid[:-1], valid[1:]):
            if 0 < b - a - 1 <= max_gap:
                for c in ['foot_x', 'foot_y']:
                    g.iloc[a:b+1, g.columns.get_loc(c)] = np.linspace(g.iloc[a][c], g.iloc[b][c], b-a+1)
                g.iloc[a+1:b, g.columns.get_loc('interpolated')] = True
        parts.append(g.rename_axis('frame').reset_index())
    return pd.concat(parts, ignore_index=True) if parts else d.assign(interpolated=False)


def match_annotations(gt, pred, gate):
    """Associazione spaziale 1:1 per frame, indipendente dagli ID predetti."""
    rows = []
    for frame, g in gt.groupby('frame'):
        p = pred[pred.frame.eq(frame)].dropna(subset=['foot_x', 'foot_y'])
        g = g.reset_index(drop=True)
        p = p.reset_index(drop=True)
        matched = {}
        if len(g) and len(p):
            dist = np.linalg.norm(g[['x_px', 'y_px']].to_numpy()[:, None] - p[['foot_x', 'foot_y']].to_numpy()[None], axis=2)
            # Penalità grande: massimizza prima il numero di associazioni entro gate.
            cost = np.where(dist <= gate, dist, (len(g)+len(p)+1)*(gate+1))
            aa, bb = linear_sum_assignment(cost)
            matched = {int(a): (str(p.iloc[b].track_id), float(dist[a,b])) for a,b in zip(aa,bb) if dist[a,b] <= gate}
        for i, r in g.iterrows():
            tid, err = matched.get(i, (None, np.nan))
            rows.append({**r.to_dict(), 'pred_id': tid, 'error_px': err})
    return pd.DataFrame(rows)


def metrics(matches):
    err = matches.error_px.dropna()
    switches = fragments = 0
    seq = matches[matches.sequence_id.fillna('').ne('')]
    for _, g in seq.groupby(['sequence_id', 'player_id']):
        previous = None
        had_match = False
        interrupted = False
        last_frame = None
        for r in g.sort_values('frame').itertuples():
            if last_frame is not None and r.frame != last_frame + 1:
                previous, had_match, interrupted = None, False, False
            last_frame = r.frame
            if pd.isna(r.pred_id):
                interrupted = had_match
                continue
            if previous is not None and r.pred_id != previous:
                switches += 1
            if interrupted:
                fragments += 1
            previous, had_match, interrupted = r.pred_id, True, False
    return dict(n_annotatable=len(matches), n_matched=len(err), coverage=len(err)/len(matches) if len(matches) else None,
                mean_error_px=float(err.mean()) if len(err) else None,
                rmse_px=float(np.sqrt(np.mean(err**2))) if len(err) else None,
                id_switches=switches if len(seq) else None, fragmentations=fragments if len(seq) else None)


def projection_diagnostics(tracks, homographies, fps, speed_limit_m_s=12.):
    """H immagine->campo in feet, per frame; nessun riuso su frame mancanti."""
    rows=[]
    for r in tracks.itertuples():
        h=homographies.get(str(int(r.frame)))
        x=y=np.nan
        status='missing_geometry'
        if h is not None:
            h=np.asarray(h,dtype=float)
            if h.shape != (3,3): raise ValueError('Omografia attesa 3x3')
            q=h @ np.array([r.foot_x,r.foot_y,1.])
            status='invalid_projection'
            if np.isfinite(q).all() and abs(q[2])>1e-10:
                x,y=q[:2]/q[2]*0.3048
                status='inside' if 0<=x<=94*.3048 and 0<=y<=50*.3048 else 'outside'
        rows.append(dict(frame=r.frame,track_id=r.track_id,x_m=x,y_m=y,status=status))
    d=pd.DataFrame(rows).sort_values(['track_id','frame'])
    delta=d.groupby('track_id')[['frame','x_m','y_m']].diff()
    d['speed_m_s']=np.hypot(delta.x_m,delta.y_m)/(delta.frame/fps)
    d['implausible_speed']=d.speed_m_s.gt(speed_limit_m_s)
    return d


def evaluate(root, manifest_path, annotations_path, gaps=(0, 2, 5, 10), gate=80., predictor=None, method_label='bounded_pixel_interpolation', baseline_predictor=None):
    root = Path(root)
    manifest = pd.read_csv(manifest_path)
    gt = pd.read_csv(annotations_path, dtype={'player_id': str})
    # Piano ridotto: le altre etichette rimangono nella valutazione spaziale,
    # ma la continuità usa soltanto le due identità selezionate per sequenza.
    tasks_path=root/'annotations'/'tasks.csv'
    if tasks_path.is_file():
        plan=pd.read_csv(tasks_path).fillna('')
        if 'required_player_ids' in plan:
            gt['sequence_id']=''
            for t in plan.itertuples():
                if not t.sequence_id:continue
                ids=[x.strip() for x in t.required_player_ids.split(';') if x.strip()]
                selected=gt.clip_id.eq(t.clip_id)&gt.frame.eq(t.frame)&gt.player_id.isin(ids)
                gt.loc[selected,'sequence_id']=t.sequence_id
    if not np.isfinite(gate) or gate<=0 or 0 not in gaps or any(not isinstance(g,int) or g<0 for g in gaps):
        raise ValueError('Gate positiva e candidati interi >=0 inclusa baseline gap=0 richiesti')
    required = {'clip_id','frame','player_id','visibility','annotatable','x_px','y_px','sequence_id'}
    if not required <= set(gt):
        raise ValueError(f'Annotazioni: servono {sorted(required)}')
    if gt.duplicated(['clip_id','frame','player_id']).any():
        raise ValueError('Annotazioni duplicate; punti e sequenze devono condividere una sola riga')
    if gt.clip_id.isna().any() or gt.player_id.isna().any() or not set(gt.clip_id) <= set(manifest.clip_id):
        raise ValueError('Clip o identità annotazioni non valide')
    gt = gt[gt.annotatable.eq(1)].copy()
    if gt.empty or not np.isfinite(gt[['frame','x_px','y_px']].to_numpy(dtype=float)).all():
        raise ValueError('Compilare annotazioni annotabili e finite prima di valutare')
    if (gt.frame < 0).any() or (gt.frame % 1 != 0).any():
        raise ValueError('Frame annotazioni: interi >= 0')
    if manifest.clip_id.duplicated().any() or not set(manifest.split) <= {'validation','test'}:
        raise ValueError('Una sola riga e uno split validation/test per clip')
    if not {'validation','test'} <= set(manifest.split):
        raise ValueError('Servono clip distinte di validazione e test')
    all_rows, all_matches = [], []
    for r in manifest.itertuples():
        a = gt[gt.clip_id.eq(r.clip_id)]
        if a.empty:
            raise ValueError(f'Annotazioni mancanti per {r.clip_id}')
        raw = load_tracks(root / r.tracks_csv)
        raw.attrs['clip_id']=r.clip_id
        for gap in gaps:
            if gap==0 and baseline_predictor:
                pred=baseline_predictor(raw)
            else:
                pred = (raw.copy() if gap==0 else predictor(raw,int(gap))) if predictor else correct(raw,int(gap))
            m = match_annotations(a, pred, gate)
            m['max_gap'], m['split'] = gap, r.split
            all_matches.append(m)
            all_rows.append(dict(clip_id=r.clip_id, split=r.split, max_gap=gap, **metrics(m)))
    per_clip = pd.DataFrame(all_rows)
    matches = pd.concat(all_matches, ignore_index=True)
    # Strati manuali: nessuna categoria dedotta dal modello è trattata come GT.
    matches['shot_type']='unverified'
    matches['camera_motion']='unverified'
    intervals_path=root/'annotations'/'scene_intervals.csv'
    if intervals_path.is_file():
        intervals=pd.read_csv(intervals_path).fillna('')
        assigned=set()
        for s in intervals.itertuples():
            if str(s.verified).lower() not in {'true','1','yes'}:continue
            if s.shot_type not in {'half_court','wide','pan_between_halves'}:
                raise ValueError('Categoria di inquadratura verificata non valida')
            if s.camera_motion not in {'static','pan','zoom','mixed'}:
                raise ValueError('Movimento camera verificato non valido')
            if s.start_frame<0 or s.end_frame<s.start_frame:raise ValueError('Intervallo scena non valido')
            for f in range(int(s.start_frame),int(s.end_frame)+1):
                key=(s.clip_id,f)
                if key in assigned:raise ValueError('Intervalli scena verificati sovrapposti')
                assigned.add(key)
            selected=matches.clip_id.eq(s.clip_id)&matches.frame.between(s.start_frame,s.end_frame)
            matches.loc[selected,['shot_type','camera_motion']]=[s.shot_type,s.camera_motion]
    # RMSE sui punti abbinati e copertura esportati insieme: la gate censura gli errori.
    selection = per_clip[per_clip.split.eq('validation')].groupby('max_gap').agg(coverage=('coverage','mean'), rmse_px=('rmse_px','mean'))
    if selection.rmse_px.isna().all():
        raise ValueError('Nessun match in validazione: verificare coordinate/gate')
    chosen = int(selection.sort_values(['coverage','rmse_px'], ascending=[False,True]).index[0])
    out = new_run(root, 'evaluation')
    per_clip.to_csv(out/'per_clip.csv', index=False)
    matches.to_csv(out/'matched_points.csv', index=False)
    strata=[]
    for field in ['visibility','shot_type','camera_motion']:
        for (clip,split,gap,label),m in matches.groupby(['clip_id','split','max_gap',field]):
            value=metrics(m)
            value['id_switches']=None;value['fragmentations']=None
            strata.append(dict(clip_id=clip,split=split,max_gap=gap,stratum=field,label=label,**value))
    pd.DataFrame(strata).to_csv(out/'strata_per_clip.csv',index=False)
    selection.to_csv(out/'validation_selection.csv')
    aggregates = []
    for (split,gap), m in matches.groupby(['split','max_gap']):
        # Identità di sequenza distinte tra clip.
        m = m.copy()
        m['sequence_id'] = np.where(m.sequence_id.fillna('').ne(''), m.clip_id.astype(str)+':'+m.sequence_id.fillna(''), '')
        aggregates.append(dict(split=split,max_gap=gap,**metrics(m)))
    pd.DataFrame(aggregates).to_csv(out/'aggregate.csv',index=False)
    per_clip[(per_clip.split.eq('test')) & per_clip.max_gap.isin([0,chosen])].to_csv(out/'final_test.csv',index=False)
    extra_inputs={}
    for r in manifest.itertuples():
        for name in ['positions_csv','homographies_json']:
            p=getattr(r,name,None)
            if isinstance(p,str) and p.strip() and (root/p).is_file():extra_inputs[f'{r.clip_id}:{name}']=fingerprint(root/p)
    if intervals_path.is_file():extra_inputs['scene_intervals']=fingerprint(intervals_path)
    if tasks_path.is_file():extra_inputs['annotation_tasks']=fingerprint(tasks_path)
    (out/'provenance.json').write_text(json.dumps(dict(chosen_max_gap=chosen, gate_px=gate,
        selection='macro coverage descending, macro RMSE ascending; validation only',
        manifest_sha256=fingerprint(manifest_path), annotations_sha256=fingerprint(annotations_path),
        tracks={r.clip_id:fingerprint(root/r.tracks_csv) for r in manifest.itertuples()},
        caveat='Errori pixel condizionati al matching entro gate; non accuratezza metrica della calibrazione.',
        corrector=method_label,software_sha256=fingerprint(__file__),extra_inputs_sha256=extra_inputs,
        height_corrector_sha256=fingerprint(Path(__file__).with_name('height_corrector.py'))),indent=2),encoding='utf-8')
    return out, per_clip, matches
