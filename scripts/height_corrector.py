"""Regola originale recuperata dal Notebook 03, celle 19,21,38 (1-based).

Il riempimento è offline: interpolazione e backfill usano anche dati futuri.
Le sorgenti delle coordinate sono esportate esplicitamente per ogni punto.
"""
import numpy as np
import pandas as pd


def flag_height(tracks, ratio_min=.82,recovery_ratio=.90,history_size=15,min_history=5):
    required={'frame','track_id','y1','y2'}
    if not required<=set(tracks): raise ValueError(f'Mancano colonne: {required-set(tracks)}')
    if not 0<ratio_min<=recovery_ratio or not 1<=min_history<=history_size:
        raise ValueError('Parametri altezza non validi')
    q=tracks.sort_values(['track_id','frame']).copy().reset_index(drop=True)
    if q.duplicated(['frame','track_id']).any(): raise ValueError('Frame/ID duplicati')
    q['box_height_px']=pd.to_numeric(q.y2,errors='coerce')-pd.to_numeric(q.y1,errors='coerce')
    q['height_expected_px']=np.nan;q['height_ratio']=np.nan;q['height_suspected']=False
    for _,indices in q.groupby('track_id',sort=False).groups.items():
        history=[]; occluded=False
        for i in indices:
            h=q.at[i,'box_height_px']
            if not np.isfinite(h) or h<=0:
                q.at[i,'height_suspected']=True;occluded=True;continue
            expected=float(np.median(history[-history_size:])) if len(history)>=min_history else np.nan
            ratio=h/expected if np.isfinite(expected) and expected>0 else np.nan
            q.at[i,'height_expected_px']=expected;q.at[i,'height_ratio']=ratio
            suspect=np.isfinite(ratio) and ((not occluded and ratio<ratio_min) or (occluded and ratio<recovery_ratio))
            q.at[i,'height_suspected']=bool(suspect)
            if suspect: occluded=True
            else: occluded=False;history.append(float(h))
    return q


def fill_variant(quality,max_gap,xy_columns=('x_m','y_m'),output_columns=('x_dynamic_m','y_dynamic_m')):
    if not isinstance(max_gap,int) or max_gap<0: raise ValueError('MAX_GAP intero >= 0 richiesto')
    required={'frame','track_id','height_suspected',*xy_columns}
    if not required<=set(quality): raise ValueError(f'Mancano colonne: {required-set(quality)}')
    parts=[]
    for tid,src in quality.groupby('track_id',sort=False):
        g=src.sort_values('frame').set_index('frame')
        frames=np.arange(int(g.index.min()),int(g.index.max())+1)
        g=g.reindex(frames)
        raw=g[list(xy_columns)].to_numpy(dtype=float)
        suspected=g.height_suspected.fillna(True).astype(bool).to_numpy()
        valid=np.isfinite(raw).all(axis=1)&~suspected
        result=raw.copy();result[~valid]=np.nan
        source=np.where(valid,'observed','missing').astype(object)
        i=0
        while i<len(result):
            if valid[i]: i+=1;continue
            start=i
            while i<len(result) and not valid[i]: i+=1
            end=i
            left=start>0 and np.isfinite(result[start-1]).all()
            right=end<len(result) and np.isfinite(result[end]).all()
            if end-start<=max_gap and left and right:
                result[start:end]=np.linspace(result[start-1],result[end],end-start+2)[1:-1]
                source[start:end]='interpolated'
            elif left:
                result[start:end]=result[start-1];source[start:end]='held_last'
            elif right:
                result[start:end]=result[end];source[start:end]='backfilled'
        fallback=~np.isfinite(result).all(axis=1)&np.isfinite(raw).all(axis=1)
        result[fallback]=raw[fallback];source[fallback]='fallback_raw'
        part=pd.DataFrame(dict(frame=frames,track_id=tid,height_suspected=suspected,max_gap=max_gap,source=source))
        part[output_columns[0]]=result[:,0];part[output_columns[1]]=result[:,1]
        parts.append(part)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['frame','track_id',*output_columns,'source'])


def pixel_variant(tracks,max_gap):
    """Adattamento della stessa regola ai pixel; distinto dall'originale sul campo."""
    q=flag_height(tracks)
    p=fill_variant(q,max_gap,('foot_x','foot_y'),('corrected_foot_x','corrected_foot_y'))
    p['foot_x']=p.corrected_foot_x;p['foot_y']=p.corrected_foot_y
    return p
