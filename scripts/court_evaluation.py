"""Valuta in pixel il correttore originale del campo, senza GT metrica inventata.

Ogni output sul campo è riproiettato con l'inversa della stessa H del frame.
Il confronto pixel non è una misura indipendente dell'accuratezza fisica di H.
"""
import json
import numpy as np
import pandas as pd
from height_corrector import flag_height,fill_variant


def to_pixel(positions,homographies,x_col='x_m',y_col='y_m'):
    d=positions.copy();d['foot_x']=np.nan;d['foot_y']=np.nan
    for frame,indices in d.groupby('frame').groups.items():
        h=homographies.get(str(int(frame)))
        if h is None:continue
        h=np.asarray(h,dtype=float)
        if h.shape!=(3,3) or not np.isfinite(h).all():raise ValueError('H immagine->campo feet non valida')
        try:inverse=np.linalg.inv(h)
        except np.linalg.LinAlgError:continue
        xy=d.loc[indices,[x_col,y_col]].to_numpy(dtype=float)/.3048
        q=np.column_stack([xy,np.ones(len(xy))])@inverse.T
        good=np.isfinite(q).all(axis=1)&(np.abs(q[:,2])>1e-10)
        projected=np.full((len(xy),2),np.nan)
        projected[good]=q[good,:2]/q[good,2,None]
        d.loc[indices,['foot_x','foot_y']]=projected
    return d


def build_predictors(root,manifest):
    configs={r.clip_id:r for r in manifest.itertuples()};cache={}
    for r in configs.values():
        for key in ['positions_csv','homographies_json']:
            value=getattr(r,key,None)
            if not isinstance(value,str) or not value.strip() or not (root/value).is_file():
                raise FileNotFoundError(f'{r.clip_id}: compilare {key} nel manifest con output della stessa esecuzione')

    def inputs(raw):
        clip=raw.attrs['clip_id'];r=configs[clip]
        if clip not in cache:
            p=pd.read_csv(root/r.positions_csv,dtype={'track_id':str})
            if not {'x_ft','y_ft','frame','track_id'}<=set(p):raise ValueError('Posizioni attese in feet: x_ft,y_ft')
            if p.duplicated(['frame','track_id']).any():raise ValueError('Posizioni duplicate')
            p['x_m']=p.x_ft*.3048;p['y_m']=p.y_ft*.3048
            q=raw.merge(p[['frame','track_id','x_m','y_m']],on=['frame','track_id'],how='left',validate='one_to_one')
            q=flag_height(q)
            h=json.loads((root/r.homographies_json).read_text())
            if not h:raise ValueError(f'{clip}: omografie vuote')
            # Le posizioni raw devono riproiettarsi sui piedi osservati; altrimenti
            # probabilmente H, unità, verso o versioni non corrispondono.
            baseline=to_pixel(q,h)
            check=raw.merge(baseline[['frame','track_id','foot_x','foot_y']],on=['frame','track_id'],suffixes=('_raw','_back'))
            finite=np.isfinite(check[['foot_x_raw','foot_y_raw','foot_x_back','foot_y_back']]).all(axis=1)
            error=np.hypot(check.foot_x_raw-check.foot_x_back,check.foot_y_raw-check.foot_y_back)
            if finite.any() and error[finite].max()>1.:
                raise ValueError(f'{clip}: raw e H non coerenti entro 1 px; verificare provenienza/verso/unità')
            if not finite.any():raise ValueError(f'{clip}: nessun punto raw riproiettabile per verificare H')
            cache[clip]=(q,h,baseline)
        return cache[clip]

    def baseline(raw):return inputs(raw)[2].copy()

    def corrected(raw,gap):
        q,h,_=inputs(raw)
        result=fill_variant(q,gap)
        return to_pixel(result,h,'x_dynamic_m','y_dynamic_m')

    return baseline,corrected
