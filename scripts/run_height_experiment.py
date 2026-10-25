"""Esegue il correttore originale sui CSV Boston esistenti, senza modelli/GPU."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_config import ROOT
from thesis_tools import new_run,fingerprint
from height_corrector import flag_height,fill_variant


def run_boston():
    out=new_run(ROOT,'height_original_boston')
    outputs=ROOT/'outputs'
    tracks_path=outputs/'boston-celtics-new-york-knicks-game-1-q2-08.09-08.03_tracks.csv'
    positions_path=outputs/'nba_positions_2d.csv'
    tracks=pd.read_csv(tracks_path)
    positions=pd.read_csv(positions_path)
    if positions.duplicated(['frame','track_id']).any(): raise ValueError('Posizioni duplicate')
    q=positions[['frame','track_id','x_ft','y_ft','status']].merge(tracks,on=['frame','track_id'],validate='one_to_one')
    if len(q)!=len(positions): raise ValueError('Tracking e posizioni non allineati')
    q['x_m']=q.x_ft*.3048;q['y_m']=q.y_ft*.3048
    q=flag_height(q)
    q.to_csv(out/'height_quality.csv',index=False)
    raw_delta=q.groupby('track_id')[['frame','x_m','y_m']].diff()
    raw_speed=(np.hypot(raw_delta.x_m,raw_delta.y_m)*30).where(raw_delta.frame.eq(1))
    raw_finite=np.isfinite(q[['x_m','y_m']]).all(axis=1)
    raw_outside=raw_finite&((q.x_m<0)|(q.x_m>94*.3048)|(q.y_m<0)|(q.y_m>50*.3048))
    rows=[dict(method='raw',max_gap=0,n_rows=len(q),finite_positions=int(raw_finite.sum()),outside=int(raw_outside.sum()),
               speed_alerts_over12=int(raw_speed.gt(12).sum()),interpolated=0,held_last=0,backfilled=0,fallback_raw=0)]
    variants={}
    for gap in range(1,7):
        d=fill_variant(q,gap)
        variants[gap]=d
        d.to_csv(out/f'positions_gap{gap}.csv',index=False)
        delta=d.groupby('track_id')[['frame','x_dynamic_m','y_dynamic_m']].diff()
        speed=(np.hypot(delta.x_dynamic_m,delta.y_dynamic_m)*30).where(delta.frame.eq(1))
        finite=np.isfinite(d[['x_dynamic_m','y_dynamic_m']]).all(axis=1)
        outside=finite&((d.x_dynamic_m<0)|(d.x_dynamic_m>94*.3048)|(d.y_dynamic_m<0)|(d.y_dynamic_m>50*.3048))
        rows.append(dict(method='original_height',max_gap=gap,n_rows=len(d),finite_positions=int(finite.sum()),outside=int(outside.sum()),
                         speed_alerts_over12=int(speed.gt(12).sum()),interpolated=int(d.source.eq('interpolated').sum()),
                         held_last=int(d.source.eq('held_last').sum()),backfilled=int(d.source.eq('backfilled').sum()),
                         fallback_raw=int(d.source.eq('fallback_raw').sum())))
    pd.DataFrame(rows).to_csv(out/'diagnostics_per_gap.csv',index=False)
    fig,axes=plt.subplots(5,2,figsize=(13,20))
    for ax,(tid,g) in zip(axes.flat,q.groupby('track_id')):
        ax.plot(g.x_m,g.y_m,color='gray',alpha=.55,label='raw')
        for gap in [3,5,6]:
            d=variants[gap];d=d[d.track_id.eq(tid)]
            ax.plot(d.x_dynamic_m,d.y_dynamic_m,linewidth=1,label=f'gap {gap}')
        ax.set(title=f'ID {tid}',xlabel='x proiettata (m)',ylabel='y proiettata (m)',xlim=(-2,31),ylim=(-2,18))
        ax.set_aspect('equal');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.suptitle('Boston: diagnostica del correttore originale, nessuna selezione su ground truth')
    fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
    original=ROOT/'archivio/notebook_originali/03_NBA_tracking_diagnostics_regola_altezza (3).ipynb'
    (out/'provenance.json').write_text(json.dumps(dict(method='original_height_court',
        source_notebook=original.relative_to(ROOT).as_posix(),source_sha256=fingerprint(original),source_cells=[19,21,38],
        inputs={str(p.relative_to(ROOT)):fingerprint(p) for p in [tracks_path,positions_path]},
        parameters=dict(height_ratio_min=.82,recovery_ratio=.90,history=15,min_history=5,gaps=list(range(1,7)),fps=30.),
        offline_future_information=True,ground_truth_evaluated=False,selected_gap=None,
        notes='Interpolazione breve, hold-last su gap lunghi/coda, backfill iniziale e fallback raw senza ancore. Non è accuratezza fisica.'),indent=2),encoding='utf-8')
    print(out,flush=True)
    return out


if __name__=='__main__':run_boston()
