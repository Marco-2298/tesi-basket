"""Generatore delle copie revisionate; gli originali non vengono modificati."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS = ROOT / 'archivio' / 'notebook_originali'

def cell(src, kind='code'):
    c = dict(cell_type=kind, metadata={}, source=textwrap.dedent(src).strip()+'\n')
    if kind == 'code': c.update(outputs=[], execution_count=None)
    return c

def save(name, cells):
    p = ROOT/name
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=1),encoding='utf-8')

CONFIG = '''
from pathlib import Path
import os, sys, json, csv
from datetime import datetime, timezone
PROJECT_DIR = Path(os.environ.get('NBA_PROJECT_DIR', Path.cwd())).expanduser().resolve()
if not (PROJECT_DIR / 'scripts' / 'thesis_tools.py').is_file():
    raise FileNotFoundError('Impostare NBA_PROJECT_DIR alla cartella del progetto (in Colab montare prima Drive).')
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from thesis_tools import new_run
ROOT = PROJECT_DIR
RUN_DIR = new_run(ROOT, 'baseline')
WORK_DIR = RUN_DIR / 'work'
WORK_DIR.mkdir()
CLIP_ID = os.environ.get('NBA_CLIP_ID', '')
CONFIDENCE = 0.4
NMS_IOU = 0.70
YOLO_WEIGHTS = Path(os.environ.get('NBA_YOLO_WEIGHTS', str(ROOT/'models'/'yolo11_players.pt')))
print('Output isolati:', RUN_DIR)
'''

ADAPTER = '''
from types import SimpleNamespace
from ultralytics import YOLO
if not YOLO_WEIGHTS.is_file():
    raise FileNotFoundError(f'Fornire il checkpoint YOLO11 e NBA_YOLO_WEIGHTS: {YOLO_WEIGHTS}')
class YoloAdapter:
    def __init__(self, path):
        self.model = YOLO(str(path))
        if 'yolo11' not in str(self.model.model.yaml.get('yaml_file', '')).lower():
            raise ValueError('Verificare che il checkpoint sia YOLO11: architettura non riconosciuta')
    def infer(self, image, confidence=0.4, iou_threshold=0.7):
        r = self.model.predict(image, conf=confidence, iou=iou_threshold, verbose=False)[0]
        predictions = []
        for xyxy, score, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            x1,y1,x2,y2 = map(float, xyxy)
            name = str(r.names[int(cls)])
            # Il modello COCO generico 'person' include arbitri: verificare il checkpoint.
            predictions.append(SimpleNamespace(x=(x1+x2)/2,y=(y1+y2)/2,width=x2-x1,height=y2-y1,confidence=float(score),class_name=name))
        return [SimpleNamespace(predictions=predictions)]
player_model = YoloAdapter(YOLO_WEIGHTS)
PLAYER_MODEL_ID = str(YOLO_WEIGHTS)
'''

def revised01():
    old=json.loads((ORIGINALS/'01_NBA_tracking_baseline_finale.ipynb').read_text(encoding='utf-8'))['cells']
    cells=[cell('''# 01 — Baseline YOLO11 + SAM2, copia revisionata
    Originale conservato. Richiede GPU, repository SAM2, checkpoint e modello del campo.
    In Colab montare Drive prima della configurazione e impostare NBA_PROJECT_DIR.
    Installazioni esplicite nell’ambiente: ultralytics, supervision==0.27.0, sports (branch feat/basketball), inference (CPU per keypoint), opencv-python, numpy, pandas, scipy, matplotlib, tqdm.
    Le API esterne e le dipendenze non sono state verificate qui. Nessuna installazione automatica di inference-gpu.
    Una sola clip per esecuzione; scripts/run_batch.py esegue il notebook in kernel separati.
    SAM2 inizializzato al frame 0: ingressi successivi restano un limite della baseline.
    ''','markdown'),cell(CONFIG)]
    for i,c in enumerate(old,1):
        if i in [1,5,7,11,19,23,52]:
            replacement={1:None,5:"print('Drive deve essere già montato se necessario.')",7:"import supervision as sv",11:"print('Configurazione già definita nella cella 2.')",19:"import importlib.util\nif importlib.util.find_spec('inference') is None:\n    raise ImportError('Installare inference CPU per i keypoint, oppure recuperare gli output geometrici esistenti.')",23:ADAPTER,52:None}[i]
            if replacement: cells.append(cell(replacement))
            continue
        src=''.join(c['source'])
        if i==13:
            src="""import shutil
if not CLIP_ID:
    raise ValueError('Selezionare NBA_CLIP_ID dal manifest; Boston non viene inclusa automaticamente.')
clips = ROOT / 'datasets' / 'tracking_test_v1' / 'clips'
candidates = [p for p in clips.glob('*') if p.is_file() and p.stem == CLIP_ID]
if len(candidates) != 1:
    raise FileNotFoundError(f'Clip {CLIP_ID}: trovati {candidates} in {clips}')
DRIVE_VIDEO_PATH = candidates[0]
VIDEO_NAME = DRIVE_VIDEO_PATH.name
VIDEO_PATH = WORK_DIR / VIDEO_NAME
shutil.copy2(DRIVE_VIDEO_PATH, VIDEO_PATH)
(RUN_DIR/'run_config.json').write_text(json.dumps(dict(clip_id=CLIP_ID,video=str(DRIVE_VIDEO_PATH),weights=str(YOLO_WEIGHTS),confidence=CONFIDENCE,nms_iou=NMS_IOU)),encoding='utf-8')
"""
        src=src.replace('PROJECT_DIR / "outputs"','RUN_DIR').replace('PROJECT_DIR / "configs" / "sam2_commit.txt"','RUN_DIR / "sam2_commit.txt"')
        if i==21: src='import PIL\nfrom PIL import Image, ImageFont\nprint(PIL.__version__)'
        if i==25: src=src.replace('confidence=0.4','confidence=CONFIDENCE').replace('iou_threshold=0.9','iou_threshold=NMS_IOU')
        if i==27: src=src.replace('remove_duplicate_boxes(boxes, scores)','remove_duplicate_boxes(boxes, scores, iou_threshold=NMS_IOU)')+"\nif len(initial_boxes) == 0:\n    raise RuntimeError('Nessun giocatore iniziale: verificare classi e checkpoint YOLO11.')\n"
        if i==49:
            src=src.replace('tracks_by_frame = defaultdict(list)','homographies = {}\ntracks_by_frame = defaultdict(list)')
            src=src.replace('H = inverse','H = inverse\n                                homographies[str(processed)] = H.tolist()')
            src += "\n(RUN_DIR/'homographies_image_to_court_ft.json').write_text(json.dumps(homographies),encoding='utf-8')\n"
        if i==33: src=src.replace('sys.executable, "-m", "pip", "install", "-e", str(SAM2_DIR)','sys.executable, "-m", "pip", "install", "-e", str(SAM2_DIR)')
        if i==35: src=src.replace('os.chdir(SAM2_DIR)', '# Percorsi assoluti: mantieni la directory del progetto.')
        if c['cell_type']=='markdown': src=src.replace('RF-DETR','YOLO11').replace('Installazione della libreria Roboflow Inference','Verifica di Roboflow Inference CPU')
        cells.append(cell(src,c['cell_type']))
    cells.append(cell('''## Interpretazione
    Le coordinate della mappa sono in feet (1 ft = 0.3048 m). Le omografie sono salvate per frame.
    Il residuo sui keypoint usati nel fit non misura l’accuratezza fisica indipendente.
    Risultati numerici dell’originale non sono stati trasferiti alla nuova baseline.
    ''','markdown'))
    save('01_NBA_tracking_revisionato.ipynb',cells)

def revised02():
    old=json.loads((ORIGINALS/'02_NBA_detection_evaluation_finale.ipynb').read_text(encoding='utf-8'))['cells']
    cells=[cell('# 02 — Valutazione detection YOLO11\nDataset COCO indipendente da verificare; distinta dalla valutazione temporale. Originale conservato.','markdown'),cell(CONFIG)]
    for i,c in enumerate(old,1):
        if i in [1,2,3,4,5,6,7,9,10,11]: continue
        src=''.join(c['source'])
        if i==8: src=ADAPTER+'\nimport hashlib\nfrom pycocotools.coco import COCO'
        if i==13: src="print('File COCO:', list(ROOT.glob('datasets/**/*coco*.json')))"
        if i==14: src=src[src.index('COCO_JSON ='):]; src="TEST_IMAGES = ROOT/'datasets'/'basketball_player_detection_v18'/'test'\n"+src
        if i==18: src=src.replace("OUT = ROOT / 'outputs' / ('detection_eval_v4_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))\nOUT.mkdir(parents=True, exist_ok=False)","OUT = new_run(ROOT, 'detection_evaluation')\nimport json")
        cells.append(cell(src,c['cell_type']))
    save('02_NBA_detection_revisionato.ipynb',cells)

def new_notebooks():
    save('04_NBA_manifest_annotazioni.ipynb',[
        cell('# 04 — Inventario e preparazione annotazioni\nEseguire per primo. Nessuna annotazione viene inventata. Circa 100 punti giocatore-frame, più due sequenze di 15 frame per clip (identità coerenti); controllare e adattare i frame al video.','markdown'),cell(CONFIG),
        cell('''
        import numpy as np
        import pandas as pd
        import cv2
        clips_dir = ROOT/'datasets'/'tracking_test_v1'/'clips'
        paths = sorted(p for p in clips_dir.glob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.avi'})
        if len(paths) != 7:
            raise FileNotFoundError(f'Attese 7 clip in {clips_dir}, trovate {len(paths)}. Boston dimostrativa va tenuta in videos/.')
        if len({p.stem for p in paths}) != 7:
            raise ValueError('I nomi clip senza estensione devono essere univoci')
        OUT = new_run(ROOT, 'annotation_preparation')
        records, plans = [], []
        for j,p in enumerate(paths):
            cap=cv2.VideoCapture(str(p))
            fps=cap.get(cv2.CAP_PROP_FPS); count=0
            width=cap.get(cv2.CAP_PROP_FRAME_WIDTH); height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while True:
                ok, image=cap.read()
                if not ok: break
                count+=1
            cap.release()
            if count < 1 or not np.isfinite(fps) or fps <= 0: raise ValueError(f'Video non leggibile: {p}')
            records.append(dict(clip_id=p.stem,video=str(p.relative_to(ROOT)),fps=fps,frames=count,width=width,height=height,duration_s=count/fps,split='',shot_type='',shot_verified=False,tracks_csv=''))
            # 15 punti per prime due clip e 14 per altre cinque = 100 slot, non etichette.
            for f in np.linspace(0,count-1,15 if j<2 else 14).round().astype(int):
                plans.append(dict(clip_id=p.stem,frame=int(f),player_id='',visibility='',annotatable='',x_px='',y_px='',sequence_id=''))
        manifest=pd.DataFrame(records)
        manifest.to_csv(OUT/'manifest.csv',index=False)
        pd.DataFrame(plans).to_csv(OUT/'point_annotation_slots.csv',index=False)
        seq=[]
        for r in records:
            for k,start in enumerate([r['frames']//4,r['frames']//2],1):
                for f in range(start,min(start+15,r['frames'])):
                    seq.append(dict(clip_id=r['clip_id'],frame=f,player_id='',visibility='',annotatable='',x_px='',y_px='',sequence_id=f'seq{k}'))
        pd.DataFrame(seq).to_csv(OUT/'sequence_annotation_slots.csv',index=False)
        display(manifest)
        print('Compilare copie dei CSV; unire punti e sequenze senza duplicare clip/frame/player_id:',OUT)
        '''),
        cell('''## Protocollo manuale e split
        Frame indicizzati da zero, coordinate nel video originale. player_id è un’identità umana stabile nella clip, indipendente da SAM2.
        visibility: visible / partial / occluded; annotatable: 1 solo se il contatto a terra è localizzabile con affidabilità, 0 altrimenti.
        Non usare il bordo della maschera come ground truth. Annotare tutti i giocatori selezionati nelle sequenze consecutive, aggiungendo righe per ciascuno.
        Conservare anche gli occlusi non annotabili; le metriche pixel usano solo annotatable=1.
        shot_type: half_court / wide / pan_between_halves, verificato da persona (shot_verified=True).
        Assegnare e congelare validation/test per clip prima di confrontare i candidati, ad esempio 3/4 clip cercando categorie rappresentate.
        Sette clip non sostengono generalizzazioni; il campione può non consentire confronti attendibili per categoria.
        tracks_csv deve indicare un CSV grezzo relativo al progetto, riusando output validi; non associare output Boston a queste clip.
        ''','markdown')])
    save('03_NBA_correzione_valutazione.ipynb',[
        cell('# 03 — Correzione e valutazione offline\nInput: manifest compilato, annotazioni manuali e CSV grezzi. Il correttore originale manca: il candidato implementato è una nuova interpolazione lineare entro ID, senza smoothing né eliminazioni per velocità.','markdown'),cell(CONFIG),
        cell('''
        import pandas as pd
        import numpy as np
        from thesis_tools import evaluate, load_tracks, correct, projection_diagnostics
        MANIFEST = ROOT/'annotations'/'manifest.csv'
        ANNOTATIONS = ROOT/'annotations'/'annotations.csv'
        MAX_GAP_CANDIDATES = (0,2,5,10)
        MATCH_GATE_PX = 80.0  # fissare prima del test; riportare sensibilità separata
        for p in [MANIFEST, ANNOTATIONS]:
            if not p.is_file(): raise FileNotFoundError(f'Compilare il file manuale: {p}; vedere Notebook 04')
        OUT, per_clip, matched = evaluate(ROOT, MANIFEST, ANNOTATIONS, MAX_GAP_CANDIDATES, MATCH_GATE_PX)
        display(per_clip)
        print('Risultati:',OUT)
        '''),
        cell('''
        import matplotlib.pyplot as plt
        summary=pd.read_csv(OUT/'aggregate.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for split,g in summary.groupby('split'):
            axes[0].plot(g.max_gap,g.rmse_px,marker='o',label=split)
            axes[1].plot(g.max_gap,g.coverage,marker='o',label=split)
        axes[0].set(ylabel='RMSE abbinati (px)',xlabel='MAX_GAP (frame)')
        axes[1].set(ylabel='Copertura annotazioni',xlabel='MAX_GAP (frame)',ylim=(0,1.05))
        for ax in axes: ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=180)
        display(pd.read_csv(OUT/'final_test.csv'))
        '''),
        cell('''
        # Esportazione traiettorie e diagnostica pixel: velocità non usata per cancellare punti.
        manifest=pd.read_csv(MANIFEST)
        chosen=json.loads((OUT/'provenance.json').read_text())['chosen_max_gap']
        diagnostic=[]
        for r in manifest.itertuples():
            for gap in sorted({0,chosen}):
                d=correct(load_tracks(ROOT/r.tracks_csv),int(gap))
                d.to_csv(OUT/f'{r.clip_id}_gap{gap}_tracks.csv',index=False)
                for tid,g in d.groupby('track_id'):
                    dt=g.frame.diff()/float(r.fps)
                    speed=np.hypot(g.foot_x.diff(),g.foot_y.diff())/dt
                    diagnostic.append(dict(clip_id=r.clip_id,max_gap=gap,track_id=tid,max_speed_px_s=speed.max(),n_transitions=speed.notna().sum()))
        pd.DataFrame(diagnostic).to_csv(OUT/'pixel_speed_diagnostics.csv',index=False)
        geo_summary=[]
        for r in manifest.itertuples():
            # H salvate dal Notebook 01; per output storici aggiungere il percorso nel manifest.
            hpath=ROOT/getattr(r,'homographies_json','__missing__') if isinstance(getattr(r,'homographies_json',None),str) else ROOT/'__missing__'
            if not hpath.is_file():
                print(r.clip_id, 'diagnostica metrica non disponibile: compilare homographies_json nel manifest')
                continue
            hs=json.loads(hpath.read_text())
            for gap in sorted({0,chosen}):
                d=projection_diagnostics(correct(load_tracks(ROOT/r.tracks_csv),int(gap)),hs,float(r.fps))
                d.to_csv(OUT/f'{r.clip_id}_gap{gap}_projection.csv',index=False)
                geo_summary.append(dict(clip_id=r.clip_id,max_gap=gap,n_rows=len(d),n_projected=d.x_m.notna().sum(),outside=d.status.eq('outside').sum(),implausible_speed=d.implausible_speed.sum()))
        if geo_summary:
            pd.DataFrame(geo_summary).to_csv(OUT/'projection_per_clip.csv',index=False)
            pd.DataFrame(geo_summary).groupby('max_gap').sum(numeric_only=True).to_csv(OUT/'projection_aggregate.csv')
        '''),
        cell('''
        # Esempio su frame annotato: croci GT, cerchi raw, quadrati corretti.
        import cv2
        example=matched[matched.split.eq('test')].sort_values('error_px',ascending=False).iloc[0]
        r=manifest[manifest.clip_id.eq(example.clip_id)].iloc[0]
        cap=cv2.VideoCapture(str(ROOT/r.video)); cap.set(cv2.CAP_PROP_POS_FRAMES,int(example.frame))
        ok,image=cap.read(); cap.release()
        if not ok: raise RuntimeError('Frame esempio non leggibile')
        fig,ax=plt.subplots(figsize=(14,7)); ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        raw=load_tracks(ROOT/r.tracks_csv)
        for label,d,marker in [('raw',raw,'o'),('corretto',correct(raw,chosen),'s')]:
            p=d[d.frame.eq(example.frame)]; ax.scatter(p.foot_x,p.foot_y,marker=marker,label=label,facecolors='none',edgecolors='yellow' if label=='raw' else 'cyan')
        a=matched[(matched.clip_id.eq(example.clip_id)) & matched.frame.eq(example.frame) & matched.max_gap.eq(0)]
        ax.scatter(a.x_px,a.y_px,c='red',marker='x',label='GT'); ax.legend()
        fig.savefig(OUT/'error_example.png',dpi=180)
        '''),
        cell('''## Interpretazione delle metriche
        Matching ungherese 1:1 per frame entro soglia pixel: copertura e RMSE vanno letti insieme; i grandi errori esclusi dal gate abbassano la copertura.
        ID switch: cambi di ID predetto tra associazioni dello stesso giocatore in una sequenza consecutiva, anche dopo mancate associazioni.
        Frammentazione: ripresa di un’associazione dopo un’interruzione interna. Frame non annotabili/non consecutivi interrompono il confronto.
        Queste sono misure operative sui giocatori annotati, non IDF1/HOTA/MOTA su scene completamente annotate.
        L’interpolazione non corregge scambi di identità. Le metriche aggregate sono pooled; la selezione usa medie per clip di validazione.
        I candidati test esportati servono per trasparenza: non selezionare parametri dal test.
        Nessun errore assoluto in metri è dichiarabile da annotazioni pixel. Proiezioni fuori campo e velocità metriche richiedono CSV geometrici e unità esplicite.
        ''','markdown')])

if __name__=='__main__':
    revised01(); revised02(); new_notebooks()
