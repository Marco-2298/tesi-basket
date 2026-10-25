"""Aggiornamento riproducibile; backup integrale prima di cambiare ogni notebook."""
import ast
import json
from pathlib import Path
from datetime import datetime, timezone
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_notebooks import cell

CONFIG='''
from pathlib import Path
import os,sys,json,csv
from datetime import datetime,timezone
PROJECT_DIR=Path(os.environ.get('NBA_PROJECT_DIR',Path.cwd())).expanduser().resolve()
if not (PROJECT_DIR/'configs/project.json').is_file():
    raise FileNotFoundError('Aprire dalla cartella progetto oppure impostare NBA_PROJECT_DIR. In Colab usare la radice su Drive.')
sys.path.insert(0,str(PROJECT_DIR/'scripts'))
from project_config import load_config
from thesis_tools import new_run
ROOT=PROJECT_DIR
CONFIG=load_config()
CLIPS_DIR=CONFIG['clips_dir']
TEST_IMAGES=CONFIG['detection_test_dir']
YOLO_WEIGHTS=Path(os.environ.get('NBA_YOLO_WEIGHTS',str(CONFIG['weights'])))
CONFIDENCE=CONFIG['confidence']
NMS_IOU=CONFIG['nms_iou']
CLIP_ID=os.environ.get('NBA_CLIP_ID','clip_01')  # pilot esplicito; batch imposta ogni clip
RUN_DIR=new_run(ROOT,'NOTEBOOK_KIND')
WORK_DIR=RUN_DIR/'work';WORK_DIR.mkdir()
print('Detector:',CONFIG['model_variant'],'| clip:',CLIP_ID,'| output:',RUN_DIR)
'''
ADAPTER='''
from yolo_onnx import YoloOnnxAdapter
from thesis_tools import fingerprint
player_model=YoloOnnxAdapter(YOLO_WEIGHTS)
PLAYER_MODEL_ID=CONFIG['model_id']
(RUN_DIR/'detector_provenance.json').write_text(json.dumps(dict(
    model_id=PLAYER_MODEL_ID,variant=CONFIG['model_variant'],weights_sha256=fingerprint(YOLO_WEIGHTS),
    metadata=player_model.metadata,preprocessing='RGB stretch 1280x1280 /255',runtime='onnxruntime CPU'),indent=2),encoding='utf-8')
'''


def save(name,cells,backup):
    p=ROOT/name
    if p.exists():shutil.copy2(p,backup/name)
    for i,c in enumerate(cells,1):
        c['id']=f'cell-{i:03d}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    p.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
                                 nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=1),encoding='utf-8')


def main():
    backup=ROOT/'reports'/('notebook_backups_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True,exist_ok=False)
    changed=[]
    for name in ['01_NBA_tracking_revisionato.ipynb','02_NBA_detection_revisionato.ipynb']:
        n=json.loads((ROOT/name).read_text(encoding='utf-8'));cells=n['cells']
        cells[1]=cell(CONFIG.replace('NOTEBOOK_KIND','baseline' if name.startswith('01') else 'detection_preparation'))
        for c in cells:
            s=''.join(c['source'])
            if c['cell_type']=='code':
                if 'class YoloAdapter:' in s or 'player_model=YoloOnnxAdapter' in s:
                    s=ADAPTER+ ('\nimport hashlib\nfrom pycocotools.coco import COCO' if name.startswith('02') else '')
                s=s.replace("clips = ROOT / 'datasets' / 'tracking_test_v1' / 'clips'","clips = CLIPS_DIR")
                s=s.replace("TEST_IMAGES = ROOT/'datasets'/'basketball_player_detection_v18'/'test'","TEST_IMAGES = CONFIG['detection_test_dir']")
                if name.startswith('01'):
                    s=s.replace('SAM2_DIR = Path("/content/segment-anything-2-real-time")',
                                'SAM2_DIR = Path(os.environ.get("NBA_SAM2_DIR", "/content/segment-anything-2-real-time"))')
                    s=s.replace('SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt"',
                                'SAM2_CHECKPOINT = Path(os.environ.get("NBA_SAM2_CHECKPOINT", str(SAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt")))')
                    if '"git", "clone"' in s and 'SAM2_REF =' not in s:
                        marker='subprocess.run(\n    [sys.executable'
                        pin='''SAM2_REF = os.environ.get("NBA_SAM2_REF", "e3623cda3f748df999f76db663f08da7a884a820")
dirty = subprocess.check_output(["git", "-C", str(SAM2_DIR), "status", "--porcelain"], text=True).strip()
if dirty:
    raise RuntimeError("Repository SAM2 con modifiche locali: conservarle prima di selezionare il commit della baseline.")
subprocess.run(["git", "-C", str(SAM2_DIR), "checkout", "--detach", SAM2_REF], check=True)

'''
                        s=s.replace(marker,pin+marker)
                    if 'court_model = get_model(model_id=COURT_MODEL_ID)' in s:
                        s=s.replace('court_model = get_model(model_id=COURT_MODEL_ID)', '''api_key=os.environ.get("ROBOFLOW_API_KEY", "")
if not api_key:
    try:
        from google.colab import userdata
        api_key=userdata.get("ROBOFLOW_API_KEY")
    except Exception:
        api_key=""
if not api_key:
    raise RuntimeError("Aggiungere ROBOFLOW_API_KEY nei Secrets Colab e abilitare l'accesso; non inserire chiavi nel notebook.")
court_model = get_model(model_id=COURT_MODEL_ID, api_key=api_key)
del api_key''')
                if name.startswith('01') and "run_config.json" in s and 'input_video_sha256.txt' not in s:
                    s+='\nfrom thesis_tools import fingerprint\n(RUN_DIR/"input_video_sha256.txt").write_text(fingerprint(DRIVE_VIDEO_PATH))\n'
                if name.startswith('02') and 'predictions=[]' in s and 'RECOMPUTE_DETECTIONS=' not in s:
                    s=s.replace('predictions=[]',"predictions=[]")
                    # L'inferenza COCO resta esplicita: i risultati storici sono già disponibili.
                    s="""RECOMPUTE_DETECTIONS=False  # Cambiare solo per un nuovo esperimento motivato
if not RECOMPUTE_DETECTIONS:
    raise RuntimeError('Predizioni COCO già presenti in outputs/detection_eval_v4_20261011T093539Z. Riutilizzarle dopo verifica di protocollo e indipendenza; impostare True solo per nuova inferenza.')
"""+s
                c['source']=s
            elif name.startswith('01') and 'ultralytics' in s:
                c['source']=s.replace('ultralytics,','onnxruntime,').replace('copia revisionata','copia revisionata (YOLO11-small ONNX)')
            if c['cell_type']=='markdown' and name.startswith('01'):
                c['source']=''.join(c['source']).replace('supervision==0.27.0','supervision==0.30.9')
        if name.startswith('02'):
            cells[2]=cell('''
            import hashlib
            from pycocotools.coco import COCO
            USE_SAVED_PREDICTIONS=True  # default: nessuna nuova inferenza sulle 94 immagini
            SAVED_RUN=ROOT/'outputs/detection_eval_v4_20261011T093539Z'
            PLAYER_MODEL_ID=CONFIG['model_id']
            if not USE_SAVED_PREDICTIONS:
                from yolo_onnx import YoloOnnxAdapter
                player_model=YoloOnnxAdapter(YOLO_WEIGHTS)
            ''')
            cells[4]=cell("print('Annotazioni:',CONFIG['detection_test_dir']/'_annotations.coco.json')")
            s=''.join(cells[7]['source'])
            marker='# Controllo di una predizione reale;'
            if marker in s:
                s=s[:s.index(marker)]+"\nimport cv2\nprint('Classi predette da verificare:',sorted(PRED_PLAYER_CLASSES))\n"
            cells[7]=cell(s)
            original=json.loads((ROOT/'archivio/notebook_originali/02_NBA_detection_evaluation_finale.ipynb').read_text(encoding='utf-8'))
            inference=''.join(original['cells'][19]['source'])
            reuse='''
            saved_metrics=json.loads((SAVED_RUN/'metrics.json').read_text())
            if saved_metrics['annotation_sha256']!=hashlib.sha256(COCO_JSON.read_bytes()).hexdigest():
                raise ValueError('Annotazioni attuali diverse da quelle delle predizioni salvate')
            if saved_metrics['model_id']!=PLAYER_MODEL_ID or set(saved_metrics['gt_classes'])!=GT_PLAYER_CLASSES or set(saved_metrics['pred_classes'])!=PRED_PLAYER_CLASSES:
                raise ValueError('Protocollo salvato non compatibile con modello/classi selezionate')
            predictions=json.loads((SAVED_RUN/'predictions.json').read_text())
            if not {p['image_id'] for p in predictions}<=set(coco_gt.getImgIds()):
                raise ValueError('Predizioni riferite a immagini sconosciute')
            CONF_FOR_AP=saved_metrics['conf_for_ap'];NMS_IOU=saved_metrics['nms_iou']
            import numpy as np
            (OUT/'predictions.json').write_text(json.dumps(predictions))
            (OUT/'reuse_provenance.json').write_text(json.dumps(dict(source=str(SAVED_RUN.relative_to(ROOT)),
                predictions_sha256=hashlib.sha256((SAVED_RUN/'predictions.json').read_bytes()).hexdigest(),
                inference_reexecuted=False,independence_verified=False),indent=2))
            print('Predizioni storiche riutilizzate:',len(predictions))
            '''
            import textwrap
            combined='if USE_SAVED_PREDICTIONS:\n'+textwrap.indent(textwrap.dedent(reuse).strip(),'    ')
            combined+='\nelse:\n'+textwrap.indent(inference,'    ')
            cells[11]=cell(combined)
        save(name,cells,backup);changed.append(name)
    cells04=[cell('''# 04 — Dataset e annotazioni locali
    I video e il manifest storico sono stati recuperati e verificati. Percorsi condivisi in configs/project.json.
    Il Notebook 04 originale è conservato separatamente; non rigenerare immagini campione inutilmente.
    ''','markdown'),cell(CONFIG.replace('NOTEBOOK_KIND','dataset_review')),
        cell('''
        import pandas as pd
        from annotation_store import read_rows,validate_dataset
        manifest=pd.read_csv(CONFIG['manifest'])
        display(manifest)
        tasks=pd.read_csv(CONFIG['tasks'])
        print('Frame con campionamento spaziale:',tasks.point_target.sum(),'| frame nel piano:',len(tasks))
        print('Annotazioni manuali compilate:',len(read_rows(CONFIG['annotations'])))
        '''),cell('''## Annotare
        In Windows aprire **Annota.cmd** nella radice del progetto. L'app usa i frame originali, senza predizioni sovrapposte.
        Cliccare il contatto a terra e salvare con una identità umana stabile (p01, p02…). Per occlusione completa o contatto non localizzabile salvare annotatable=0.
        Piano ridotto: 5 giocatori nei nuovi frame sparsi; soltanto 2 identità scelte nel pulsante dedicato nelle sequenze.
        I dati già annotati restano tutti validi. Il piano attuale conserva i 5 frame iniziali e aggiunge 2 frame per ciascuna delle altre 6 clip, più tre sequenze di 8 frame.
        Non copiare gli ID SAM2 come ground truth. Le sequenze proposte vanno verificate se attraversano un taglio.
        ''','markdown'),cell('''
        errors=validate_dataset(read_rows(CONFIG['manifest']),read_rows(CONFIG['tasks']),read_rows(CONFIG['annotations']))
        print('\\n'.join(errors) if errors else 'Struttura annotazioni completa; resta la verifica umana della qualità.')
        '''),cell('''## Split e inquadrature
        Compilare annotations/manifest.csv: validation/test per clip, senza dividere frame della stessa clip tra i due insiemi.
        Se clip provengono dalla stessa partita, preferire gruppi per partita quando possibile e dichiarare i limiti del campione.
        Usare annotations/scene_intervals.csv per distinguere cambiamenti entro clip: half_court/wide/pan_between_halves e camera_motion static/pan/zoom/mixed.
        I campi sono ground truth soltanto dopo verifica umana. Non è stata assegnata alcuna categoria automaticamente.
        ''','markdown')]
    save('04_NBA_manifest_annotazioni.ipynb',cells04,backup);changed.append('04_NBA_manifest_annotazioni.ipynb')
    cells03=[cell('''# 03 — Correttore originale e valutazione
    Recuperato il Notebook 03 originale. Sezione A: riproduzione esatta sul campo per Boston, solo diagnostica.
    Sezione B: valutazione pixel delle sette clip quando saranno compilati GT, tracking e omografie.
    Il correttore originale sul campo è riproiettato sui frame per il confronto con le annotazioni pixel.
    ''','markdown'),cell(CONFIG.replace('NOTEBOOK_KIND','evaluation_preparation')),
        cell('''## A — Riprodurre il correttore originale senza inferenza
        Soglie originali 0.82/0.90, storia 15 frame e almeno 5 osservazioni. MAX_GAP 1–6.
        Interpolazione tra ancore, hold-last per gap lunghi/coda, backfill all'inizio e fallback raw senza ancore.
        Algoritmo offline: usa anche frame futuri. Un marker sempre visibile non dimostra un punto corretto.
        ''','markdown'),cell('''
        from pathlib import Path
        from IPython.display import Image,display
        import pandas as pd
        existing=sorted((ROOT/'outputs'/'height_original_boston').glob('*/diagnostics_per_gap.csv'))
        if existing:
            BOSTON_OUT=existing[-1].parent
        else:
            from run_height_experiment import run_boston
            BOSTON_OUT=run_boston()
        display(pd.read_csv(BOSTON_OUT/'diagnostics_per_gap.csv'))
        display(Image(filename=str(BOSTON_OUT/'trajectories.png'),width=1000))
        print('Boston dimostrativa, senza GT e senza selezione di MAX_GAP:',BOSTON_OUT)
        '''),cell('''## B — Verifica prima della valutazione sulle sette clip
        Si confrontano raw e correttore originale sul campo, riproiettati in pixel con H inversa del medesimo frame.
        Sono necessari positions_csv e homographies_json della stessa esecuzione dei tracks_csv; si verifica il ritorno raw entro 1 px.
        Le metriche pixel non sono l'errore assoluto in metri della calibrazione. L'interpolazione pixel semplice resta un riferimento aggiuntivo.
        ''','markdown'),cell('''
        from annotation_store import read_rows,validate_dataset
        errors=validate_dataset(read_rows(CONFIG['manifest']),read_rows(CONFIG['tasks']),read_rows(CONFIG['annotations']))
        manifest=pd.read_csv(CONFIG['manifest'])
        if manifest.tracks_csv.isna().any() or manifest.tracks_csv.fillna('').eq('').any():
            errors.append('Registrare i CSV tracking delle sette clip in annotations/manifest.csv')
        if errors:raise RuntimeError('Input ancora da completare:\\n'+'\\n'.join(errors[:20]))
        '''),cell('''
        from thesis_tools import evaluate
        from court_evaluation import build_predictors
        raw_court,corrected_court=build_predictors(ROOT,manifest)
        OUT,per_clip,matched=evaluate(ROOT,CONFIG['manifest'],CONFIG['annotations'],
            gaps=(0,1,2,3,4,5,6),gate=CONFIG['match_gate_px'],predictor=corrected_court,
            baseline_predictor=raw_court,method_label='original_height_court_reprojected_to_pixel')
        REF_OUT,_,_=evaluate(ROOT,CONFIG['manifest'],CONFIG['annotations'],
            gaps=(0,2,5,10),gate=CONFIG['match_gate_px'],method_label='bounded_pixel_interpolation_reference')
        display(per_clip);display(pd.read_csv(OUT/'final_test.csv'))
        print('Correttore originale riproiettato in pixel:',OUT,'| riferimento pixel:',REF_OUT)
        '''),cell('''
        import matplotlib.pyplot as plt
        summary=pd.read_csv(OUT/'aggregate.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for split,g in summary.groupby('split'):
            axes[0].plot(g.max_gap,g.rmse_px,marker='o',label=split)
            axes[1].plot(g.max_gap,g.coverage,marker='o',label=split)
        axes[0].set(ylabel='RMSE match (px)',xlabel='MAX_GAP')
        axes[1].set(ylabel='Copertura GT annotabile',xlabel='MAX_GAP',ylim=(0,1.05))
        for ax in axes:ax.grid(alpha=.3);ax.legend()
        fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180)
        '''),cell('''## Protocollo
        Matching spaziale 1:1 entro gate pixel: errori condizionati ai match, da leggere insieme alla copertura.
        ID switch e frammentazioni si calcolano soltanto su sequenze manuali consecutive; non sono HOTA/IDF1/MOTA complete.
        Selezione MAX_GAP sulle sole clip validation; test finale per baseline e candidato selezionato.
        Il test esportato per tutti i candidati è trasparenza sperimentale, non materiale per ottimizzare la soglia.
        L'interpolazione non corregge gli scambi di ID. Le soglie di altezza restano quelle originali, non ottimizzate sul test.
        Con sette clip, discutere i risultati come studio esplorativo e descrivere distribuzione per clip/partita/inquadratura.
        ''','markdown')]
    save('03_NBA_correzione_valutazione.ipynb',cells03,backup);changed.append('03_NBA_correzione_valutazione.ipynb')
    (backup/'changes.json').write_text(json.dumps(dict(files=changed,root=str(ROOT)),indent=2),encoding='utf-8')
    print('Backup:',backup)


if __name__=='__main__':main()
