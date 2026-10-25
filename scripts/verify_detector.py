"""Smoke test reale: un solo frame per clip, nessuna riesecuzione del tracking."""
import json
from pathlib import Path
import time
import cv2
from PIL import Image,ImageDraw
from project_config import ROOT,load_config
from thesis_tools import new_run,fingerprint
from yolo_onnx import YoloOnnxAdapter


def main():
    c=load_config();out=new_run(ROOT,'detector_check')
    model=YoloOnnxAdapter(c['weights'])
    rows=[]; thumbs=[]
    for p in sorted(c['clips_dir'].glob('*.mov')):
        cap=cv2.VideoCapture(str(p));ok,frame=cap.read();cap.release()
        if not ok: raise RuntimeError(f'Primo frame non leggibile: {p}')
        started=time.perf_counter()
        preds=model.infer(frame,c['confidence'],c['nms_iou'])[0].predictions
        players=[d for d in preds if d.class_name in c['player_classes']]
        serial=[]
        for d in players:
            x1,y1=int(d.x-d.width/2),int(d.y-d.height/2)
            x2,y2=int(d.x+d.width/2),int(d.y+d.height/2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(frame,f'{d.class_name} {d.confidence:.2f}',(x1,max(14,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
            serial.append(vars(d))
        if not cv2.imwrite(str(out/f'{p.stem}_frame0.jpg'),frame): raise RuntimeError('Immagine non salvata')
        rows.append(dict(clip_id=p.stem,frame=0,player_detections=len(players),seconds=time.perf_counter()-started,predictions=serial))
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); image.thumbnail((427,240))
        panel=Image.new('RGB',(427,270),'white');panel.paste(image,(0,25))
        ImageDraw.Draw(panel).text((6,5),f'{p.stem}: {len(players)} detection giocatori (da verificare)',fill='black')
        thumbs.append(panel)
    sheet=Image.new('RGB',(427*2,270*4),'#eeeeee')
    for i,panel in enumerate(thumbs):sheet.paste(panel,((i%2)*427,(i//2)*270))
    sheet.save(out/'first_frames.jpg')
    evidence=dict(model='YOLO11s',weights_sha256=fingerprint(c['weights']),runtime='onnxruntime CPU',
                  preprocessing='RGB, stretch 1280x1280, /255; NMS per classe',metadata=model.metadata,clips=rows,
                  caveat='Verifica di caricamento e inferenza. Non certifica precisione/recall, equivalenza numerica Roboflow o presenza di tutti i giocatori.')
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(out,flush=True)
    print([(r['clip_id'],r['player_detections']) for r in rows],flush=True)


if __name__=='__main__':main()
