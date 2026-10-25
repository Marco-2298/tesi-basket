"""Adapter del backup YOLO11s: preprocessing descritto dai suoi metadati.

CPU ONNX Runtime; nessuna installazione Inference/PyCUDA o download di pesi.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np


def nms_indices(boxes, scores, threshold):
    order=np.argsort(-scores,kind='stable'); kept=[]
    while len(order):
        i=int(order[0]); kept.append(i); rest=order[1:]
        if not len(rest): break
        wh=np.maximum(0,np.minimum(boxes[i,2:],boxes[rest,2:])-np.maximum(boxes[i,:2],boxes[rest,:2]))
        inter=wh.prod(axis=1)
        area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(axis=1)
        iou=inter/np.maximum(area[i]+area[rest]-inter,1e-12)
        order=rest[iou<=threshold]
    return kept


class YoloOnnxAdapter:
    def __init__(self, weights):
        import onnxruntime as ort
        weights=Path(weights)
        if not weights.is_file(): raise FileNotFoundError(weights)
        config=json.loads((weights.parent/'inference_config.json').read_text())
        self.class_names=(weights.parent/'class_names.txt').read_text().splitlines()
        network=config['network_input']
        if network['resize_mode']!='stretch' or network['color_mode']!='rgb' or network['scaling_factor']!=255:
            raise ValueError('Preprocessing non supportato: verificare inference_config.json')
        self.size=(network['training_input_size']['width'],network['training_input_size']['height'])
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(weights),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name
        self.metadata=self.session.get_modelmeta().custom_metadata_map
        if 'YOLO11s' not in self.metadata.get('description',''):
            raise ValueError('Il backup atteso è YOLO11s: controllare i metadati ONNX')
        names=ast.literal_eval(self.metadata['names'])
        if [names[i] for i in range(len(names))]!=self.class_names:
            raise ValueError('Classi ONNX diverse da class_names.txt')

    def infer(self,image,confidence=0.4,iou_threshold=0.7):
        if image is None or image.ndim!=3: raise ValueError('Immagine BGR non valida')
        if not 0<confidence<=1 or not 0<=iou_threshold<=1: raise ValueError('Soglie fuori intervallo')
        height,width=image.shape[:2]
        resized=cv2.resize(image,self.size)
        tensor=np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        output=self.session.run(None,{self.input_name:tensor})[0]
        expected=4+len(self.class_names)
        if output.ndim!=3 or output.shape[0]!=1 or output.shape[1]!=expected:
            raise ValueError(f'Output YOLO non atteso: {output.shape}, canali attesi {expected}')
        rows=output[0].T
        scores=rows[:,4:].max(axis=1); ids=rows[:,4:].argmax(axis=1)
        select=scores>=confidence; rows=rows[select];scores=scores[select];ids=ids[select]
        xy=rows[:,:4]
        boxes=np.column_stack([xy[:,:2]-xy[:,2:]/2,xy[:,:2]+xy[:,2:]/2])
        boxes[:,[0,2]]*=width/self.size[0];boxes[:,[1,3]]*=height/self.size[1]
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
        predictions=[]
        # NMS per classe, poi unificazione e NMS dei soli giocatori nei notebook.
        for cls in sorted(set(ids)):
            subset=np.flatnonzero(ids==cls)
            for index in nms_indices(boxes[subset],scores[subset],iou_threshold):
                i=subset[index];x1,y1,x2,y2=map(float,boxes[i])
                if x2<=x1 or y2<=y1: continue
                predictions.append(SimpleNamespace(x=(x1+x2)/2,y=(y1+y2)/2,width=x2-x1,height=y2-y1,
                                                   confidence=float(scores[i]),class_name=self.class_names[int(cls)]))
        return [SimpleNamespace(predictions=predictions)]
