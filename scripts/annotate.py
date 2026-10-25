"""Annotatore locale Tkinter: punti manuali su frame originali, nessuna rete."""
import json
import tkinter as tk
from tkinter import ttk,messagebox,simpledialog
import csv
import shutil
from datetime import datetime,timezone
import cv2
from PIL import Image,ImageTk
from project_config import ROOT,load_config
from annotation_store import read_rows,upsert,VISIBILITY,validate_dataset,task_complete,required_players


class Annotator:
    def __init__(self,window):
        self.window=window;self.config=load_config()
        self.tasks=read_rows(self.config['tasks']);self.manifest=read_rows(self.config['manifest'])
        self.clips={r['clip_id']:r for r in self.manifest}
        self.progress_path=ROOT/'annotations/reviewed_frames.json'
        self.reviewed=set(json.loads(self.progress_path.read_text())) if self.progress_path.exists() else set()
        existing=read_rows(self.config['annotations'])
        self.index=next((i for i,t in enumerate(self.tasks) if not task_complete(t,existing)),len(self.tasks)-1)
        self.point=None;self.window.title('Tesi Basket — annotazioni manuali')
        self.title=ttk.Label(window,font=('Segoe UI',12));self.title.pack(padx=12,pady=8)
        self.canvas=tk.Canvas(window,width=854,height=480,bg='black');self.canvas.pack(padx=12)
        self.canvas.bind('<Button-1>',self.click)
        controls=ttk.Frame(window);controls.pack(fill='x',padx=12,pady=8)
        ttk.Label(controls,text='Identità umana:').grid(row=0,column=0)
        self.player=tk.StringVar(value='p01');self.player_box=ttk.Combobox(controls,textvariable=self.player,width=12)
        self.player_box.grid(row=0,column=1,padx=6)
        ttk.Label(controls,text='Visibilità:').grid(row=0,column=2)
        self.visibility=tk.StringVar(value='visible')
        ttk.Combobox(controls,textvariable=self.visibility,values=VISIBILITY,state='readonly',width=14).grid(row=0,column=3,padx=6)
        self.annotatable=tk.BooleanVar(value=True)
        ttk.Checkbutton(controls,text='Punto a terra annotabile',variable=self.annotatable,command=self.toggle_point).grid(row=0,column=4)
        ttk.Label(controls,text='Nota:').grid(row=1,column=0,pady=6)
        self.note=tk.StringVar();ttk.Entry(controls,textvariable=self.note,width=90).grid(row=1,column=1,columnspan=4,sticky='ew')
        buttons=ttk.Frame(window);buttons.pack(pady=5)
        ttk.Button(buttons,text='← Frame precedente',command=self.previous).pack(side='left',padx=5)
        ttk.Button(buttons,text='Salva giocatore',command=self.save).pack(side='left',padx=5)
        ttk.Button(buttons,text='Frame completato →',command=self.next).pack(side='left',padx=5)
        ttk.Button(buttons,text='Controlla completezza',command=self.check).pack(side='left',padx=5)
        ttk.Button(buttons,text='Scegli i 2 ID della sequenza',command=self.choose_players).pack(side='left',padx=5)
        self.status=ttk.Label(window,wraplength=900);self.status.pack(padx=12,pady=8)
        ttk.Label(window,text='Piano ridotto: SOLO 5 giocatori nei frame sparsi, SOLO 2 nelle sequenze di 8 frame.\nConserviamo tutte le etichette già inserite. Identità umane coerenti; non inventare coordinate se non annotabili.',wraplength=900).pack(padx=12,pady=8)
        self.load()

    @staticmethod
    def task_key(t):return f'{t["clip_id"]}:{t["frame"]}'

    def load(self):
        self.task=self.tasks[self.index];c=self.clips[self.task['clip_id']]
        cap=cv2.VideoCapture(str(ROOT/c['video']));cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.task['frame']))
        ok,frame=cap.read();cap.release()
        if not ok:raise RuntimeError('Frame non leggibile')
        self.height,self.width=frame.shape[:2];self.scale=min(1100/self.width,600/self.height,1.)
        self.display_w=round(self.width*self.scale);self.display_h=round(self.height*self.scale)
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((self.display_w,self.display_h))
        self.photo=ImageTk.PhotoImage(image)
        self.canvas.config(width=self.display_w,height=self.display_h)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.point=None;self.note.set('')
        rows=[r for r in read_rows(self.config['annotations']) if r['clip_id']==self.task['clip_id']]
        self.player_box['values']=sorted({r['player_id'] for r in rows})
        current=[r for r in rows if int(r['frame'])==int(self.task['frame'])]
        for r in current:
            if r['annotatable']=='1':
                x=float(r['x_px'])*self.scale;y=float(r['y_px'])*self.scale
                self.canvas.create_oval(x-4,y-4,x+4,y+4,outline='#00ffff',width=2)
                self.canvas.create_text(x+8,y-8,text=r['player_id'],fill='#00ffff',anchor='w')
        purpose=f"sequenza {self.task['sequence_id']}" if self.task['sequence_id'] else 'punto sparso'
        self.title.config(text=f"{self.index+1}/{len(self.tasks)} — {self.task['clip_id']} — frame {self.task['frame']} — {purpose}")
        required=required_players(self.task)
        goal=f"SOLO gli ID {', '.join(required)}" if required else ('Scegli prima i 2 ID della sequenza' if self.task['sequence_id'] else f"SOLO {self.task.get('target_count',5)} giocatori")
        all_rows=read_rows(self.config['annotations'])
        complete=sum(task_complete(t,all_rows) for t in self.tasks)
        self.status.config(text=f"Obiettivo: {goal}. Salvati: {', '.join(r['player_id'] for r in current) or 'nessuno'}. Piano: {complete}/{len(self.tasks)} frame completi.\n{self.task.get('instruction','')}")

    def choose_players(self):
        if not self.task['sequence_id']:
            messagebox.showinfo('Frame sparso','Qui bastano 5 giocatori; non serve scegliere una coppia stabile.');return
        value=simpledialog.askstring('Due identità stabili','Inserisci SOLO 2 ID separati da virgola (es. p01,p05).\nDevono indicare le stesse due persone negli 8 frame.\nSe hai già etichettato questo frame, riusa i loro ID.',initialvalue=','.join(required_players(self.task)))
        if value is None:return
        players=[x.strip() for x in value.split(',') if x.strip()]
        if len(players)!=2 or len(set(players))!=2:
            messagebox.showerror('Coppia non valida','Inserire esattamente due ID distinti.');return
        for t in self.tasks:
            if t['clip_id']==self.task['clip_id'] and t['sequence_id']==self.task['sequence_id']:
                t['required_player_ids']=';'.join(players)
        path=self.config['tasks'];history=path.parent/'history';history.mkdir(exist_ok=True)
        shutil.copy2(path,history/('tasks_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.csv'))
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.tasks[0]));w.writeheader();w.writerows(self.tasks)
        self.player.set(players[0]);self.load()

    def click(self,event):
        if not (0<=event.x<self.display_w and 0<=event.y<self.display_h):return
        self.annotatable.set(True);self.point=(event.x/self.scale,event.y/self.scale)
        self.canvas.delete('candidate');x,y=event.x,event.y
        self.canvas.create_line(x-8,y,x+8,y,fill='#ff5555',width=2,tags='candidate')
        self.canvas.create_line(x,y-8,x,y+8,fill='#ff5555',width=2,tags='candidate')
        self.status.config(text=f'Candidato: x={self.point[0]:.1f}, y={self.point[1]:.1f}. Premi Salva giocatore.')

    def toggle_point(self):
        if not self.annotatable.get():self.point=None;self.canvas.delete('candidate')

    def save(self):
        if self.task['sequence_id'] and self.player.get().strip() not in required_players(self.task):
            messagebox.showwarning('Solo 2 giocatori','Scegli prima la coppia della sequenza e salva soltanto quei due ID.');return
        if self.annotatable.get() and self.point is None:
            messagebox.showerror('Punto mancante','Clicca il punto a terra oppure disattiva annotabile.');return
        x,y=self.point if self.annotatable.get() else ('','')
        row=dict(clip_id=self.task['clip_id'],frame=int(self.task['frame']),player_id=self.player.get().strip(),
                 visibility=self.visibility.get(),annotatable=int(self.annotatable.get()),x_px=x,y_px=y,
                 sequence_id=self.task['sequence_id'],note=self.note.get())
        try:upsert(self.config['annotations'],row,self.width,self.height)
        except ValueError as e:messagebox.showerror('Etichetta non valida',str(e));return
        self.reviewed.discard(self.task_key(self.task));self.load()

    def previous(self):self.index=max(0,self.index-1);self.load()

    def next(self):
        if self.point is not None:
            messagebox.showwarning('Punto non salvato','Premi Salva giocatore prima di continuare.');return
        current=[r for r in read_rows(self.config['annotations']) if r['clip_id']==self.task['clip_id'] and int(r['frame'])==int(self.task['frame'])]
        if not task_complete(self.task,read_rows(self.config['annotations'])):
            messagebox.showwarning('Obiettivo del frame', 'Nelle sequenze scegli e salva soltanto i 2 ID richiesti; nei frame sparsi bastano 5 etichette, anche non annotabili.');return
        self.reviewed.add(self.task_key(self.task))
        self.progress_path.write_text(json.dumps(sorted(self.reviewed),indent=2),encoding='utf-8')
        rows=read_rows(self.config['annotations'])
        pending=[i for i,t in enumerate(self.tasks) if not task_complete(t,rows)]
        if not pending:
            messagebox.showinfo('Piano concluso','Annotazioni richieste completate. Non servono altri giocatori. Restano gli eventuali metadati e lo split.');return
        self.index=next((i for i in pending if i>self.index),pending[0]);self.load()

    def check(self):
        errors=validate_dataset(read_rows(self.config['manifest']),self.tasks,read_rows(self.config['annotations']))
        if errors:messagebox.showinfo('Attività rimanenti','\n'.join(errors[:15]))
        else:messagebox.showinfo('Struttura completa','Annotazioni strutturalmente complete. Controllare manualmente identità, split e qualità.')


if __name__=='__main__':
    window=tk.Tk();Annotator(window);window.mainloop()
