# Detection e tracking dei giocatori nelle partite di basket

**Tecniche di Computer Vision e Deep Learning per la detection e il tracking durante le partite di basket**

Tesi di Laurea Magistrale in Ingegneria Informatica – Curriculum Artificial Intelligence and Machine Learning
Autore: Marco · Relatore: Giuseppe Sansonetti · Università: Università Roma 3 ·

---

## Indice

1. [Obiettivo](#1-obiettivo)
2. [Pipeline](#2-pipeline)
3. [Risultati principali](#3-risultati-principali)
4. [Struttura della repository](#4-struttura-della-repository)
5. [Notebook](#5-notebook)
6. [Dati e modelli](#6-dati-e-modelli)
7. [Come riprodurre gli esperimenti](#7-come-riprodurre-gli-esperimenti)
8. [Problemi tecnici noti e soluzioni](#8-problemi-tecnici-noti-e-soluzioni)
9. [Limiti](#9-limiti)
10. [Stato del lavoro e sviluppi](#10-stato-del-lavoro-e-sviluppi)
11. [Riferimenti](#11-riferimenti)

---

## 1. Obiettivo

Partendo da normali riprese televisive (una sola telecamera in movimento), il progetto ricostruisce **chi** sono i giocatori, **dove** si trovano sul campo e **come** si muovono nel tempo, e confronta in modo quantitativo diverse tecniche per ogni passaggio:

- **detection** dei giocatori (CNN e transformer, preaddestrati e con fine-tuning);
- **tracking** multi-oggetto (tracking-by-detection classico e propagazione di maschere con SAM2);
- **localizzazione 2D** sul campo tramite keypoint e omografia, con correzione temporale delle coordinate;
- **applicazioni**: classificazione delle squadre, mappa 2D animata, heatmap, distanze e velocità.

Domande di ricerca:

| Domanda | Dove trova risposta |
|---|---|
| Serve il fine-tuning del detector? Quale architettura? | E1 |
| Quanto bene si mantengono le identità nel tempo? | E2 |
| Quali errori introduce SAM2 e come ridurli? | E2, E3 |
| Il post-processing migliora le traiettorie? | E2 (interpolazione), E3 (correttore) |
| Le coordinate supportano un’analisi tattica? | Demo (qualitativa) |

## 2. Pipeline

```
video broadcast NBA
   │
   ├─► detector giocatori (RT-DETR-L / YOLO11 fine-tuned)          ← E1
   │
   ├─► tracking: BoT-SORT, ByteTrack, OC-SORT, C-BIoU, SORT         ← E2
   │            oppure SAM2.1 (Hiera Large) + ri-detection periodica
   │   + interpolazione dei buchi (≤ 20 frame)
   │
   ├─► punto a terra (centro del bordo inferiore della box)
   │   + correzione temporale basata sull’altezza della box          ← E3
   │
   ├─► keypoint del campo (33 riferimenti) → omografia RANSAC per frame
   │
   └─► coordinate in piedi/metri sul campo NBA
         ├─ classificazione squadre (colore maglia, K-Means per traccia)
         ├─ mappa 2D animata affiancata al video
         ├─ heatmap per squadra
         └─ distanze e velocità
```

**Migliorie sperimentali introdotte nel progetto**

- **SAM2 + ri-detection**: ogni 5 frame il detector cerca giocatori non coperti da maschere (IoU < 0,3); se confermati in due controlli consecutivi, SAM2 viene reinizializzato mantenendo gli ID. Risolve il problema dei giocatori assenti al frame 0.
- **Interpolazione** lineare dei buchi nelle tracce (≤ 20 frame).
- **Attenzione a blocchi** per eseguire SAM2 su GPU pre-Ampere (T4) senza out-of-memory (risultato identico).
- **Pulizia delle tracce nella demo**: rimozione dei doppioni, scarto dei piedi coperti dal tabellone, divisione delle tracce che cambiano squadra, esclusione dell’arbitro.

## 3. Risultati principali

### E1 – Detection (test Roboflow v18 in-domain e SportsMOT out-of-domain)

| Modello | AP50:95 in-domain | AP50:95 SportsMOT | FPS (T4) |
|---|---|---|---|
| YOLO11n fine-tuned | 0,754 | 0,625 | 88,6 |
| YOLO11s fine-tuned | 0,796 | 0,641 | 79,9 |
| **YOLO11m fine-tuned** | **0,830** | 0,662 | 51,7 |
| **RT-DETR-L fine-tuned** | 0,823 | **0,708** | 14,6 |
| YOLO11s COCO (zero-shot) | 0,486 | 0,547 | 75,9 |

Il fine-tuning è indispensabile (+0,31 AP50:95). RT-DETR-L generalizza meglio sul dataset indipendente ed è il detector principale; YOLO11m è il migliore in-domain e 3,5× più veloce.

### E2 – Tracking (SportsMOT, sequenze di basket, split val)

**15 sequenze complete (12.557 frame), tracking-by-detection:**

| Detector + tracker | HOTA | MOTA | IDF1 | ID switch |
|---|---|---|---|---|
| **RT-DETR ft + BoT-SORT** | **0,539** | 0,898 | **0,571** | 641 |
| RT-DETR ft + C-BIoU | 0,521 | 0,894 | 0,558 | 720 |
| YOLO11m ft + BoT-SORT | 0,518 | 0,905 | 0,557 | 623 |
| YOLO11s COCO + BoT-SORT | 0,381 | 0,620 | 0,387 | 847 |

**15 sequenze × primi 300 frame, confronto con SAM2 (GPU L4):**

| Metodo | HOTA | AssA | MOTA | IDF1 | ID switch | FPS |
|---|---|---|---|---|---|---|
| **SAM2 + ri-detection** | **0,734** | 0,785 | 0,727 | **0,854** | 65 | ≈ 1,4 |
| SAM2 solo frame 0 | 0,713 | **0,796** | 0,672 | 0,827 | **54** | ≈ 1,6 |
| RT-DETR ft + BoT-SORT | 0,676 | 0,611 | 0,888 | 0,775 | 249 | – |
| YOLO11m ft + BoT-SORT | 0,654 | 0,574 | **0,904** | 0,766 | 244 | – |

- Il **detector conta più del tracker** (+0,16 HOTA tra COCO e fine-tuned a parità di tracker).
- **SAM2 mantiene molto meglio le identità** (circa 4 volte meno ID switch); i tracker classici localizzano meglio (MOTA più alta) e sono molto più veloci.
- La **ri-detection** migliora la baseline SAM2: +0,021 HOTA, +0,055 MOTA, +0,027 IDF1.
- L’interpolazione dà guadagni piccoli e consistenti, tranne con SORT.

### E3 – Coordinate sul campo (7 clip NBA proprie, 176 punti annotati a mano)

| Metodo (test: 4 clip, 91 punti) | Copertura | RMSE (px) |
|---|---|---|
| SAM2 solo frame 0 | 0,923 | 17,13 |
| **SAM2 + ri-detection** | **0,945** | **14,31** |
| SAM2 + ri-detection + correttore sull’altezza | 0,934 | 15,50 |

La ri-detection migliora anche la localizzazione (RMSE –16%); nella clip_01 recupera il portatore di palla mancante. Il correttore temporale originale **non** produce un miglioramento misurabile (risultato negativo documentato: si “blocca” sui cambi di inquadratura).

### Demo applicativa (clip_05, ≈ 10 s)

| | BoT-SORT | SAM2 + ri-detection |
|---|---|---|
| Tracce per 10 giocatori | 23 | **14** |
| Scambi tra avversari corretti | 9 | **2** |
| Coerenza squadre | 0,981 | 0,973 |
| Velocità max (mediana) | 5,1 m/s | 5,1 m/s |

Squadre classificate correttamente, arbitro escluso, heatmap e velocità plausibili e stabili rispetto al tracker.

## 4. Struttura della repository

```
Tesi Basket/
├── notebooks/                 notebook degli esperimenti (vedi sezione 5)
├── scripts/                   codice originale della pipeline baseline e strumenti
│   ├── height_corrector.py    correttore temporale basato sull’altezza
│   ├── court_evaluation.py    valutazione sulle annotazioni manuali
│   ├── thesis_tools.py        matching, metriche, utilità
│   ├── annotate.py            strumento di annotazione (avvio: Annota.cmd)
│   └── run_batch.py           pipeline baseline su più clip
├── annotations/               annotazioni manuali e manifest delle clip
│   ├── annotations.csv        216 etichette (176 punti a terra annotabili)
│   ├── tasks.csv              sequenze per le metriche di identità
│   ├── manifest*.csv          clip, split validation/test, percorsi delle tracce
│   └── history/               storico delle modifiche alle annotazioni
├── experiments/               risultati (CSV, figure) di E1, E2, E3
│   ├── E1_detection/
│   ├── E2_tracking/
│   └── E3_correzione/
├── configs/                   configurazione del progetto
├── tesi/                      materiale per la stesura
├── requirements.txt
└── README.md
```

Non sono nella repository (vedi `.gitignore`), ma su Google Drive in `MyDrive/Tesi_Basket`: video, modelli e pesi, dataset, output delle tracce (`outputs/`), video demo (`demo/`), run di training.

## 5. Notebook

| Notebook | Contenuto | Esecuzione |
|---|---|---|
| `00_setup_e_pulizia` | Inventario e riorganizzazione del progetto | Locale |
| `09_genera_tracce_SAM2_7clip` | Pipeline originale sulle 7 clip: SAM2 f0 e SAM2 + ri-detection, keypoint e omografie | Colab (T4/L4) |
| `10_E1_detection_confronto` | Fine-tuning e confronto di 5 detector, in-domain e su SportsMOT | Colab T4 |
| `11_E2_tracking_confronto` | 5 tracker × 3 detector + SAM2 su SportsMOT, metriche HOTA/MOTA/IDF1 | Colab T4 (tracker), L4 (SAM2) |
| `12_E3_correzione_coordinate` | Correttore temporale, selezione su validation, test, confronto f0/redet | Locale o Colab (CPU) |
| `13_demo_squadre_mappa2D` | Squadre, mappa 2D, heatmap, distanze; BoT-SORT vs SAM2 | Colab T4 |

Tutti i notebook usano la stessa cella di setup: su Colab il progetto è `MyDrive/Tesi_Basket`, in locale `F:\Tesi Basket`.

## 6. Dati e modelli

| Risorsa | Descrizione |
|---|---|
| **Basketball Player Detection v18** (Roboflow, `basketball-player-detection-3-ycjdo`) | 464 train / 96 valid / 94 test; 10 classi (5 classi giocatore unificate per la valutazione) |
| **SportsMOT** (split val, 15 sequenze di basket) | Box e ID su tutti i frame; scaricato con `trackers download sportsmot --split val` |
| **7 clip NBA proprie** | 854×480, ≈ 29 fps, 1.320 frame; split validation (clip 01, 03, 07) / test (04, 05, 06, 08) |
| **Annotazioni manuali** | 40 frame, 176 punti di contatto a terra, 3 sequenze consecutive per le identità |
| Detector baseline | YOLO11s Roboflow `basketball-player-detection-3-ycjdo/4` |
| Keypoint del campo | Roboflow `basketball-court-detection-2/14` |
| SAM2 | SAM2.1 Hiera Large, fork `Gy920/segment-anything-2-real-time` (commit `e3623cd`) |
| Campo | libreria `roboflow/sports` (branch `feat/basketball`), campo NBA in piedi |

## 7. Come riprodurre gli esperimenti

**Su Google Colab (consigliato)**

1. Copiare il progetto in `MyDrive/Tesi_Basket` (con `videos/`, `annotations/`, `models/`).
2. Aggiungere il Secret `ROBOFLOW_API_KEY` (icona a forma di chiave a sinistra) per dataset e modelli Roboflow.
3. Aprire il notebook, eseguire la **prima cella** (pip), poi **Runtime → Riavvia sessione**, poi eseguire dalla seconda cella.
4. Ordine consigliato: `10` → `09` → `11` → `12` → `13`. Le celle saltano automaticamente il lavoro già fatto (training, detection, tracce).

**In locale (notebook 12)**

```powershell
cd "F:\Tesi Basket"
.\.venv-eval\Scripts\activate
pip install -r requirements.txt
```
In VS Code selezionare il kernel `.venv-eval`.

**GPU**

- T4: sufficiente per detector, tracker e demo; SAM2 funziona grazie all’attenzione a blocchi, ma è lento.
- L4: consigliata per SAM2 (Flash Attention, picco di memoria ≈ 3 GB).
- Gli FPS vanno sempre riportati indicando la GPU.

## 8. Problemi tecnici noti e soluzioni

| Problema | Causa | Soluzione |
|---|---|---|
| Out-of-memory di SAM2 + ri-detection su T4 | Kernel di attenzione “math” (head dim 256) su GPU pre-Ampere | Attenzione calcolata a blocchi di query, oppure GPU L4 |
| Kernel che si chiude (Segmentation fault) nel notebook 13 | Import di `triton` incompatibile con PyTorch dopo i pip install | `sys.modules['triton'] = None` prima degli import; niente `inference` nel pip della demo |
| `ImportError` di Pillow (`_Ink`) su Colab | Pacchetti aggiornati in una sessione già avviata | pip → Riavvia sessione → seconda cella |
| Conflitto `inference-gpu` / `supervision` | Versioni incompatibili | `inference` + `supervision==0.30.9` |
| Array read-only con pandas 3 | Copy-on-write | Copie esplicite con `np.array` |
| Notebook 09 lento | Modelli Roboflow su CPU (onnxruntime) | Accettato; i tempi non sono usati come misura di velocità |

## 9. Limiti

- Campione proprio piccolo (7 clip, 1.320 frame); annotazioni a punti, quindi niente metriche MOT complete in E3.
- SAM2 valutato sui primi 300 frame di ogni sequenza SportsMOT; tracker con parametri di default.
- Omografia non validata con corrispondenze indipendenti (inlier ed errore di fit sono solo diagnostici).
- SAM2 + ri-detection può creare maschere doppie sullo stesso giocatore; BoT-SORT scambia gli avversari nelle occlusioni.
- Distanze e velocità della demo non validate: vanno lette come stime qualitative.

## 10. Stato del lavoro e sviluppi

| Attività | Stato |
|---|---|
| E1 – confronto detector | ✅ completato |
| E2 – confronto tracker + SAM2 (15 sequenze) | ✅ completato |
| E3 – correzione coordinate (f0 e redet) | ✅ completato |
| Demo squadre / mappa 2D / heatmap (BoT-SORT vs SAM2) | ✅ completata |
| Eventi di gioco sulla lavagna 2D: tiri e shot chart, possesso, passaggi, intercetti | ⏳ 1–9 novembre 2026 |
| Stesura della tesi (LaTeX) | ⏳ dal 10 novembre 2026 |

Sviluppi futuri: correttore temporale adattivo, controllo delle maschere doppie dopo la ri-detection, tracking affidabile della palla, riconoscimento delle azioni, OCR dei numeri di maglia.

## 11. Riferimenti

Metodi e strumenti principali (riferimenti completi nella bibliografia della tesi):

- Ultralytics YOLO11; RT-DETR (detector transformer real-time)
- SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU; libreria Roboflow `trackers`
- SAM2 – Segment Anything Model 2 (Meta AI)
- SportsMOT (dataset multi-object tracking sportivo)
- HOTA, CLEAR-MOT (MOTA), IDF1
- Roboflow `inference`, `supervision`, `sports`
- Omografia piana con RANSAC (OpenCV)

---

_Il codice è sviluppato per scopi di ricerca nell’ambito della tesi. I video delle partite NBA non sono inclusi nella repository._
