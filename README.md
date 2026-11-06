# Detection e tracking dei giocatori nelle partite di basket

**Tecniche di Computer Vision e Deep Learning per la detection e il tracking durante le partite di basket**

Tesi di Laurea Magistrale in Ingegneria Informatica – Curriculum Artificial Intelligence and Machine Learning
Autore: Marco · Relatore: Giuseppe Sansonetti · Università: Università Roma 3

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
- **applicazioni**: classificazione delle squadre, mappa 2D animata, heatmap, distanze e velocità;
- **eventi di gioco**: tiri, canestri, passaggi, rimbalzi, palle rubate e rimesse, riconosciuti automaticamente e disegnati sulla lavagna tattica 2D.

Domande di ricerca:

| Domanda | Dove trova risposta |
|---|---|
| Serve il fine-tuning del detector? Quale architettura? | E1 |
| Quanto bene si mantengono le identità nel tempo? | E2 |
| Quali errori introduce SAM2 e come ridurli? | E2, E3 |
| Il post-processing migliora le traiettorie? | E2 (interpolazione), E3 (correttore) |
| Le coordinate supportano un’analisi tattica? | Demo (qualitativa) |
| Un detector dedicato trova meglio la palla? | E4 |
| Si possono riconoscere gli eventi di gioco, e quanto bene? | Modulo eventi (sviluppo + test cieco) |

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
         ├─ distanze e velocità
         └─ eventi (notebook 15):
              tabellone (punteggio) + palla (detector E4 + filtro di Kalman)
              → possesso → regole → tiri, canestri, passaggi, rimbalzi,
                palle rubate, rimesse → lavagna 2D, linea del tempo, valutazione
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

Il fine-tuning è indispensabile (+0,31 AP50:95 a parità di architettura, YOLO11s). RT-DETR-L generalizza meglio sul dataset indipendente ed è il detector principale; YOLO11m è il migliore in-domain e 3,5× più veloce.

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

- Il **detector conta più del tracker**: a parità di tracker (BoT-SORT) i detector fine-tuned portano l’HOTA da 0,381 a 0,518–0,539. Cambiano insieme architettura, taglia e addestramento, quindi l’effetto del solo fine-tuning qui non è isolato.
- Solo BoT-SORT compensa il movimento della telecamera; tutti i tracker usano i parametri di default della libreria (SORT, per esempio, crea tracce da confidenza 0,25).
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

### E4 – Detector dedicato alla palla (notebook 16)

YOLO11s a una sola classe, 1280 px, addestrato su Roboflow v18 (solo la palla) + 312 frame annotati della gara 7 delle Finals 2016 (243 con la palla), presi solo dal primo 70% del video principale; test proprio di 107 frame dall'ultimo 30%.

| Detector | Test proprio AP50 | AP50:95 | Recall@0,25 | Test v18 AP50 | FPS |
|---|---|---|---|---|---|
| COCO «sports ball», 1280 | 0,095 | 0,066 | 0,024 | 0,038 | 40,6 |
| Vecchio multi-classe, 640 | 0,508 | 0,257 | 0,470 | 0,822 | 49,2 |
| Vecchio multi-classe, 1280 | 0,409 | 0,313 | 0,458 | 0,889 | 41,0 |
| **Nuovo, 1280** | **0,768** | **0,485** | **0,675** | 0,855 | 56,3 |

Il guadagno viene dai frame del dominio annotati, non dalla sola risoluzione.

### Eventi di gioco (notebook 15, clip NBA Finals 2016)

Abbinamento per tipo di evento e tempo (tolleranza 1 s) contro annotazioni manuali fatte prima di eseguire il modello.

| | Clip | Rilevati / reali / corretti | Precisione | Recall | F1 |
|---|---|---|---|---|---|
| Sviluppo, regole v3 | az1, az2, az4, az5, az6 | 29 / 40 / 26 | 0,90 | 0,65 | **0,75** |
| **Test cieco v3**,  | test1, test2 | 13 / 23 / 10 | 0,77 | 0,43 | **0,56** |
| **Test v4** | test1, test2 | 15 / 23 / 11 | 0,73 | 0,48 | 0,58 |


- Il numero di generalizzazione è quello della **v3**, congelata prima del test; la v4 è stata sviluppata dopo aver visto il test ed è riportata a parte.
- Nel test quasi tutti gli errori nascono prima delle regole: tagli di scena falsi, detection, attribuzione della palla.
- Indipendenza del test: il detector della palla è addestrato solo sul video principale (4° quarto), mentre le clip di test vengono dal secondo video (1° quarto).

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
│   └── eventi_manuali.csv     eventi annotati a mano sulle clip NBA (sviluppo e test)
├── experiments/               risultati (CSV, figure) di E1–E4
│   ├── E1_detection/
│   ├── E2_tracking/
│   ├── E3_correzione/         run f0 e run con ri-detection
│   └── E4_ball/
├── events/                    eventi sulla clip SportsMOT (notebook 14)
├── events_nba/                eventi per clip NBA: eventi automatici, valutazione,
│                              lavagne, linea del tempo, riepilogo; valutazione totale
├── configs/                   configurazione del progetto
├── tesi/                      materiale per la stesura
├── requirements.txt
└── README.md
```

Non sono nella repository (vedi `.gitignore`), ma su Google Drive in `MyDrive/Tesi_Basket`: video, modelli e pesi, dataset, output delle tracce (`outputs/`), video demo (`demo/`), run di training, video diagnostici e detection in cache degli eventi (`*.mp4`, `*.npy`), dati per il simulatore delle regole (`dati_*.zip`).

## 5. Notebook

| Notebook | Contenuto | Esecuzione |
|---|---|---|
| `00_setup_e_pulizia` | Inventario e riorganizzazione del progetto | Locale |
| `09_genera_tracce_SAM2_7clip` | Pipeline originale sulle 7 clip: SAM2 f0 e SAM2 + ri-detection, keypoint e omografie | Colab (T4/L4) |
| `10_E1_detection_confronto` | Fine-tuning e confronto di 5 detector, in-domain e su SportsMOT | Colab T4 |
| `11_E2_tracking_confronto` | 5 tracker × 3 detector + SAM2 su SportsMOT, metriche HOTA/MOTA/IDF1 | Colab T4 (tracker), L4 (SAM2) |
| `12_E3_correzione_coordinate` | Correttore temporale, selezione su validation, test, confronto f0/redet | Locale o Colab (CPU) |
| `13_demo_squadre_mappa2D` | Squadre, mappa 2D, heatmap, distanze; BoT-SORT vs SAM2 | Colab T4 |
| `14_eventi_lavagna2D` | Prima versione degli eventi su una clip SportsMOT (superata dal 15) | Colab T4 |
| `15_eventi_nba_v1_congelato` | Prima versione delle regole, congelata come riferimento | Colab T4 |
| `15_eventi_nba_v3` | Regole v3, congelate per il test cieco | Colab T4 |
| `15_eventi_nba_v4` | Revisione v4, successiva al test | Colab T4 |
| `15_eventi_nba_esecuzione/` | Un notebook eseguito per clip: az1–az6 con le regole v3, test1 e test2 con la revisione v4. L'esecuzione v3 del test cieco (F1 0,50) è riportata nella relazione; i file di `events_nba/finals2016_test*` corrispondono alla v4 | Colab T4 |
| `16_E4_detector_palla` | E4: dataset, addestramento e confronto del detector della palla | Colab L4 |
| `17_estrai_clip` | Panoramica del video, estrazione delle clip e controlli di qualità | Colab |

Tutti i notebook usano la stessa cella di setup: su Colab il progetto è `MyDrive/Tesi_Basket`, in locale `F:\Tesi Basket`.

## 6. Dati e modelli

| Risorsa | Descrizione |
|---|---|
| **Basketball Player Detection v18** (Roboflow, `basketball-player-detection-3-ycjdo`) | 464 train / 96 valid / 94 test; 10 classi (5 classi giocatore unificate per la valutazione) |
| **SportsMOT** (split val, 15 sequenze di basket) | Box e ID su tutti i frame; scaricato con `trackers download sportsmot --split val` |
| **7 clip NBA proprie** | 854×480, ≈ 29 fps, 1.320 frame; split validation (clip 01, 03, 07) / test (04, 05, 06, 08) |
| **Annotazioni manuali** | 40 frame, 176 punti di contatto a terra, 3 sequenze consecutive per le identità |
| **NBA Finals 2016, gara 7** | Video principale (4° quarto, 29,97 fps) e secondo video (1° quarto, 30 fps); clip di sviluppo az1–az6, clip di test test1 e test2; non inclusi |
| **Eventi annotati** | 8 clip, 66 eventi (`annotations/eventi_manuali.csv`): 40 nelle 5 clip di sviluppo valutate, 23 nelle 2 clip di test, 3 in az3 (esclusa dalla valutazione quantitativa: oltre l'80% di non-gioco) |
| **Frame propri per la palla** | 419 frame del video principale annotati su Roboflow (E4): 312 di addestramento, 107 di test |
| Detector baseline | YOLO11s Roboflow `basketball-player-detection-3-ycjdo/4` |
| Keypoint del campo | Roboflow `basketball-court-detection-2/14` |
| SAM2 | SAM2.1 Hiera Large, fork `Gy920/segment-anything-2-real-time` (commit `e3623cd`) |
| Campo | libreria `roboflow/sports` (branch `feat/basketball`), campo NBA in piedi |

## 7. Come riprodurre gli esperimenti

**Su Google Colab (consigliato)**

1. Copiare il progetto in `MyDrive/Tesi_Basket` (con `videos/`, `annotations/`, `models/`).
2. Aggiungere il Secret `ROBOFLOW_API_KEY` (icona a forma di chiave a sinistra) per dataset e modelli Roboflow.
3. Aprire il notebook, eseguire la **prima cella** (pip), poi **Runtime → Riavvia sessione**, poi eseguire dalla seconda cella.
4. Ordine consigliato: `10` → `09` → `11` → `12` → `13` → `16` → `17` → `15`. Le celle saltano automaticamente il lavoro già fatto (training, detection, tracce).

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
| Palla trovata in pochi frame e scambiata con altro | Detector multi-classe a 640 px, palla di 5–10 px | Detector dedicato a 1280 px (E4) + filtro di Kalman |
| Falsi cambi di punteggio | Dissolvenze e deriva della grafica del tabellone | Riferimento a maggioranza su 1,5 s |
| Falsi tagli di scena | Flash dei fotografi, movimenti rapidi della telecamera | Limite documentato (11 tagli falsi su 13 in az6); sviluppo futuro |
| ID diversi tra esecuzioni | Contatore degli ID che sopravvive nella sessione | Riavvio della sessione prima di ogni esecuzione completa |

## 9. Limiti

- Campione proprio piccolo (7 clip, 1.320 frame); annotazioni a punti, quindi niente metriche MOT complete in E3.
- SAM2 valutato sui primi 300 frame di ogni sequenza SportsMOT; tracker con parametri di default.
- SAM2 + ri-detection può creare maschere doppie sullo stesso giocatore; BoT-SORT scambia gli avversari nelle occlusioni.
- Distanze e velocità della demo non validate: vanno lette come stime qualitative.
- Eventi valutati su poche clip di una sola partita (63 eventi valutati, 23 nel test), annotati da una sola persona: i risultati sugli eventi sono preliminari, uno studio di caso.
- Distanze, velocità e heatmap sono dimostrazioni applicative: l'omografia non è validata con misure indipendenti, quindi non sono misure fisiche certificate.
- Il tabellone è letto con coordinate fisse per la grafica ESPN di questa partita.
- La provenienza del dataset Roboflow v18 non è documentata.

## 10. Stato del lavoro e sviluppi

| Attività | Stato |
|---|---|
| E1 – confronto detector | ✅ completato |
| E2 – confronto tracker + SAM2 (15 sequenze) | ✅ completato |
| E3 – correzione coordinate (f0 e redet) | ✅ completato |
| Demo squadre / mappa 2D / heatmap (BoT-SORT vs SAM2) | ✅ completata |
| E4 – detector della palla | ✅ completato |
| Eventi di gioco: regole v1 → v4, valutazione di sviluppo e test cieco | ✅ completato |
| Stesura della tesi (LaTeX) | ⏳ |

Sviluppi futuri: rilevatore di tagli di scena robusto a flash e movimenti di camera, correttore temporale adattivo, controllo delle maschere doppie dopo la ri-detection, tracking affidabile della palla, riconoscimento delle azioni, OCR dei numeri di maglia.

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
