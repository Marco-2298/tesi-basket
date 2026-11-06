# Analisi automatica di partite di basket con Computer Vision

Progetto di tesi magistrale in Ingegneria Informatica (curriculum Artificial Intelligence and Machine Learning), Università Roma Tre.

**Autore:** Marco D'Albis

Questo repository raccoglie gli esperimenti svolti per analizzare partite NBA a partire da riprese televisive. Ho lavorato sul rilevamento e sul tracking dei giocatori, sulla proiezione delle loro posizioni su un campo 2D e su un primo sistema di riconoscimento degli eventi di gioco.

L'utilizzo di video broadcast rende il problema meno controllato rispetto a un sistema con telecamere fisse: l'inquadratura si muove, alcuni giocatori escono dal campo visivo, ci sono occlusioni e la palla è spesso piccola o poco visibile. Il progetto confronta diversi approcci a queste difficoltà e ne documenta anche i limiti.

## Contenuti

- [Pipeline](#pipeline)
- [Dati e modelli](#dati-e-modelli)
- [Esperimenti e risultati](#esperimenti-e-risultati)
- [Struttura del repository](#struttura-del-repository)
- [Notebook](#notebook)
- [Esecuzione](#esecuzione)
- [Problemi noti e limiti](#problemi-noti-e-limiti)
- [Riferimenti](#riferimenti)

## Pipeline

La pipeline parte dal video e produce tracce dei giocatori e coordinate sul campo. Da queste informazioni ricavo le visualizzazioni 2D; per gli eventi di gioco utilizzo anche il rilevamento della palla e la lettura del punteggio.

```text
Video NBA
  |
  +-- Detection giocatori (YOLO11 / RT-DETR)
  |      |
  |      +-- Tracking-by-detection (BoT-SORT, ByteTrack, OC-SORT, C-BIoU, SORT)
  |      +-- Segmentazione e tracking con SAM2.1
  |
  +-- Keypoint del campo -> omografia con RANSAC
  |
  +-- Posizione dei piedi -> coordinate sul campo NBA
         |
         +-- Classificazione delle squadre
         +-- Mappa 2D, traiettorie, heatmap, distanze e velocità
         +-- Eventi di gioco (palla + punteggio + regole temporali)
```

Per SAM2 ho confrontato l'inizializzazione al primo frame con una variante che effettua periodicamente una nuova detection: ogni 5 frame vengono cercati giocatori non coperti dalle maschere esistenti; due verifiche consecutive permettono di aggiungerli al tracking. Ho inoltre provato l'interpolazione dei tratti mancanti delle tracce (fino a 20 frame).

La posizione a terra viene stimata dal bordo inferiore della bounding box e proiettata sul campo mediante omografia. Ho testato anche un correttore temporale basato sull'altezza della box, senza ottenere un miglioramento sul test.

## Dati e modelli

| Risorsa | Utilizzo |
|---|---|
| Basketball Player Detection v18 (Roboflow, `basketball-player-detection-3-ycjdo`) | Fine-tuning e valutazione dei detector; 464 immagini di training, 96 di validation e 94 di test |
| SportsMOT, split validation | Confronto dei tracker su 15 sequenze di basket, per un totale di 12.557 frame |
| 7 clip NBA | Test sulle coordinate e demo; 1.320 frame a 854×480, circa 29 FPS |
| Annotazioni manuali delle posizioni | 40 frame e 176 punti di contatto a terra utilizzati nella valutazione |
| NBA Finals 2016, gara 7 | Clip di sviluppo (`az1`–`az6`) e test (`test1`, `test2`) per gli eventi |
| Annotazioni degli eventi | `annotations/eventi_manuali.csv`; conteggi e metriche da riallineare dopo l'ultima modifica al CSV |
| Circa 200 frame annotati della gara 7 | Integrazione dei dati per il detector dedicato alla palla |

Modelli e strumenti principali:

- YOLO11 e RT-DETR-L per la detection dei giocatori;
- BoT-SORT, ByteTrack, OC-SORT, C-BIoU e SORT per il tracking-by-detection;
- SAM2.1 Hiera Large per segmentazione e propagazione delle maschere;
- YOLO11s a una classe e risoluzione 1280 per la detection della palla;
- modello Roboflow `basketball-court-detection-2/14` per i keypoint del campo;
- libreria `roboflow/sports` per la rappresentazione del campo NBA.

I video e i pesi più grandi non sono inclusi nel repository. La loro posizione originale nel progetto è indicata più avanti.

## Esperimenti e risultati

Le tabelle riportano i risultati salvati nella versione precedente del progetto. **Le metriche degli eventi, in particolare quelle di `test1`, sono provvisorie:** dopo la modifica di `eventi_manuali.csv` vanno ricalcolate e sostituite qui. Non sono stati inseriti valori ipotetici.

### E1 — Detection dei giocatori

Ho confrontato modelli YOLO11 di diverse dimensioni e RT-DETR-L, con una valutazione sul test Roboflow v18 e su SportsMOT, usato come insieme esterno.

| Modello | AP50:95 Roboflow v18 | AP50:95 SportsMOT | FPS (T4) |
|---|---:|---:|---:|
| YOLO11n fine-tuned | 0,754 | 0,625 | 88,6 |
| YOLO11s fine-tuned | 0,796 | 0,641 | 79,9 |
| YOLO11m fine-tuned | **0,830** | 0,662 | 51,7 |
| RT-DETR-L fine-tuned | 0,823 | **0,708** | 14,6 |
| YOLO11s COCO, senza fine-tuning | 0,486 | 0,547 | 75,9 |

Il fine-tuning di YOLO11s porta l'AP50:95 su Roboflow v18 da 0,486 a 0,796. YOLO11m ottiene il risultato migliore sul test interno; RT-DETR-L è invece il più accurato su SportsMOT, a fronte di un'inferenza più lenta. Per i confronti successivi ho quindi considerato sia accuratezza sia costo computazionale.

### E2 — Tracking

Il primo confronto riguarda il tracking-by-detection su 15 sequenze complete di SportsMOT (12.557 frame).

| Detector e tracker | HOTA | MOTA | IDF1 | ID switch |
|---|---:|---:|---:|---:|
| RT-DETR ft + BoT-SORT | **0,539** | 0,898 | **0,571** | 641 |
| RT-DETR ft + C-BIoU | 0,521 | 0,894 | 0,558 | 720 |
| YOLO11m ft + BoT-SORT | 0,518 | **0,905** | 0,557 | 623 |
| YOLO11s COCO + BoT-SORT | 0,381 | 0,620 | 0,387 | 847 |

Ho poi confrontato SAM2 con alcuni tracker tradizionali sui primi 300 frame delle stesse 15 sequenze. Il confronto è quindi su una porzione più corta e non va letto come se fosse svolto sulle sequenze complete.

| Metodo | HOTA | AssA | MOTA | IDF1 | ID switch | FPS |
|---|---:|---:|---:|---:|---:|---:|
| SAM2 + ri-detection | **0,734** | 0,785 | 0,727 | **0,854** | 65 | ≈ 1,4 |
| SAM2 solo frame iniziale | 0,713 | **0,796** | 0,672 | 0,827 | **54** | ≈ 1,6 |
| RT-DETR ft + BoT-SORT | 0,676 | 0,611 | 0,888 | 0,775 | 249 | — |
| YOLO11m ft + BoT-SORT | 0,654 | 0,574 | **0,904** | 0,766 | 244 | — |

SAM2 ottiene una migliore continuità delle identità, ma è sensibilmente più lento. L'aggiunta della ri-detection migliora copertura e IDF1 rispetto all'inizializzazione limitata al primo frame, aumentando però leggermente il numero di ID switch. I tracker classici rimangono più pratici per un'elaborazione veloce. I valori FPS di SAM2 si riferiscono alla GPU L4; quelli di E1 alla T4.

### E3 — Coordinate sul campo

Ho valutato la stima dei punti a terra sulle clip NBA annotate manualmente. Il test comprende 4 clip e 91 punti.

| Metodo | Copertura | RMSE (pixel) |
|---|---:|---:|
| SAM2 solo frame iniziale | 0,923 | 17,13 |
| SAM2 + ri-detection | **0,945** | **14,31** |
| SAM2 + ri-detection + correttore temporale | 0,934 | 15,50 |

La ri-detection riduce l'errore rispetto alla versione inizializzata solo al primo frame e recupera alcuni giocatori non presenti all'inizio delle clip. Il correttore temporale, invece, non ha migliorato il risultato sul test: in alcuni cambi d'inquadratura ha reso le coordinate meno affidabili.

Nella demo su `clip_05` (circa 10 secondi) ho confrontato anche la frammentazione delle tracce:

| Misura | BoT-SORT | SAM2 + ri-detection |
|---|---:|---:|
| Tracce associate ai 10 giocatori | 23 | 14 |
| Scambi fra avversari corretti manualmente | 9 | 2 |
| Coerenza della classificazione squadra | 0,981 | 0,973 |

La demo include mappa 2D animata, heatmap, distanze e velocità. Queste ultime sono indicative: non sono state validate rispetto a misure metriche indipendenti.

### E4 — Detection della palla

Per la palla ho addestrato un YOLO11s a classe singola, con input a 1280 pixel, utilizzando il dataset Roboflow v18 e circa 200 frame annotati della gara 7 delle Finals 2016. I frame aggiunti provengono dalla parte iniziale del video principale e non dalle clip di test degli eventi.

| Detector | AP50 test proprio | AP50:95 | Recall@0,25 | AP50 v18 | FPS |
|---|---:|---:|---:|---:|---:|
| COCO `sports ball`, 1280 | 0,095 | 0,066 | 0,024 | 0,038 | 40,6 |
| Precedente multi-classe, 640 | 0,508 | 0,257 | 0,470 | 0,822 | 49,2 |
| Precedente multi-classe, 1280 | 0,409 | 0,313 | 0,458 | 0,889 | 41,0 |
| YOLO11s dedicato, 1280 | **0,768** | **0,485** | **0,675** | 0,855 | 56,3 |

Il modello dedicato migliora soprattutto sul test ricavato dal video NBA. Il confronto tra le due risoluzioni del vecchio modello mostra che il vantaggio non dipende semplicemente dall'aumento dei pixel in ingresso.

### Riconoscimento degli eventi

L'ultimo modulo combina detection della palla, filtro di Kalman, informazioni sul possesso, variazioni del punteggio e regole temporali per rilevare tiri, canestri, passaggi, rimbalzi, palle rubate e rimesse.

Gli eventi automatici vengono confrontati con un CSV annotato manualmente, tramite abbinamento per tipo di evento e tempo (tolleranza di 1 secondo). Ho distinto una versione delle regole congelata prima del test (`v3`) da una revisione successiva (`v4`): i risultati della seconda non vanno presentati come valutazione cieca.

**Metriche della versione precedente, da aggiornare dopo il nuovo run di `test1`:**

| Valutazione | Clip | Rilevati / reali / corretti | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| Sviluppo, v3 | az1, az2, az4, az5, az6 | 29 / 40 / 26 | 0,90 | 0,65 | 0,75 |
| Test cieco, v3 | test1, test2 | 13 / 23 / 9 | 0,69 | 0,39 | 0,50 |
| Test dopo revisione, v4 | test1, test2 | 15 / 23 / 10 | 0,67 | 0,43 | 0,53 |

La valutazione più severa richiede, oltre al tipo e al tempo, anche la corretta attribuzione di squadra e giocatori. Nel vecchio test v3 erano corretti 7 eventi su 23 (F1 pari a 0,39); anche questo valore deve essere ricalcolato. Le principali difficoltà osservate riguardano falsi tagli di scena, errori nel rilevamento della palla e attribuzione del possesso.

## Struttura del repository

```text
Tesi Basket/
├── notebooks/                 esperimenti e notebook per le singole clip
├── scripts/                   script della pipeline e strumenti di valutazione
│   ├── height_corrector.py
│   ├── court_evaluation.py
│   ├── thesis_tools.py
│   ├── annotate.py
│   └── run_batch.py
├── annotations/               annotazioni manuali e manifest
│   ├── annotations.csv
│   ├── tasks.csv
│   ├── manifest*.csv
│   └── eventi_manuali.csv
├── experiments/               CSV e figure degli esperimenti E1–E4
│   ├── E1_detection/
│   ├── E2_tracking/
│   ├── E3_correzione/
│   └── E4_ball/
├── events/                    prime prove degli eventi su SportsMOT
├── events_nba/                eventi, valutazioni e visualizzazioni NBA
├── configs/
├── tesi/
├── requirements.txt
└── README.md
```

Nella configurazione usata durante lo sviluppo, i video, i pesi dei modelli, alcuni dataset e gli output più pesanti si trovano su Google Drive in `MyDrive/Tesi_Basket`. Non sono versionati nel repository (vedere `.gitignore`). Tra i file esterni rientrano anche le tracce, i video della demo e alcune cache della detection degli eventi.

## Notebook

| Notebook | Cosa contiene | Ambiente utilizzato |
|---|---|---|
| `00_setup_e_pulizia` | Controllo e organizzazione dei file del progetto | Locale |
| `09_genera_tracce_SAM2_7clip` | Tracce SAM2 sulle clip NBA, con e senza ri-detection; keypoint e omografie | Colab T4/L4 |
| `10_E1_detection_confronto` | Addestramento e confronto dei detector | Colab T4 |
| `11_E2_tracking_confronto` | Tracking-by-detection e confronto con SAM2 su SportsMOT | Colab T4/L4 |
| `12_E3_correzione_coordinate` | Valutazione della posizione a terra e correttore temporale | Locale o Colab CPU |
| `13_demo_squadre_mappa2D` | Classificazione squadre, mappa 2D, heatmap, distanze | Colab T4 |
| `14_eventi_lavagna2D` | Prima prova degli eventi su SportsMOT | Colab T4 |
| `15_eventi_nba_v1_congelato` | Prima versione di riferimento delle regole | Colab T4 |
| `15_eventi_nba_v3` | Regole utilizzate per il test cieco | Colab T4 |
| `15_eventi_nba_v4` | Revisione successiva al test | Colab T4 |
| `15_eventi_nba_esecuzione/` | Esecuzioni separate per le clip di sviluppo e test | Colab T4 |
| `16_E4_detector_palla` | Dataset, training e valutazione del detector dedicato alla palla | Colab L4 |
| `17_estrai_clip` | Estrazione delle clip e controlli sul materiale video | Colab |

I notebook 15 consumano dati e risultati prodotti nelle fasi precedenti. Il notebook 16 serve per generare il detector dedicato alla palla; una volta disponibili i pesi, non occorre riaddestrarlo per ogni nuova valutazione degli eventi.

## Esecuzione

### Google Colab

1. Preparare una cartella `MyDrive/Tesi_Basket` con il progetto e le risorse esterne necessarie (`videos/`, `annotations/`, `models/` e gli eventuali output intermedi richiesti dal notebook).
2. Impostare il secret `ROBOFLOW_API_KEY` per accedere alle risorse Roboflow.
3. Aprire il notebook di interesse ed eseguire la cella di installazione delle dipendenze.
4. Se richiesto dagli aggiornamenti dei pacchetti, riavviare la sessione e proseguire con le celle successive.
5. Controllare i percorsi iniziali: i notebook sono stati sviluppati con il progetto in `MyDrive/Tesi_Basket` su Colab e in `F:\Tesi Basket` in locale. Su altre macchine vanno adattati.

Per rifare gli esperimenti dalla preparazione dei modelli, un ordine di riferimento è:

```text
10 (detection) -> 09 (tracce NBA) -> 11 (tracking) -> 12 (coordinate)
                                               -> 13 (demo)
16 (detector palla) + dati delle clip (17) -> 15 (eventi e valutazione)
```

Non è necessario eseguire tutto dall'inizio se si dispone già di modelli, tracce e detection salvate. Alcune celle prevedono proprio il riutilizzo degli output intermedi.

### Hardware

I test sono stati svolti principalmente su GPU T4 e L4 di Colab. La T4 è stata utilizzata per detector, tracker e demo; la L4 è risultata più adatta a SAM2. I valori FPS riportati negli esperimenti dipendono dalla GPU e dalla configurazione di esecuzione e non sono direttamente confrontabili se misurati su hardware differente.

## Problemi noti e limiti

Durante lo sviluppo ho incontrato alcuni problemi pratici che possono ripresentarsi quando si rieseguono i notebook:

- **SAM2 su T4:** l'attenzione standard può esaurire la memoria; è stata utilizzata una versione a blocchi, oppure la GPU L4.
- **Dipendenze Colab:** dopo alcuni aggiornamenti di PyTorch, Pillow e librerie collegate è necessario riavviare il runtime; nel notebook 13 è stato riscontrato anche un conflitto con `triton`.
- **ID delle tracce:** alcune esecuzioni nella stessa sessione possono mantenere contatori precedenti; per confronti ripetibili conviene avviare una sessione pulita.
- **Palla e possesso:** la palla rimane difficile da localizzare quando è nascosta, sfocata o molto piccola. Gli errori si propagano al riconoscimento degli eventi.
- **Tagli di scena:** flash fotografici e movimenti bruschi della telecamera possono essere interpretati come cambi di scena; il problema non è completamente risolto.

I risultati vanno letti tenendo presenti anche i limiti della valutazione:

- Le 7 clip NBA usate per la valutazione delle coordinate costituiscono un campione ridotto, con annotazioni puntuali anziché tracce MOT complete.
- SAM2 è stato confrontato con i tracker classici sui primi 300 frame delle sequenze SportsMOT, non sull'intera durata.
- L'omografia non è stata validata mediante un insieme indipendente di corrispondenze metriche: l'errore di fit e gli inlier di RANSAC sono controlli diagnostici.
- Le distanze e le velocità visualizzate nella demo non sono misure metriche validate.
- Il modulo eventi è stato verificato su poche clip di una sola partita, annotate da una sola persona. La lettura del tabellone usa coordinate specifiche per la grafica ESPN di quel video.
- La provenienza completa del dataset Roboflow v18 non è documentata nel materiale utilizzato.

Fra i possibili sviluppi ci sono un rilevatore dei cambi di scena più robusto ai flash, una gestione migliore delle maschere duplicate di SAM2, una stima più stabile della palla e del possesso e il riconoscimento dei numeri di maglia.

## Riferimenti

Modelli, dataset e librerie utilizzati nel progetto:

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [RT-DETR](https://arxiv.org/abs/2304.08069)
- [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2)
- [Fork SAM2 realtime utilizzato](https://github.com/Gy920/segment-anything-2-real-time) — commit `e3623cd`
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [Roboflow Supervision](https://github.com/roboflow/supervision)
- [Roboflow Sports](https://github.com/roboflow/sports)
- [OpenCV](https://opencv.org/)

I riferimenti scientifici completi ai metodi di tracking e alle metriche HOTA, MOTA e IDF1 sono riportati nella bibliografia della tesi.

---

Il progetto è stato sviluppato per attività di ricerca e sperimentazione nell'ambito della tesi magistrale. I filmati delle partite NBA non vengono distribuiti insieme al codice.
