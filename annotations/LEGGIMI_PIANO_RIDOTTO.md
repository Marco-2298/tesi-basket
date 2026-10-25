# Piano ridotto: riaprire Annota.cmd

Chiudere la vecchia finestra dell'app e riaprire Annota.cmd: una finestra già
aperta mantiene in memoria il vecchio piano da 295 frame.

Sono conservate tutte le 50 etichette già salvate nei cinque frame di clip_01
(38 con coordinate annotabili, 12 non annotabili). Il CSV manuale è rimasto
identico byte per byte; backup in reports/annotation_plan_reduction/.

Il piano ora contiene 40 frame complessivi, inclusi i cinque già affrontati:

- Per ciascuna delle altre sei clip: due frame, SOLO cinque giocatori per frame
  (60 nuove etichette complessive, anche non annotabili se necessario).
- Tre sequenze, ciascuna di otto frame, nelle clip 01, 05 e 08: SOLO due persone
  stabili. Al massimo 48 etichette; due possono essere riusate dal frame 26 già
  annotato. Nessun obbligo di completare gli altri giocatori delle sequenze.

Si riparte da clip_01, frame 26. Premi **Scegli i 2 ID della sequenza** e indica
due identità già presenti, riconoscendo le persone dai punti/etichette cyan.
Non serve salvare di nuovo quel frame: dopo la scelta puoi premere **Frame
completato**. Segui soltanto le stesse due persone nei sette frame successivi.

La scelta è salvata nel piano per tutti gli otto frame. I giocatori aggiuntivi
già annotati rimangono utilizzabili per gli errori spaziali ma non impongono
l'annotazione di altre identità lungo la sequenza. La valutazione della
continuità userà solo la coppia selezionata.

Nelle altre due sequenze scegli liberamente due persone riconoscibili; gli ID
sono umani, coerenti entro clip e indipendenti dagli ID SAM2. Se una persona è
occlusa o fuori campo, salva la relativa etichetta non annotabile senza
coordinate inventate. Le brevi sequenze sono proposte da verificare se ci sono
tagli di montaggio; la continuità sarà valutata su tre clip, non su tutte e sette.

Non serve raggiungere esattamente 100 punti visibili: annotazioni non
annotabili sono informazioni utili. Le circa 110 etichette del campionamento
spaziale (incluse quelle già fatte) e le sequenze sono due parti distinte.
I risultati andranno riportati anche per clip, perché clip_01 è più annotata.

Chiudere e riaprire l'app riprende dal primo obiettivo incompleto. Quando il
piano è concluso l'app lo segnala, senza chiedere di annotare gli altri giocatori.
Il vecchio ZIP Colab rimane utilizzabile per le inferenze; non estrarne il
vecchio piano di annotazione sopra i file locali aggiornati.
