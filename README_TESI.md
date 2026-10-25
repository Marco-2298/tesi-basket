# Tesi: detection e tracking dei giocatori nel basket

## Esperimenti
| # | Notebook | Cosa misura | Dati | Dove gira |
|---|---|---|---|---|
| (prep.) | `09_genera_tracce_SAM2_7clip` | genera le tracce SAM2 (f0 e redet) delle 7 clip | videos/ | Colab |
| E1 | `10_E1_detection_confronto` | 5 detector: AP, P, R, FPS | Roboflow test + SportsMOT basket | Colab |
| E2 | `11_E2_tracking_confronto` | 5 tracker + SAM2 (+ ri-detection) + interpolazione: HOTA, MOTA, IDF1 | SportsMOT basket | Colab |
| E3 | `12_E3_correzione_coordinate` | correttore originale (altezza) vs grezzo vs interpolazione pixel; SAM2 f0 vs redet | 7 clip + annotazioni | locale o Colab |
| Demo | `13_demo_squadre_mappa2D` | squadre, mappa 2D, heatmap, distanze | 1 clip | Colab |

## Percorsi
- Locale: `F:\Tesi Basket`
- Colab: `MyDrive/NBA_Tracking_Project`, con dentro le stesse cartelle `videos/`, `annotations/`, `notebooks/`

## Ordine
1. `00` in locale: pulizia (prima `DRY_RUN = True`).
2. Carica su Drive `notebooks/`, `videos/` e `annotations/` dentro `MyDrive/NBA_Tracking_Project`.
3. Su Colab: `09`, poi `10` e `11`.
4. Scarica da Drive `outputs/sam2_f0`, `outputs/sam2_redet`, `outputs/geometry` e `annotations/manifest_sam2_*.csv`, e mettili nelle stesse cartelle in F:.
5. In locale: `12` (prima con `VARIANT='f0'`, poi con `'redet'`).
6. `13`: demo con il tracker migliore emerso da E2.
