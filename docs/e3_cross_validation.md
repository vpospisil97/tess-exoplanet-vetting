# E3 — pětinásobná skupinová cross-validation včetně E1/E2

## Metodika a hranice platnosti

- Výchozí dataset: 1 918 označených signálů. CV používá pouze 1 628 původních train+validation signálů; 290 původních testovacích příkladů zůstává odděleno.
- Pět vnějších foldů metodou `StratifiedGroupKFold` seskupuje signály podle hvězdy TIC. Každý vývojový signál je jednou ve vnější validaci.
- Baseline se vyhodnotí jednou v každém foldu, CNN/Mantis a ablace E1/E2 ve třech seedech 42, 123, 2026 v každém foldu.
- Pro neuronové sítě se počet epoch vybírá oddělenou **vnitřní** TIC-validací. Po výběru se model inicializuje znovu a natrénuje na celé vnější trénovací části.
- Stejné výchozí hyperparametry jako v dokončeném projektu; nedělá se nový grid search. Normalizace baseline a Mantis se učí jen na aktuálním trénovacím foldu.
- E1: nepoužitá větev CNN se nuluje **až po enkodéru**. E2: vstupní kanál masky se nuluje, zatímco flux zůstává stejný.
- Výsledky CNN `both` a E2 `with_mask` jsou **stejné uložené CNN běhy**, nikoli nové nezávislé modely. Podmínky jsou tedy párované foldem a seedem.
- Hyperparametry byly původně vybrány na validačních příkladech, které nyní patří do CV poolu; jde proto o **dodatečné ověření robustnosti**, ne zcela nezávislý, nestranný odhad výkonu po výběru modelu.
- Uváděná SD je výběrová SD z pěti průměrů foldů. Nejde o 15 nezávislých datových splitů ani o interval spolehlivosti.

## Hlavní modely – validace v pěti vnějších foldech

| Model | AP | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Logistická regrese | 0.8016 ± 0.0247 | 0.7114 ± 0.0394 | 0.6726 ± 0.0361 | 0.6756 ± 0.0612 |
| CNN | 0.9000 ± 0.0151 | 0.8321 ± 0.0240 | 0.7414 ± 0.0317 | 0.7784 ± 0.0507 |
| Mantis-8M + MLP | 0.9289 ± 0.0071 | 0.8797 ± 0.0109 | 0.7947 ± 0.0232 | 0.8452 ± 0.0158 |

## E1 – ablace lokálního a globálního pohledu

| Varianta | AP | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Oba pohledy | 0.9000 ± 0.0151 | 0.8321 ± 0.0240 | 0.7414 ± 0.0317 | 0.7784 ± 0.0507 |
| Pouze lokální | 0.8855 ± 0.0266 | 0.8098 ± 0.0357 | 0.7278 ± 0.0354 | 0.7441 ± 0.0561 |
| Pouze globální | 0.8729 ± 0.0085 | 0.7868 ± 0.0132 | 0.7059 ± 0.0202 | 0.7656 ± 0.0197 |

**Oba minus lokální:** ΔAP = 0.0145 ± 0.0148 (pět průměrů foldů; kladných párů 12/15).

**Oba minus globální:** ΔAP = 0.0271 ± 0.0170 (pět průměrů foldů; kladných párů 13/15).

## E2 – ablace explicitní masky pokrytí

| Varianta | AP | ROC-AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| S maskou | 0.9000 ± 0.0151 | 0.8321 ± 0.0240 | 0.7414 ± 0.0317 | 0.7784 ± 0.0507 |
| Bez masky | 0.9015 ± 0.0152 | 0.8441 ± 0.0238 | 0.7754 ± 0.0246 | 0.8306 ± 0.0180 |

**S maskou minus bez masky:** ΔAP = -0.0014 ± 0.0040 (pět průměrů foldů; kladných párů 8/15).

## AP podle vnějšího foldu (průměr přes seedy)

| Fold | Baseline | CNN | Mantis | E1 lokální | E1 globální | E2 bez masky |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8113 | 0.9068 | 0.9373 | 0.9113 | 0.8594 | 0.9033 |
| 2 | 0.7644 | 0.8943 | 0.9246 | 0.8642 | 0.8810 | 0.9015 |
| 3 | 0.8139 | 0.8908 | 0.9203 | 0.8627 | 0.8714 | 0.8901 |
| 4 | 0.8278 | 0.9229 | 0.9350 | 0.9174 | 0.8794 | 0.9255 |
| 5 | 0.7908 | 0.8852 | 0.9273 | 0.8721 | 0.8731 | 0.8869 |

## Interpretace a omezení

- E1/E2 představují citlivost na konkrétní hvězdy a náhodnou inicializaci při pěti skupinových rozděleních, nikoli důkaz kauzality či záruku výkonu na neověřených kandidátech.
- Rozdíl mezi dvěma variantami je vhodnější hodnotit **párově v rámci stejných foldů a seedů** než srovnáním nesouvisejících průměrů.
- Neaktivní větev v E1 nemá stejnou efektivní kapacitu jako dvouvětvová CNN a nulový flux může i bez explicitní masky zčásti indikovat chybějící pozorování.
- Výsledky tohoto E3 nejsou náhradou dříve zamrazeného jednorázového testu; původní metriky se nepřepisují.
- CV SD neodstraňuje riziko biasu katalogových štítků, reprezentativnosti dat ani výběru hyperparametrů před CV.

## Reprodukce

```powershell
python -u .\scripts\training\prepare_e3_folds.py
python -u .\scripts\training\cross_validate_baseline.py
python -u .\scripts\training\cross_validate_cnn.py
python -u .\scripts\training\cross_validate_mantis.py
python -u .\scripts\training\cross_validate_ablations.py --experiment all
python -u .\scripts\training\report_e3_full.py
```
