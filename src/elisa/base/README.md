# Výnimky ELISA: význam, hranice a príklady

Pravidlá pre výber výnimiek definovaných v [error.py](error.py). Stav overený voči kódu **2026-11-02**.
Dokument rozlišuje význam chyby od spôsobu, akým ju spracuje konkrétny volajúci.

## Syntéza krivky a vyhodnotenie kandidáta

Výnimka sama osebe nepenalizuje model. O penalizácii rozhoduje až fittingový callback.
Pri priamom vytvorení modelu alebo volaní syntézy LC/RV propagovaná výnimka ukončí dané volanie;
volajúci ju môže zachytiť. Bez fittingovej hranice sa automaticky nevytvárajú penalizačné reziduály ani `-inf`.

Pri fittingu sa menia fyzikálne aj pomocné matematické parametre. Kandidát môže mať neprípustný model,
nekonvergujúci solver, nepokrytú tabuľkovú doménu alebo nepoužiteľnú numerickú geometriu.
Takéto rozpoznané zlyhanie má odmietnuť kandidáta a dovoliť vyhodnotenie ďalšieho.
Netreba dokazovať, že každé numerické zlyhanie má čisto fyzikálnu príčinu.

Chybná pevná konfigurácia alebo preukázané porušenie interného kontraktu majú výpočet prerušiť.
Opakované odmietanie kandidátov by ich neopravilo.

| Kontext | Príklad | Význam zlyhania |
|---|---|---|
| Priama syntéza | Používateľ zavolá `Observer.lc()` alebo `Observer.rv()` | Nezachytená chyba ukončí volanie; žiadna implicitná fittingová penalizácia. |
| Setup fitu | Pozorovania, medze, atlas, custom LD, počiatočný model | Neplatný vstup alebo nepoužiteľný počiatočný model ukončí prípravu. |
| Kandidát LSQ/MCMC | Nové parametre → model → krivka → cieľová funkcia | Rozpoznaná kandidátna chyba sa na príslušnej hranici zmení na penalizáciu. |
| Interný kontrakt | Chýbajúci flatten, nesúlad selectoru s meshom | Chyba implementácie/poradia krokov; nemá sa zamaskovať ako bežný zlý kandidát. |

Hot path zahŕňa aj konštrukciu a reinicializáciu fyzikálneho modelu, jeho komponentov a odvodených vlastností.
Platí to aj pre `SingleSystem`, hoci dnes nemá rovnaký vstavaný fitter ako binary modely.
Samotný názov `__init__` preto nie je dôvod na `ElisaInitialParametersError`.

## Ako vybrať kategóriu

Rozhoduje pôvod a význam chyby, nie pôvodný builtin typ ani adresár funkcie:

1. Je chybná pevná konfigurácia tasku, dát, počiatočného modelu, filtra alebo volania operácie?
   Použiť konkrétnu vstupnú kategóriu z nasledujúcej tabuľky.
2. Môže aktuálny kandidát vytvoriť fyzikálne alebo numericky nepoužiteľný výpočet?
   Použiť modelovú/doménovú kategóriu; pri nešpecifickom numerickom zlyhaní `ElisaNumericError`.
3. Garantuje správny interný flow daný stav, formát alebo poradie krokov?
   Použiť `ElisaUnexpectedError` a uviesť porušený kontrakt.
4. Je funkcia nízkoúrovňová a nevie určiť doménový význam?
   Vystaviť príslušný medzivrstvový marker; význam priradí jej caller.

Pevný selector nie je automaticky interný kontrakt. Napríklad `Observer.rv(method=...)` prijíma používateľský
argument, ale interný `BinarySystem.compute_rv` už očakáva podporovanú metódu.
Na verejnej hranici je nesprávna voľba `ElisaOperationInputError`, v internom dispatchi `ElisaUnexpectedError`.

## Vstupy, konfigurácia a interné kontrakty

Tieto kategórie nie sú všeobecnými známkami numericky neprípustného fyzikálneho kandidáta.
Ich význam sa nesmie odvodzovať zo širokého MCMC odchytu; skutočné odchyty sú popísané nižšie.

| Výnimka | Kedy ju použiť | Existujúci príklad |
|---|---|---|
| `ElisaAnalyticsTaskError` | Neplatná konfigurácia analytics tasku alebo jeho prepojenia. | `_fit_passbands`, výber vzorkovania a kompatibilita JD/fáz v [shared.py](../analytics/binary_fit/shared.py). Platné passbandy nezodpovedajú datasetom. |
| `ElisaAnalyticsDataError` | Neplatné pozorovania, ich jednotky/metadáta alebo neistoty. | `prepare_observations` v [cost_fns.py](../analytics/models/cost_fns.py); nekladné neistoty, nefinite pozorovania. |
| `ElisaInitialParametersError` | Neplatné počiatočné hodnoty, medze, jednotky/constrainty; chýbajúca povinná konfigurácia modelu; nepoužiteľný počiatočný model pri setupovej syntéze. Tiež pevná atlas/custom-LD konfigurácia. | `_validate_required_component_properties` v [system.py](system.py), `normalize_atlas` v [atm/const.py](../atm/const.py), setup adaptívneho vzorkovania. |
| `ElisaOperationInputError` | Neplatná voľba alebo kombinácia argumentov konkrétnej verejnej operácie. | `Observer.manage_time_series`: chýbajú oba spôsoby zadania alebo sú zadané fázy aj časy. `correct_potentials`: chýbajú phases aj distances. |
| `ElisaSettingsError` | Neplatné nastavenia, logging/INI konfigurácia, nepovolená mutácia settings alebo príprava podporných assetov. | Neznámy `INTERSECTION_BACKEND`/`RV_METHOD` v [settings.py](../conf/settings.py), nenájdená schéma. |
| `ElisaPassbandError` | Neplatný pevný filter, response tabuľka, jej vzorky alebo vlnové hranice. | `PassbandContainer.get_passband_df` v [passband.py](../observer/passband.py). Filter sa nemení driftom fyzikálnych parametrov. |
| `ElisaInputKwargsError` | Spoločný základ chýb názvov argumentov; uprednostniť konkrétneho potomka. | Rodič nasledujúcich dvoch tried. |
| `ElisaInvalidKwargsError` | Nepodporovaný názov argumentu. | `invalid_kwarg_checker` v [utils.py](../utils.py). |
| `ElisaMissingKwargsError` | Chýbajúci povinný názov argumentu. | `check_missing_kwargs` v [utils.py](../utils.py). Nezamieňať s chybnou číselnou hodnotou existujúceho poľa. |
| `ElisaPredictionAlignmentError` | Nesúlad kľúčov/tvarov predikcie a pozorovaní, reducerov alebo interpolačnej podpory. | `map_rv_predictions` v [shared.py](../analytics/binary_fit/shared.py), `_validate_residual_shapes` v [cost_fns.py](../analytics/models/cost_fns.py). NaN/Inf v hodnotách je numerická chyba. |
| `ElisaUnexpectedError` | Porušený interný stav, formát, metadáta alebo poradie krokov garantované ELISA flowom. Tiež neklasifikované zlyhanie obalené diagnostickým kontextom tam, kde to konkrétna hranica výslovne robí. | Neflattenovaný kontajner v `generate_harmonics`, selector nesúhlasí s aktuálnym meshom, chýbajúce pripravené MCMC štatistiky. |
| `ElisaGraphicsError` | Neplatná konfigurácia grafickej operácie. | Neplatná rovina grafického rezu v [binary plot.py](../binary_system/graphic/plot.py). Nie je to kandidátna fyzika. |
| `ElisaMonteCarloError` | Konfigurácia a spracovanie MCMC reťazcov/pomocných operácií. | Nesprávny tvar alebo nefinite počiatočný ensemble v [mcmc.py](../analytics/binary_fit/mcmc.py); chýbajúca reťaz pri summary. Nie je to numerické odmietnutie kandidáta. |
| `ElisaNotImplementedError` | Chýbajúca implementácia abstraktnej alebo nepodporovanej operácie. | `PositionContainer` v [container.py](container.py), nepodporovaný pulzačný model. |

`ElisaInitialParametersError` a `ElisaParametersError` majú odlišný význam aj pri rovnakom vzorci.
Constraint singularita v počiatočnom vektore je setupový problém; singularita nového kandidáta je modelové odmietnutie.
Nemeniteľnosť vstupu počas fittingu sama osebe nerozhoduje medzi Initial/Operation/Settings/Unexpected.

## Fyzikálne a numerické zlyhania kandidáta

Pri priamom volaní modelu tieto chyby propagujú. Pri kandidátnom vyhodnotení sú nasledujúce kategórie
pokryté LSQ penalizačnou skupinou, priamo alebo dedením; MCMC ich dnes tiež odmietne.

| Výnimka | Kedy ju použiť | Existujúci príklad |
|---|---|---|
| `ElisaParametersError` | Neplatné aktuálne fyzikálne parametre alebo vzťahy odvodených vlastností. | `StarProperties.mass`, `t_eff`, `SingleSystem.check_stability`, kandidátne constrainty. Aj modelový konštruktor môže túto chybu vyhodiť. |
| `ElisaNumericError` | NaN/Inf, overflow alebo iný rozpoznaný nepoužiteľný numerický výsledok bez vhodnejšej doménovej kategórie. | `validate_component_distance`, nefinite syntetické krivky, Qhull pri triangulácii hviezdy, clipping. |
| `ElisaAtmosphereError` | Pre kandidáta nemožno vybrať/načítať potrebnú atmosférickú tabuľku alebo získať použiteľný atmosférický výsledok. | Kandidátna teplota/gravitácia/metallicita mimo mriežky, chýbajúca vybraná tabuľka v [atm.py](../atm/atm.py). |
| `ElisaLimbDarkeningError` | Nemožno získať použiteľné LD koeficienty pre kandidáta. | Nepokrytá metalicita, chýbajúca vybraná tabuľka, zlyhanie interpolátora alebo nefinite interpolácia v [ld.py](../ld.py). |
| `ElisaTemperatureError` | Nepoužiteľný teplotný výpočet alebo neprípustná teplota. | Nulový menovateľ v `calculate_polar_effective_temperature`, nulový tok v `renormalize_temperatures` v [temperature.py](surface/temperature.py). |
| `ElisaSpotError` | Neprípustná geometria alebo zlyhanie výpočtu spotu. | Strata bodov/plôch, neúspešný spot solver, zásah do krčka v [mesh.py](../binary_system/surface/mesh.py). |
| `PulsationModeError` | Neplatný mód alebo nepoužiteľný výpočet pulzačnej perturbácie. | Kontroly v [pulse/pulsations.py](../pulse/pulsations.py). Nezamieňať s chýbajúcim flatten krokom. |
| `ElisaMaxIterationError` | Rozpoznaná nekonvergencia iterácie, vyhodená cez `raise_elisa_max_iteration_error`. | Orbitálny solver excentrickej anomálie alebo radius solver. Nevracať koreň po oznámenej nekonvergencii iba preto, že vyzerá číselne prijateľne. |
| `MorphologyError` | Fyzikálne neprípustná morfológia odvodená z parametrov. | `BinarySystem.compute_morphology`. Neznámy názov morfológie pri neskoršom dispatchi je interný kontrakt. |

Predpokladáme správne pripravené atmosférické/LD zdroje. Výber tabuliek závisí od kandidáta, a preto chýbajúca
vybraná tabuľka môže podľa dohodnutej defenzívnej politiky znamenať odmietnutie kandidáta.
Permission/device/ostatné systémové I/O chyby alebo poškodený pevný CSV obsah sa v týchto readeroch prekladajú
na `ElisaUnexpectedError` s príčinou. Neplatný pevný atlas alebo custom LD mapa patrí k inicializačným vstupom,
nie k penalizovanej atmosférickej/LD kategórii.

Pri geometrii je na niektorých miestach zámerne širšia tolerancia: clippingové adaptéry a nízkoúrovňový
spot solver odmietnu kandidáta aj pri chybe, ktorej presný pôvod ešte nie je rozlíšený.
Táto politika sa nesmie automaticky rozšíriť na ľubovoľný `except Exception` v aplikácii.

## Nízkoúrovňové a ostatné značky

| Výnimka | Pravidlo pre caller |
|---|---|
| `ElisaLookupError` | Numerický lookup nevie nájsť okolie/najbližšiu hodnotu. Atmosférický caller preloží marker na `ElisaAtmosphereError`, LD caller na `ElisaLimbDarkeningError`. Nie je samostatnou kandidátnou kategóriou; únik do vstavaného LSQ/MCMC sa mení na `ElisaUnexpectedError`. |
| `ElisaNumericSolverError` | Marker z `opt.fsolver.fsolver`. Caller musí priradiť doménový význam; dnešný binary spot caller ho prebalí na `ElisaSpotError`. Marker sám nie je v `SYNTHESIS_FAILURES`. |
| `ElisaError` | Spoločný základ viacerých doménových tried. Sám neznamená ani penalizáciu, ani úplný záchyt všetkých ELISA výnimiek. |
| `SolutionBubbleError` | Nosič navrhnutého riešenia cez atribút `solution`. Ponechaná zámerne aj bez aktuálneho produkčného callera; nie je všeobecnou kategóriou odmietnutia kandidáta. |

PYPEX je samostatná geometrická knižničná vrstva. Vo vnútri používa builtin výnimky a vlastné lokálne helpery;
ELISA význam priraďuje až coverage adapter. Python protokoly tiež zachovávajú zodpovedajúce builtin typy,
napríklad package `__getattr__` používa `AttributeError` cez `raise_attribute_error`.

Samostatná shape kontrola `_inputs` v clippingu používa builtin `ValueError` cez `raise_value_error`.
Pri priamom použití označuje neplatný tvar polygonu/trojuholníka; coverage adapter ju prekladá na
`ElisaNumericError` s pôvodnou výnimkou v `__cause__`, takže kandidát sa naďalej penalizuje.

## Príklady rozhodnutia z aktuálneho kódu

Nasledujúce bloky sú skrátené úryvky uvedených funkcií, nie samostatné spustiteľné programy.

### Hodnota kandidáta verzus vnútorný formát

[validate_component_distance](../binary_system/surface/validation.py) sa volá v `build_faces` a `build_surface_gravity`:

```python
if components_distance is None:
    raise_elisa_unexpected_error("Surface construction requires a resolved component distance; received None.")
# Nasleduje kontrola, že ide o reálny skalár, nie pole/bool/complex.
# Po overení reprezentovateľnosti čísla:
if not finite or components_distance <= 0:
    msg = f"Component distance must be finite and strictly positive; received {components_distance!r}."
    raise_elisa_numeric_error(msg)
```

NaN/Inf alebo nekladná vzdialenosť znamená nepoužiteľnú kandidátnu geometriu. `None` znamená, že kontajnerový flow
nevyriešil vzdialenosť pred volaním buildera. Nie sú to dva názvy pre tú istú príčinu.

### Úplnosť komponentu verzus neplatná aktuálna hodnota

[_validate_required_component_properties](system.py) kontroluje aktuálne atribúty, nie iba pôvodné kwargs:

```python
untouched_default = name not in instance.kwargs and value is up.NaN
if value is None or untouched_default:
    missing.append(f"`{name}`")
else:
    values[name] = value
```

Chýbajúca povinná konfigurácia je `ElisaInitialParametersError`. Nový numerický NaN alebo Inf sa nesmie zameniť
za pôvodný nedodaný sentinel; aktuálne hodnoty prejdú cez property validátory a môžu skončiť ako `ElisaParametersError`.
Konečná hodnota doplnená po vytvorení hviezdy je platná aj bez pôvodného kwargs kľúča.
Kontrola neprestavuje hviezdu a zachováva interné jednotky. Existujúca identita NaN sentinelu nerozlíši
nedotknutý default od opätovného priradenia presne toho istého sentinelu.

### Rovnaký constraint v setupe a v kandidátovi

`prepare_properties_set` a `constraints_evaluator` v [parameters.py](../analytics/params/parameters.py)
prenášajú kontext `raise_initial`:

| Kontext | Voľba | Výnimka pri chybe vyhodnotenia |
|---|---|---|
| Kontrola počiatočného vektora v setupe | `raise_initial=True` | `ElisaInitialParametersError` |
| Aktuálny kandidát LC/RV | `raise_initial=False` | `ElisaParametersError` |

Napríklad delenie nulou v constrainte uchová pôvodnú výnimku v `__cause__`.
Ani samostatné vyhodnotenie s `raise_initial=False` samo nepenalizuje: musí ho obklopovať fittingový callback.

### Chýbajúca príprava kontajnera

[generate_harmonics](../pulse/container_ops.py) predpokladá flatten krok. Rovnako [StarContainer.calculate_areas](container.py)
predpokladá vybudované `points` a `faces`; nejde o subset filtrovaný viditeľnosťou alebo zákrytom.
[_selector_to_mask a _validated_selected_face_areas](surface/coverage.py) vyžadujú, aby metadáta aj výsledné plochy
patrili tomu istému meshu. Tieto porušenia sú `ElisaUnexpectedError`.
Degenerované numerické výsledky majú dostať kandidátnu chybu pri svojom výpočte, pred skladaním coverage poľa.

### Tabuľka pre kandidáta verzus systémové zlyhanie

[_read_ld_csv](../ld.py) rozlišuje chýbajúcu vybranú tabuľku od ostatných I/O chýb:

```python
except FileNotFoundError as exc:
    raise_elisa_limb_darkening_error(f"There is no file like {path}.", exc)
except OSError as exc:
    raise_elisa_unexpected_error(f"Cannot read limb-darkening table {path}: {exc}", exc)
```

Pokazený CSV obsah/stĺpce má vlastný unexpected preklad. Atmosférický reader používa rovnaké rozdelenie.

### Zámerne defenzívny clipping

[_clipping_coordinates](../binary_system/surface/clipping.py) prekladá konverzné `TypeError`, `ValueError` aj
`OverflowError` na `ElisaNumericError` s príčinou. Overflow môže súvisieť s extrémnymi parametrami;
text namiesto čísla alebo nepravidelné vnorené zoznamy skôr naznačujú kontrakt. Presný pôvod zatiaľ nerozlišujeme
a podľa výslovného rozhodnutia kandidáta penalizujeme. Nie je to všeobecné pravidlo pre všetky konverzie.

## Dedičnosť a vyhadzovanie cez helpery

Väčšina doménových tried dedí od `ElisaError`. Výnimky z tohto pravidla sú:

| Trieda | Priamy builtin základ |
|---|---|
| `ElisaUnexpectedError` | `RuntimeError` |
| `ElisaNumericError` | `FloatingPointError` |
| `ElisaPredictionAlignmentError` | `ValueError` |
| `ElisaNotImplementedError` | `NotImplementedError` |

`ElisaMaxIterationError` dedí priamo od `ElisaError`. `MorphologyError`,
samostatné domény ATM/LD/teploty/spoty/pulzácie a `ElisaParametersError` sa do penalizačnej
politiky zaraďujú explicitne. **`except ElisaError` nie je úplný záchyt všetkých výnimiek z tohto modulu.**

Pri nových alebo upravovaných explicitných kontrolách používať príslušný `raise_elisa_*` helper:

```python
raise_elisa_numeric_error(message)       # nová rozpoznaná numerická chyba
raise_elisa_spot_error(message, exc)     # preklad existujúcej chyby so zachovaným __cause__
```

Helper vždy vyhodí výnimku a má návratový typ `NoReturn`; nevracia objekt chyby.
`return raise_elisa_unexpected_error(message)` sa používa aj v koncových vetvách funkcií s návratovou hodnotou,
aby statické nástroje rozpoznali ukončenie vetvy. Ak chybu iba prepúšťame v nezmenenom význame, použiť bare `raise`.
Samotné presunutie explicitného `raise` do helpera nerieši klasifikáciu.

Pri preklade zachovať príčinu a uviesť konkrétny problém, relevantnú hodnotu alebo zdroj.
Nový typ vždy posúdiť voči celému reťazcu lokálnych odchytov: napríklad coverage adaptéry a
`SingleSystem.calculate_radii` majú zámerne široké prebalenia. Nová unexpected chyba v ich vnútri
nemusí doraziť do fittingového callbacku v pôvodnom type.
`KeyboardInterrupt`/`SystemExit` sa nemajú meniť na chybu kandidáta.

## Súčasné fittingové hranice

Toto je opis aktuálneho kódu, nie univerzálne správanie odvodené iba z názvu triedy.

### LSQ

LC/RV callbacky v [least_squares.py](../analytics/binary_fit/least_squares.py) prekladajú na penalizačné reziduály
iba typy/potomkov zo [SYNTHESIS_FAILURES](../analytics/binary_fit/failure_policy.py):

```python
SYNTHESIS_FAILURES = (
    MorphologyError, PulsationModeError,
    ElisaLimbDarkeningError, ElisaAtmosphereError, ElisaParametersError,
    ElisaSpotError, ElisaTemperatureError,
    ElisaMaxIterationError, ElisaNumericError,
)
```

V predvolenom režime sa únik `ElisaLookupError` preloží na unexpected s príčinou. Ostatné chyby propagujú.
Pre `N` pozorovaní `_synthesis_penalty_residuals` vracia konečný vektor dĺžky `N`, ktorého každá položka je
`sqrt(MAX_USABLE_FLOAT / N)`. Jeho štvorcová norma je približne `MAX_USABLE_FLOAT`; pri lineárnej loss je
SciPy cost polovica tejto normy. Robustná loss tú istú hodnotu transformuje podľa svojej definície.

Kandidátna hranica zahŕňa aj `standardized_residuals(..., validate=False)`.
Reziduály sú `(observed - synthetic) / uncertainty`. Nefinite reziduály aj konečné reziduály s
nereprezentovateľnou štvorcovou normou vyvolajú `ElisaNumericError` a dostanú rovnakú penalizáciu.

### MCMC

[MCMCFit.ln_probability](../analytics/binary_fit/mcmc.py) používa rovnaké rozhodovanie ako LSQ:
rozpoznané `SYNTHESIS_FAILURES` odmietne cez `-np.inf`, ostatné výnimky predvolene propaguje.
Únik `ElisaLookupError` preloží na `ElisaUnexpectedError` s príčinou.
Prior rovný `-inf` je normálne odmietnutie bez diagnostiky; iný nefinite prior, likelihood alebo
posterior je `ElisaNumericError` s obmedzenou diagnostikou.

### PENALIZE_ALL_EXCEPTIONS

Spoločný boolean setting má predvolenú hodnotu `False`; INI kľúč `penalize_all_exceptions` patrí do
existujúcej sekcie `[computational]`. Podporuje aj `settings.override(PENALIZE_ALL_EXCEPTIONS=True)`.
Pri `False` oba fittery penalizujú iba rozpoznané fyzikálne alebo numericky nevyhodnotiteľné kandidáty.
Pri `True` prevedú každú bežnú `Exception` počas kandidátneho výpočtu na penalizáciu: LSQ vráti
penalizačné reziduály a MCMC `-np.inf`. Diagnostika zachová pôvodnú výnimku, aj uniknutý lookup marker.
Tento režim môže maskovať chyby konfigurácie alebo implementácie; odporúča sa iba ako dočasné núdzové riešenie.
Setup, používateľské operácie mimo callbacku a procesné `BaseException` zostávajú mimo núdzovej penalizácie.

`prepared_likelihood` v [cost_fns.py](../analytics/models/cost_fns.py) používa stabilizovanú aritmetiku a pri
nepredstaviteľnom konečnom výsledku vyhodí `ElisaNumericError`; priame volanie ju vystaví, MCMC ju odmietne.
Diagnostika MCMC je obmedzená veľkosťou; záznamy workerov sa automaticky nezlučujú do parent fitu.

### Adaptívny setup

`adaptive_sampling` v [shared.py](../analytics/binary_fit/shared.py) orchestruje hustú syntézu a následné
vzorkovanie podľa dĺžky krivky. Výnimky syntézy/normalizácie spracúva `_synthesize_initial_bolometric_curve`:

- Explicitne vybrané modelové/doménové/numerické chyby prekladá na `ElisaInitialParametersError` s príčinou,
  pretože zlyhal počiatočný model a setup sa nemôže dokončiť.
- Už označené initial/operation/passband/alignment/settings/unexpected chyby prepúšťa.
- Ostatné bežné chyby v tomto bloku obalí ako `ElisaUnexpectedError` s kontextom a príčinou.

Tento setupový odchyt má vlastný explicitný zoznam a zámerne nezávisí od `SYNTHESIS_FAILURES`.
Samotné `_sample_phases_by_curve_length` už iba interpoluje fázy z pripravenej krivky.

## Podklady

Rozhodnutia a historické výnimky sú v auditoch:

- [Explicitné výnimky hot/non-hot path](../../../docs/audits/EXPLICIT_EXCEPTIONS_HOT_PATH_AUDIT.md).
- [Klasifikácia ValueError](../../../docs/audits/VALUE_ERRORS_CLASSIFICATION_AUDIT.md).

Prázdne pracovné tabuľky znamenajú dokončenie ich inventára. Nie sú dôkazom, že všetky implicitné chyby
NumPy/SciPy/Astropy/Pandas alebo všetky možné cesty odchytov majú úplné pokrytie.
