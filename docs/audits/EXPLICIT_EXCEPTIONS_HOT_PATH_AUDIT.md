# Audit explicitných výnimiek v hot path LC/RV a mimo nej

Aktuálna referenčná dokumentácia kategórií, príkladov a fittingových hraníc:
[src/elisa/base/README.md](../../src/elisa/base/README.md). Nasledujúci audit zachováva aj historické rozhodnutia.

Dátum: **2026-10-30**. Pracovný audit explicitných výnimiek jadra aplikácie a konfigurácie. Vrstva používateľského
rozhrania je úplne mimo rozsahu; pri ďalších aktualizáciách sa do inventára nezaraďuje. Inventár pôvodného snapshotu
doplnený o výsledky normalizácie a dohodnutú politiku výnimiek; historické počty nie sú aktuálnym úplným snapshotom
kódu.

Pôvodný snapshot: Git HEAD `c2863e88a20a380db7933a244788a9909ef40835`; SHA-256 zoradených ciest a obsahu všetkých **240
Python súborov**: `28e26799098c173f48e7655726b74698234c4f78a819de2e5683b31dbbea21db`. Riadky odkazujú na tento pracovný
stav a po ďalších úpravách sa môžu posunúť.

## Súhrn dohodnutého stavu pred ďalšou etapou

**Checkpoint: normalizácia pracovného zoznamu mimo hot path je dokončená; ďalšia etapa bude významová revízia hot path a
interných invariantov. Teraz sa ďalší refaktor nespúšťa.** Prázdna cold tabuľka neznamená, že boli vyriešené všetky
implicitné chyby alebo všetky odchyty aplikácie. Čísla nižšie pochádzajú z historického snapshotu a jednotlivých
kontrol, nie z nového úplného inventára.

### Základné rozhodnutia

- Pri vyhodnocovaní kandidáta je cieľom **defenzívne správanie: radšej kandidáta penalizovať než ukončiť celé
  fitovanie**. Zahŕňa to fyzikálne neprípustné aj matematicky či numericky nepoužiteľné stavy vyvolané driftom
  parametrov, vrátane nekonvergencie solvera. Na odmietnutie kandidáta netreba dokazovať fyzikálnu príčinu:
  stačí, že jeho model nemožno spoľahlivo vyhodnotiť. Potvrdené porušenia interného kontraktu a chyby pevných
  vstupov zostávajú oddelené; táto zásada neznamená plošné potláčanie všetkých výnimiek.
- Kategória má vyjadrovať **význam chyby a hranicu, ktorá ju vystavuje**, nie adresár ani pôvodný builtin názov.
  Rozlišujeme chybný vstup, neprípustný model/numerické zlyhanie kandidáta a porušený interný kontrakt.
- Hot path zahŕňa aj **všetku tvorbu a reinicializáciu fyzikálnych modelov**, vrátane `SingleSystem` a jeho odvodených
  vlastností. Neexistencia dnešného single-star fittera nie je dôvod tieto vetvy označiť ako cold. Budúci drift
  parametrov má dostať vhodnú modelovú/numerickú kategóriu bez ďalšieho prerábania inicializácie.
- Jednorazový setup tasku/dát a používateľské operácie sú odlišné hranice. To, že kontrola beží iba pri setupe, samo
  osebe nedokazuje, že chybu spôsobili používateľské dáta.
- Fitting vedome používa `_verify=False` a preskakuje JSON schémy. **Nepridávať schémové validácie ani opakované
  kontroly pripravených vstupov do hot path.** Kontroly fyzikálnych parametrov kandidáta a výsledkov výpočtu majú
  vlastný význam a zostávajú.

### Kategórie a dôvody ich rozdelenia

Nie všetky triedy vznikli v tejto etape; tabuľka zachytáva dohodnutý aktuálny význam vrátane skôr zavedených štyroch
validačných vrstiev.

| Kategória                                                                | Čo znamená a prečo ju máme                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ElisaAnalyticsTaskError`                                                | Neplatná konfigurácia analytics tasku alebo jeho prepojenia: napr. platné passbandy nezodpovedajú datasetom, nekompatibilná konfigurácia JD/fáz, nesprávny výber vzorkovania. Nie je to chyba fyzikálneho kandidáta.                                                                                                                                                                                                                                                         |
| `ElisaAnalyticsDataError`                                                | Neplatné pozorovania, konverzné metadáta alebo nepoužiteľné neistoty. Patrí sem aj nemožnosť odhadnúť kladné RV neistoty z dodaných dát. `radialcurves_mean_error` je teraz interný `_radialcurves_mean_error` a priamo používa dátovú kategóriu; nepodporujeme kvôli hypotetickému cudziemu použitiu ďalšiu medzivrstvu.                                                                                                                                                    |
| `ElisaInitialParametersError`                                            | Definícia vstupov pre fitting: počiatočné hodnoty, medze, jednotky, definície constraintov. Používa sa aj na setup hranici adaptívneho vzorkovania, keď známe modelové/numerické zlyhanie nedovolí vyhodnotiť počiatočný model. Samotné umiestnenie vo fitteri nestačí na túto kategóriu; vyhodnotenie constraintov v setupe používa `raise_initial=True`, v iteráciách `raise_initial=False` a `ElisaParametersError`.                                                                                                                     |
| `ElisaParametersError`                                                   | Parametre samotného fyzikálneho modelu, jeho komponentov a vzťahy medzi **aj odvodenými** vlastnosťami. Platí pri priamom vytvorení aj pri tvorbe kandidáta; napr. `SingleSystem.check_stability` kontroluje neprípustnú rotáciu. Inicializácia fyzikálneho objektu neznamená automaticky fittingovú `ElisaInitialParametersError`.                                                                                                                                          |
| `ElisaInvalidKwargsError` / `ElisaMissingKwargsError`                    | Samostatné kategórie neznámych/chýbajúcich **názvov** argumentov. Neplatná hodnota existujúceho poľa nie je chýbajúci názov. `SingleSystem.is_property` používa invalid-kwargs kategóriu. Požadovanú politiku pri vzniku vo vnútri kandidáta nemožno zameniť za dôkaz úplnosti dnešných odchytov.                                                                                                                                                                            |
| `ElisaOperationInputError`                                               | Nesprávne zavolanie verejnej operácie nad existujúcim modelom/taskom: phase aj time súčasne v `build_container`, neplatná rovina equipotenciály, uloženie neexistujúceho výsledku, `load_chain`/`filter_chain` na ne-MCMC tasku. Oddeľuje tieto používateľské operácie od modelových parametrov aj od interných invariantov.                                                                                                                                                 |
| `ElisaSettingsError`                                                     | Konfigurácia, INI/JSON logging policy, nepovolené mutácie settings, vytvorenie HOME a chýbajúci súbor schémy. Pôvodný `ElisaDownloadError` aj jeho helper boli zrušené; jeho explicitné chyby prípravy/sťahovania assetov patria sem. Nie je to signál neplatného fyzikálneho kandidáta. Implicitné sieťové/tar/I/O chyby download managera sa plošne neprebaľovali.                                                                                                         |
| `ElisaPassbandError`                                                     | Pevný používateľský filter: neznámy názov, nedostupná/nečitateľná response tabuľka, chybná definícia, vzorky či vlnové hranice. **Filter sa nemení driftom fyzikálnych parametrov; takáto chyba nesmie byť penalizácia.** Propaguje aj cez MCMC. Dodávané konečné záporné kalibračné vzorky sú podporované; nezaviedol sa všeobecný rozsah throughput 0–1.                                                                                                                   |
| `ElisaNumericError`                                                      | Rozpoznané numerické zlyhanie: NaN/Inf, overflow alebo nepoužiteľný numerický výsledok modelu/krivky/geometrie. Nahradil názov `ElisaFloatingPointError`, pretože význam je širší. Zachované dedenie od builtin `FloatingPointError`; trieda nededí od `ElisaError`. Na kandidátnej hranici sa tieto zlyhania penalizujú.                                                                                                                                                    |
| `ElisaMaxIterationError` | Rozpoznaná nekonvergencia iterácie/solvera. Dedí priamo od `ElisaError`; všetky raisery aj príslušné odchyty používajú tento typ a fitting ho explicitne penalizuje. Solver nesmie vrátiť koreň po oznámenej nekonvergencii iba preto, že vyzerá číselne prijateľne. |
| `ElisaPredictionAlignmentError`                                          | Štruktúrny nesúlad pozorovaní, neistôt, predikcií, pásiem, RV komponentov, reducerov alebo interpolačnej podpory. Premenovaná pôvodná `PredictionAlignmentError`, definícia a raise helper sú centrálne v `base/error.py`; lokálny LC `_fail` bol odstránený. Dedí od builtin `ValueError`, nie od numerickej/synthesis rodiny. **NaN/Inf nie je alignment.** Nezamieňať zlý filter s chýbajúcim pásmom vo výslednej predikcii.                                              |
| `ElisaUnexpectedError`                                                   | Porušený interný stav/poradie krokov, ktoré správny ELISA flow garantuje; ide o vývojársku chybu. Dedí priamo od builtin `RuntimeError`, nie od `ElisaError` ani synthesis rodiny. Prvý potvrdený prípad: po príprave neistôt chýba ich mapa, datasetový kľúč alebo hodnota ostala `None`. Používa sa tiež ako kontextové prebalenie neklasifikovaného zlyhania v adaptívnom vzorkovaní. Má propagovať, nie odmietnuť kandidáta. Plošná revízia ostatných invariantov ešte neprebehla. |
| `MorphologyError` | Priamy potomok `ElisaError`, explicitne uvedený v kandidátnej penalizačnej politike aj v setup odchyte. Neprípustná morfológia má vlastnú kategóriu. |
| `ElisaLimbDarkeningError` | Priamy potomok ElisaError; zjednocuje pôvodné LimbDarkeningError a LimbDarkeningSourceError. Potvrdené: nepokrytá metalicita, chýbajúce vybrané tabuľky a NaN/Inf výsledok interpolácie. Ostatné premenované použitia čakajú na klasifikáciu podľa tabuľky LD nižšie. |
| `ElisaGraphicsError`, `ElisaMonteCarloError`, `ElisaNotImplementedError` | Oddelené grafické konfigurácie, chyby MCMC reťazcov/pomocných operácií a neimplementované/abstraktné operácie. Monte Carlo reťaz nie je všeobecná fittingová ani modelová chyba. `ElisaNotImplementedError` zachováva dedenie od builtin `NotImplementedError`.                                                                                                                                                                                                              |

**Pozor na hierarchiu:** nie všetky ELISA-pomenované výnimky dedia od `ElisaError`.
Numeric/alignment/runtime/value/not-implemented značky majú vlastné builtin základy. `except ElisaError` preto nie je
úplný záchyt všetkých tried z tejto tabuľky. Nepridávať kvôli tomu všeobecný catch-all ani meniť dedenie bez revízie
správania.

### Dnešné odchyty — odlíšiť stav od zámeru

- LSQ dnes penalizuje `(ElisaAtmosphereError, ElisaParametersError, MorphologyError, ElisaMaxIterationError, ElisaNumericError)` cez
  `SYNTHESIS_FAILURES`. Ostatné typy z callbacku propagujú. Táto konštanta je existujúca politika LSQ, nie univerzálna
  definícia toho, čo má odchytávať každá inicializácia.
- MCMC dnes **explicitne propaguje `ElisaPassbandError` a `ElisaUnexpectedError`**. Ostatné bežné `Exception` jeho
  všeobecný odchyt stále zaznamená a premení na `-inf`. Preto napríklad alignment chyba ešte môže byť odmietnutie MCMC
  kandidáta, hoci nie je v LSQ penalizačnej skupine. Toto je zostávajúci problém revízie, nie potvrdené finálne
  správanie pre všetky kategórie.
- `adaptive_sampling` je **setup**, nie iterácia. Má vlastný explicitný odchyt
  `(ElisaAtmosphereError, ElisaParametersError, MorphologyError, ElisaMaxIterationError, ElisaNumericError)` → helper
  `ElisaInitialParametersError` s príčinou. **Nesmie závisieť od `SYNTHESIS_FAILURES`**, ktorý sa môže zmeniť alebo
  zaniknúť. Alignment/passband/settings/už označené unexpected chyby propagujú pôvodné; ostatné bežné chyby dostanú
  `ElisaUnexpectedError` s kontextom vyhodnotenia/normalizácie počiatočného modelu a pôvodným `__cause__`.
- Setupové chyby normálne ukončujú prípravu pred solverom. To neznamená, že všeobecný MCMC odchyt by každú takú triedu
  správne prepustil, keby sa dostala do callbacku. `SettingsError`, `OperationInputError`, názvy kwargs a ďalšie triedy
  nemožno automaticky považovať za globálne vyriešenú politiku len podľa mena.
- Ďalšia etapa musí preveriť **celý reťazec odchytov** vrátane lokálnych prebalení, fallbackov a worker transportu.
  Samotné zaradenie funkcie do hot path ani premenovanie triedy nedokazuje, čo sa nakoniec dostane k solveru.

### Výslovne dohodnuté hranice, ktoré sa nesmú stratiť

- **Single-star polomery:** nižšie radius helpery si ponechávajú numerické kategórie. `SingleSystem.calculate_radii`,
  volané cez `setup_radii`, však podľa výslovnej dohody prekladá každý bežný `Exception` z výpočtu polárneho/rovníkového
  polomeru na `ElisaParametersError`, s pôvodnou príčinou. Nie je to všeobecná požiadavka prebaľovať všetky modelové
  metódy. Rotačná stabilita a odvodené fyzikálne vzťahy tiež patria modelovej validačnej vrstve.
- **PYPEX:** pôvodne samostatná knižnica; má lokálny helper a builtin výnimky, **bez ELISA doménových errorov vo vnútri
  **. Priama knižnica a horizon tool zostávajú s builtin chybami. Až ELISA coverage adaptér prekladá bežné zlyhania
  geometrie na `ElisaNumericError`, lebo výpočet kandidáta má v týchto prípadoch pokračovať penalizáciou. `None` prienik
  je platná nulová plocha.
- **Sutherland–Hodgman:** batch funkcia sa volá `sutherland_hodgman_intersection_areas`; referenčná a kompilovaná vetva
  používajú rovnakú coverage politiku. Nesprávny batch tvar v `_validated_intersection_inputs` je zatiaľ defenzívne
  `ElisaNumericError`; rozlíšenie degenerovanej geometrie od interného kontraktu zostáva otvorené. Konverzné
  TypeError/ValueError/OverflowError v `_clipping_coordinates` sú zámerne `ElisaNumericError` s pôvodnou príčinou;
  kandidát sa defenzívne penalizuje. Tvar samostatného triangle helpera zostáva `ValueError`.
  NaN/Inf a nepoužiteľné plochy sú numeric.
  Validácia je oddelená do `_validated_intersection_inputs` a `_validate_intersection_areas`.
  Coverage zachová existujúci numeric error a ostatné bežné zlyhania výpočtu prevedie na numeric s príčinou.
  Kontrola tvaru polygon/triangle v `_inputs` je uzavretá: ponecháva `raise_value_error`, ktorý coverage
  prekladá na penalizovaný `ElisaNumericError`. Toto správanie je prijaté a nie je otvorenou auditnou položkou.
- **Dôsledok pre budúce `UnexpectedError`:** coverage a `calculate_radii` majú zámerne široké lokálne prebalenia. Ak sa
  do ich vnútra neskôr zavedie nová unexpected kontrola, môže sa týmto prebalením zmeniť na numeric/parameters a skončiť
  penalizáciou. Pred preklasifikovaním treba túto kolíziu vedome vyriešiť; neprepisovať hromadne triedy ani nerušiť
  dohodnutú toleranciu geometrických výpočtov bez posúdenia konkrétnej hranice.
- **Legacy LD čítače:** `get_ld_table` a `get_ld_table_by_name` nie sú v produkčnom `ld.py`; oba sú zachované lokálne v
  `scripts/benchmark/speed/limb_darkening.py` pod pôvodnými názvami, s priamym builtin `FileNotFoundError`. Slúžia
  historickej baseline/parite, nie produkčnej syntéze.

### Pravidlá ďalšej práce a otvorené rozhodnutia

1. Prejsť zostávajúce `ValueError` podľa skutočných volajúcich a garantovaného poradia.
   Príklad kandidáta na unexpected: pulzácie vyžadujú flattened container, ktorý mal flow pripraviť. „Zvyčajne sa to
   nestáva“ nestačí; treba doložiť interný invariant.
2. Revízia musí hľadať **aj existujúce helper volania**, nielen priamy `raise`. Mechanické presunutie raise do helpera
   samo osebe neuzatvára významovú otázku. Otvorené sú napríklad triangulačné `QhullError` prebalenia,
   chýbajúce/degenerované povrchové body, clippingové kontrakty a ďalšie už pomenované hot kontroly.
3. `ElisaUnexpectedError` je už zavedený pre pripravené neistoty a adaptive fallback; nerozširovať ho automaticky na
   každú chybu výpočtu. **`_ReviewError` zavedený nebol**; neisté prípady ostávajú explicitne v audite.
4. Zachovať `__cause__`, ak chyba vznikla z inej výnimky. Používať príslušné `raise_*` helpery. V PYPEX používať jeho
   lokálne builtin helpery, nie centrálnu ELISA hierarchiu. Podmienky písať čitateľne cez pomenované kontroly; väčšie
   validácie držať v samostatných `_` helperoch.
5. Vyriešené položky priebežne mazať z pracovných tabuliek. Definície tried/helperov z `base/error.py` ani nové interné
   raise mechanizmy nepridávať ako ďalšie problémy do inventára. UI zostáva mimo rozsahu; benchmarkové baseline chyby
   nie sú produkčné ELISA chyby.
6. Nepredpokladať, že všetky testy projektu sú zelené. Jednotlivé kroky mali cielené úspešné sady a širšie behy s
   reprodukovanými staršími zlyhaniami; počty nie sú sčítateľné. Posledný funkčný krok: **100 cielených testov prešlo**,
   Ruff prešiel. Tento checkpoint mení iba dokumentáciu a nespúšťa ďalšiu revíziu produkcie.

## Ako čítať delenie

„Hot path“ zahŕňa **výpočet LC/RV aj konštrukciu, inicializáciu a reinicializáciu fyzikálneho modelu a jeho komponentov
**. Platí to rovnako pre `BinarySystem`, `SingleSystem` a RV modely bez ohľadu na to, či dnes majú vstavaný fitter; nie
je to iba najvnútornejšia slučka cez fázy. Obsahuje aj validačné vetvy, ktoré pri platnom vstupe nevyhodia nič, ale
funkcia sa na tejto ceste vykonáva. Drahé cache miss vetvy a voliteľné fyzikálne scenáre sú zahrnuté, hoci sa nevykonajú
pri každom volaní.

- **S — syntéza krivky:** Observer LC/RV, orbitálne pozície, surface mesh, spoty, pulzácie, pokrytie, radiancie, limb
  darkening, odrazy, normalizácia a výstup v magnitúdach. Patria sem aj numerické a tabuľkové helpery, ktoré syntéza
  volá.
- **M — tvorba modelu pre kandidáta:** všetky modelové konštruktory, `from_json`, `init` a ich odvodené výpočty vrátane
  `SingleSystem`. Patrí sem validácia modelových Properties, konštrukcia hviezdy/spotov/pulzácií, orbitálne
  transformácie, odvodené potenciály a polomery, kontrola stability aj inicializácia beta/amplitúd/discretizácie.
  Použitie dnes iba pri jednorazovom vytvorení objektu nie je dôvod na vyradenie: rovnaká cesta môže vytvárať kandidáta
  budúceho single-star fitu. Vstavané binary/RV volania zostávajú konkrétnymi príkladmi M.
- **F — iterácia fittingu:** constrainty, kontrola aktuálnych hodnôt, mapovanie predikcií, likelihood a posterior. Toto
  je **rozšírenie prvej tabuľky nad samotnú fyzikálnu syntézu**; pri audite výlučne výpočtu krivky treba filtrovať na S
  a M. F neznamená, že ide o jednorazovú inicializačnú validáciu.
- **F\* — podmienená opakovaná validácia:** samostatné volanie `predict_light_curves` bez pripravených reducerov.
  Pripravený vstavaný fit túto kontrolu vykoná pri setup a neskôr ju preskočí.
- **Mimo hot path:** inicializačné fittingové metadáta, task/dataset setup, reťazce a výsledky, ploty, konfigurácia,
  legacy funkcie bez produkčného volania a abstraktné/neimplementované API. Modelová inicializácia vrátane single-star
  patrí do M; explicitné kontroly volania používateľských operácií mimo tvorby modelu (napr. súčasné zadanie phase/time
  do build_container) zostávajú mimo. Ľubovoľné opakovanie helpera vo vlastnej slučke používateľa nie je dôvod na
  zaradenie do hot path.

**LC, kinematické RV a radiometrické RV nemajú rovnaký call graph.** Povrch, atmosféry, odrazy a LD sa týkajú
LC/radiometrického RV podľa scenára; nemožno ich pripísať každej kinematickej RV krivke. Spoty, pulzácie, excentrické
aproximácie, intersection backendy, magnitúdy a multiprocessing sú podmienené vetvy. M pri RV závisí aj od
štandardnej/community parametrizácie.

## Čo sa počíta

Pôvodný AST inventár po vylúčení definícií raise infraštruktúry a pred priebežným vyraďovaním obsahuje **561 zdrojových
miest**: explicitné `raise`, aj bez argumentu, a volania `raise_*` helperov. Rovnaký typ v rovnakej funkcii je v tabuľke
zoskupený; **každý zdrojový riadok zostáva samostatným odkazom**.

Historické skupinové počty nižšie používajú pôvodné užšie delenie. Aktuálna definícia M zahŕňa aj single-star
konštrukciu a reinicializáciu; aktuálny pracovný zoznam je priebežne filtrovaný o vyriešené miesta.

| Skupina pôvodného snapshotu        | Zdrojových miest | Zoskupených riadkov tabuľky |
|------------------------------------|-----------------:|----------------------------:|
| Hot path a súvisiaca iterácia fitu |              247 |                         162 |
| Mimo produkčnej hot path           |              314 |                         189 |
| Celkom                             |              561 |                         351 |

V pôvodnom inventári po vylúčení definícií raise infraštruktúry je **329 príkazov `raise`** a **232 doménových volaní
raise helperov**. Inventár nepočíta iba definície tried výnimiek, `except`, `warnings.warn`, logovanie, vytvorenie
výnimky bez jej vyhodenia ani natívne chyby, ktoré môžu vzniknúť implicitne pri NumPy/Astropy/SciPy/Pandas/I/O operácii.
V aktuálnom produkčnom strome nebol nájdený žiadny `assert`.

Za doménové miesto sa uvádza **volanie** `raise_elisa_*`, nie znovu jeho vnútorné `raise`. Definície tried výnimiek a
implementácie raise helperov sa do auditu nezaraďujú.

## 1. Výnimky v hot path

### Nízkoúrovňové numerické lookupy

`ElisaLookupError(ElisaError)` je medzivrstva bez penalizačnej politiky. Päť helperov v `utils.py`
(`find_nearest_value`, `find_nearest_value_as_matrix`, `find_surrounded`, `find_surrounded_as_matrix`,
`find_idx_of_nearest`) normalizuje existujúce ValueError/IndexError/TypeError/ArithmeticError pomocou
`raise_elisa_lookup_error`; pôvodná natívna výnimka zostáva v príčine. Dva explicitné rozsahové raisy
používajú ten istý helper priamo. Výpočty, tie-breaking, okrajové hodnoty, NaN správanie a validácie sa nemenia.

Atmosférické adaptéry prekladajú lookup na `ElisaAtmosphereError`, LD výber na `ElisaLimbDarkeningError`.
`find_surrounded` môže preniesť marker z vlastných interných nearest volaní bez opätovného balenia.
`find_apsidally_corresponding_positions` zatiaľ obnovuje pôvodnú natívnu výnimku: politika calleru sa nemení.
Izolovaný neprodukčný `_atm.py` prekladá lookup z validačnej mriežky na unexpected.
Uniknutý marker je v LSQ aj MCMC preložený na `ElisaUnexpectedError`, aby sa nezmenil na tiché odmietnutie kandidáta.
`find_nearest_dist_3d` zostáva mimo tejto skupiny; nejde o vyhľadávanie na skalárnej mriežke.

### Limb darkening — prvý krok a zostávajúce rozhodnutia

Dohodnutá politika pre interpolate_on_ld_grid: chyby spracovania teploty, gravitácie a metalicity vrátane
konverzií a tvarov sa prekladajú na ElisaLimbDarkeningError s príčinou a penalizujú kandidáta. Príprava číselných
vstupov a vyhodnotenie kandidátnych súradníc sú oddelené od načítania a zostavenia zdrojového interpolátora.
Zdrojové ElisaUnexpectedError sa tým neprekladajú na penalizáciu. Passband sa považuje za pripravený, author
zostáva ignorovaný. Odložená revízia do budúcna: či tvarové/typové chyby treba oddeliť ako interné unexpected.


limb_darkening_factor má dve explicitné vstupné cesty: dodané cos_theta alebo _cosines_from_vectors.
Obe zdieľajú _limb_darkening_factor_from_cosines. None pri nepoužitých vektoroch je platné; chýbajúca geometria
pri vektorovej ceste je ElisaUnexpectedError. Zachované sú vzorce, priorita cos_theta aj nemennosť vstupov.


StarProperties.limb_darkening_coefficients: chybný formát custom mapy, skalár pre nevhodnú law a nesprávny
počet koeficientov sú ElisaInitialParametersError. Kontrola nepočíta fyzikálny model; konfigurácia je pevná.


Neúplné custom mapy pre požadované passbandy aj chýbajúci bolometric sú vyriešené ako ElisaInitialParametersError.
Ide o pevnú vstupnú konfiguráciu, nie drift parametrov; chyba propaguje a ukončuje fit. Odchyt KeyError zachováva
pôvodnú príčinu. Kontroly zostávajú pri spotrebe mapy, nepribudla ďalšia opakovaná validácia.


Nepodporovaná law vo fallback vetvách oboch výpočtov LD faktorov je vyriešená ako ElisaUnexpectedError.
Ide o interný kontrakt platnej pevnej konfigurácie, nie drift parametrov. Chýbajúca law je tiež vyriešená ako ElisaUnexpectedError; chýbajúce pripravené koeficienty sú tiež vyriešené ako ElisaUnexpectedError.


Čítanie LD tabuliek má paritu s atmosférami: _read_ld_csv prekladá FileNotFoundError na LD chybu,
ostatné OSError a chyby CSV/požadovaných stĺpcov na ElisaUnexpectedError s pôvodným __cause__.
Identifikácia konkrétnej tabuľky rozlišuje FileNotFoundError a ostatné OSError pri stat; is_file precheck je
odstránený, aby nezamaskoval systémové chyby ako chýbajúcu tabuľku. KeyboardInterrupt sa nechytá.


Vyriešené: CustomLimbDarkeningCoefficients.validate_law používa helper pre ElisaInitialParametersError.
Konflikt law s pôvodom custom koeficientov je pevná konfigurácia, nezávislá od driftu kandidátov; ukončuje fit.
Ekvivalencia linear/cosine a metadáta pri kopírovaní zostávajú zachované. Položka je vyradená z pracovných tabuliek.


`ElisaLimbDarkeningError(ElisaError)` nahrádza `LimbDarkeningError` aj `LimbDarkeningSourceError` bez aliasov.
Je explicitne v `SYNTHESIS_FAILURES` aj v setup odchyte `adaptive_sampling`; pri kandidátovi sa penalizuje,
pri príprave počiatočného modelu sa preloží na `ElisaInitialParametersError` s príčinou.
Potvrdené prípady používajú `raise_elisa_limb_darkening_error`: nepokrytá metalicita, chýbajúca vybraná tabuľka
vrátane zmiznutia pri čítaní/stat a NaN/Inf vo výsledku interpolácie. Tieto položky sú vyradené z pracovnej tabuľky.
Ostatné použitia pôvodnej LD výnimky boli iba premenované, ich doterajšia penalizácia zostáva do revízie zachovaná.
Premenovanie samo osebe nepotvrdzuje správnosť ich klasifikácie; explicitné raise zostávajú na dohľadanie.

Pracovná tabuľka LD je podľa dohodnutej klasifikácie prázdna. Všetky obyčajné výnimky z konštruktora
`LinearNDInterpolator`, vrátane `QhullError`, sa prekladajú na `ElisaLimbDarkeningError` s pôvodným `__cause__`.
Odchyt pokrýva iba konštruktor; čítanie tabuliek a ich spájanie zostávajú mimo neho. Formátovanie vstupov
nepridáva ďalšie validácie passbandu ani autora. Oddelenie patologických rozmerov/geometrie zdrojov na
`ElisaUnexpectedError` je vedome odložená budúca revízia, rovnako ako rozlíšenie typov/tvarov kandidátnych polí.

Transport atmosférických worker chýb je uzavretý: ElisaAtmosphereError a ElisaUnexpectedError zachovávajú
identitu aj príčiny, neklasifikované obyčajné chyby sa prekladajú na ElisaUnexpectedError s __cause__.
KeyboardInterrupt z workeru sa prenesie do hlavného vlákna; interrupt v hlavnom vlákne má prednosť pred chybou
vo fronte. Po zlyhaní hlavného vlákna sa fronta nevyhadzuje vo finally. Overené obidve cesty prerušenia aj ich súbeh.


Načítanie zdrojového kontajnera je klasifikované: FileNotFoundError vedie na ElisaAtmosphereError; chybný názov,
CSV, chýbajúce stĺpce, prázdne dáta, systémové a programátorské zlyhania vedú na ElisaUnexpectedError s príčinou.
Existujúce doménové výnimky sa neprebaľujú druhýkrát. error_queue prenáša klasifikovanú chybu; vyriešený je aj
súbeh s KeyboardInterrupt: fronta sa kontroluje až po úspešnom join, nie vo finally. Overené reálne chybné súbory aj simulované I/O/MemoryError.


I/O atmosférických tabuliek: FileNotFoundError zostáva ElisaAtmosphereError. Ostatné OSError pri identifikácii,
kontrole dostupnosti a čítaní CSV sa prekladajú na ElisaUnexpectedError so správou obsahujúcou cestu a s pôvodným
__cause__; nejde o ovplyvniteľné fyzikálne parametre kandidáta. Worker prenesie už preloženú chybu cez error_queue.


Atlas je vyriešený na vstupných hraniciach: atm.const.normalize_atlas normalizuje default pri konfigurácii,
StarProperties.atmosphere overuje explicitný atlas pri konštrukcii aj Star.init(), setup fitu normalizuje override
do vlastnej mapy. Neplatný atlas vyhadzuje ElisaInitialParametersError s príčinou; LSQ aj MCMC ho propagujú.
Vnútorné radiance/atm_files a domain/filename helpery predpokladajú kanonické ck/k/bb a znova nevalidujú;
radiance používa iba atlas == bb. Observer nemá žiadny validačný prepínač. Funkcia validated_atlas zostáva
ako vstupný normalizátor pre priame utility, napríklad get_atm_table. Prázdne Star.atmosphere zachováva default.


`ElisaMissingKwargsError` a `ElisaInvalidKwargsError` sú z pracovných tabuliek vyradené. **Požadovaná politika solverov:
** ak vzniknú pri tvorbe alebo vyhodnocovaní kandidáta, majú viesť k jeho odmietnutiu: LSQ má vrátiť penalizačné
reziduály a MCMC pravdepodobnosť `-inf`. Táto poznámka zaznamenáva požadované následné správanie; nepotvrdzuje úplnosť
aktuálnych odchytov a nemení obsluhu vstupných chýb pred spustením solvera. Po vyradení chýb názvov argumentov a
vyriešených kontrol zarovnania, numerických výsledkov, rozdelenia constraintov a odstránení zastaranej položky `validate_physical_parameters` má tabuľka hot path **0 riadkov / 0 zdrojových miest
pôvodného inventára**.

`MorphologyError` je vyriešená kategória: fyzikálne neprípustná morfológia odvodená z aktuálnych parametrov odmietne kandidáta. Je explicitne uvedená v `SYNTHESIS_FAILURES`; LSQ vracia penalizačné reziduály a MCMC `-inf`. Všetky štyri kontroly v `BinarySystem.compute_morphology` používajú `raise_elisa_morphology_error`. Dedičnosť ani výpočtová logika sa nemenili. Spotové chyby sú vyriešené samostatnou kategóriou `ElisaSpotError`.

`PulsationModeError` dedí priamo od `ElisaError` a je explicitne uvedená v `SYNTHESIS_FAILURES` aj v setup odchyte `adaptive_sampling`. Všetkých päť kontrol zostáva penalizovaných pri kandidátovi a používa `raise_elisa_pulsation_mode_error`; setup adaptívneho vzorkovania ich prebalí na `ElisaInitialParametersError` s príčinou. Budúca revízia: nulový pomer amplitúd v `temp_amplitude` už odmieta jediný produkčný caller `assign_amplitudes`, takže táto konkrétna kontrola môže byť interný kontrakt. Zatiaľ zostáva rovnaká kategória aj penalizácia; poznámka je tiež priamo pri kontrole.


`fsolver` je z pohľadu výnimiek vyriešený a jeho pôvodný riadok `Exception` bol odstránený z pracovnej tabuľky. `ElisaNumericSolverError` je nízkoúrovňová značka, priamy potomok `ElisaError`; nie je samostatne v `SYNTHESIS_FAILURES`. `opt.fsolver.fsolver` cez helper obalí každé `Exception` počas pokusu solvera (vedome aj možnú implementačnú chybu), so zachovaním `__cause__`. `KeyboardInterrupt` a ostatné `BaseException` neprebaľuje. Jediný produkčný caller, binary `mesh_spots`, preloží značku cez helper na penalizovaný `ElisaSpotError`, opäť s príčinou. Neúspešný návrat `(solution, use=False)` zostáva zachovaný a existujúca validácia `condition` zostáva mimo prebaľovania solvera. Spotová klasifikácia je vyriešená kategóriou `ElisaSpotError`.

Chýbajúce pripravené štatistiky v `MCMCFit.likelihood_fn` sú vyriešený interný kontrakt: `set_up()` ich musí pripraviť pred vyhodnotením kandidátov a drift parametrov ich nemení. Kontrola používa `raise_elisa_unexpected_error`; chyba propaguje cez MCMC callback a ukončí fit namiesto tichého odmietania všetkých kandidátov cez `-inf`.

`ElisaSpotError` nahrádza pôvodnú `SpotError` bez aliasu a dedí priamo od `ElisaError`. Je explicitne v `SYNTHESIS_FAILURES` aj v setup odchyte `adaptive_sampling`. Všetkých 12 miest používa `raise_elisa_spot_error`; prebalenia zachovávajú `__cause__`. Kandidát sa odmietne penalizačnými reziduálmi LSQ alebo `-inf` v MCMC; setup adaptívneho vzorkovania chybu preloží na `ElisaInitialParametersError`. Potvrdené sú všetky aktuálne prípady: úplné prekrytie a strata bodov/plôch spotu, neúspešný solver stredu či povrchových bodov, `use=False`, zásah do krčka, neprípustný polomer stredu/prvého prstenca aj nepozitívny uhlový polomer. Všetky zostávajú spotovou chybou s penalizáciou; žiadna z týchto kontrol sa teraz nepreklasifikuje na interný kontrakt. Vyriešené riadky boli odstránené z pracovnej tabuľky.

Binary `resolve_json_kind` je vyriešený interný kontrakt: caller musí dodať kompletnú dvojicu hmotností alebo `mass_ratio` a `semi_major_axis`/`asini`. Drift hodnôt nemení kľúče parametrizácie. Neúplná parametrizácia používa `raise_elisa_unexpected_error` a fit preruší; nejde o penalizovaný lookup. Diagnostika uvádza nájdené aj chýbajúce parametre vrátane správneho názvu `asini`.

`ElisaTemperatureError` nahrádza pôvodnú `TemperatureError` bez aliasu a dedí priamo od `ElisaError`. Je explicitne v `SYNTHESIS_FAILURES` aj v setup odchyte `adaptive_sampling`. Obe existujúce kontroly nepozitívnej teploty pri interpolácii albeda a bolometrického gravity darkeningu používajú `raise_elisa_temperature_error`; kandidát sa penalizuje. Výpočty a podmienky sa nemenili.

Nulový menovateľ v `calculate_polar_effective_temperature` a nulový bolometrický tok v `renormalize_temperatures` používajú `ElisaTemperatureError`, nie generický `ElisaNumericError`; ide o penalizované zlyhania teplotného výpočtu kandidáta. Beta a albedo tabuľky sú pevné `src/elisa/data/gravity_darkening.json` a `albedos.json`, nastavené cez `PATH_TO_BETA`/`PATH_TO_ALBEDOS`. Teplota určuje bod interpolácie, nie výber zdroja. Čítanie, JSON a získanie `x`/`y` aj identita albedo súboru prebaľujú chyby na `ElisaUnexpectedError` s príčinou. Produkční calleri interpolácií (`System.setup_betas` a `BinarySystem.setup_albedos`) odovzdávajú `Star.t_eff`; `StarProperties.t_eff` cez `quantity_transform` garantuje konečný skalár a potom kladnosť. Preto sa do týchto interpolácií nepridala redundantná kontrola NaN/Inf. Samostatná kontrola fyzikálnych/numerických výsledkov povrchových teplôt zostáva otvorená.

Štyri kontroly chýbajúcich masiek reflexnej symetrie sú vyriešené cez `raise_elisa_unexpected_error`. `use_quarter_star_test` riadi vytvorenie masiek aj výber symetrickej vetvy; pri zapnutej symetrii musí existovať boolean pole, nie `None`, bez ohľadu na drift hodnôt či počet viditeľných plôch. Komentáre sú pri kontrolách. V `faces.get_visibility_tests` bol opravený preklep `over-contfact` na `over-contact`, čím sa aktivuje zamýšľané vylúčenie nadbytočného kvadrantu. Zhoda výsledných teplôt overcontact reflexie s výpočtom na celom povrchu je pokrytá regresným testom.

Otvorené po revízii teplotných výpočtov:

| Miesto | Súčasný stav | Návrh na posúdenie |
|---|---|---|
| Povrchová teplota, renormalizácia, reflexia a pulzačné perturbácie | Nie všetky vetvy lokálne kontrolujú konečnosť a kladnosť výsledných teplôt | Potenciálna medzera: numericky neplatná teplota môže zlyhať až v atmosférach/LD alebo predikcii. Vybrať spoločnú hranicu bez redundantných kontrol v hot path. |

Štyri zostávajúce runtime kontroly mimo coverage sú vyriešené: chýbajúci mesh a chýbajúce základné body symetrie v `rebuild_mesh_detached`, návrat `None` napriek `return_values=True` v `update_surface_params` a nepodporovaný `fork` v `get_mp_context` používajú `raise_elisa_unexpected_error` (pri prebalení s príčinou). Nie sú kandidátnou penalizáciou. `ElisaUnexpectedError` dedí priamo od builtin `RuntimeError`; dedenie od legacy `ElisaRuntimeError` bolo odstránené. Aj kontrola chýbajúceho `bb_path` v `compute_surface_coverage` je vyriešená ako `ElisaUnexpectedError` cez helper. Predchádzajúca vetva `in_eclipse` musí vytvoriť obrys alebo vyhodiť chybu; degenerovaná projekcia neoprávňuje neskorší návrat `None`. Posledná runtime položka bola odstránená z pracovnej tabuľky a nepoužívaná trieda `ElisaRuntimeError` bola úplne zmazaná bez aliasu.

**Revízia explicitných TypeError je dokončená:** všetkých 10 vyhodení používa existujúci `raise_elisa_unexpected_error`. Ide o kontrakty typu objektu/rozhrania kontajnera, celočíselných počtov, troch JSON sekcií, validovanej inklinácie a interného výsledku Lagrange solvera. Pri prebalení chyby porovnania inklinácie sa zachováva pôvodný `__cause__`. Komentáre sú pri kontrolách; pracovné riadky boli odstránené. Nová trieda `ElisaTypeError` ani podtrieda kontraktových chýb sa nezaviedla. Tieto chyby fit prerušia, nepenalizujú kandidáta.

**Odchyt natívneho TypeError neznamená to isté ako explicitný kontraktový raise.** Nasledujúce existujúce preklady sa nemajú plošne meniť na `ElisaTypeError` ani `ElisaUnexpectedError`:

| Skupina odchytov | Dnešný význam |
|---|---|
| `constraints_evaluator` | `TypeError` z vyhodnotenia constraintu → pri kandidátovi `ElisaParametersError` (penalizácia), pri `raise_initial=True` `ElisaInitialParametersError`. |
| `base.transform` | Neúspešná konverzia modelovej hodnoty → `ElisaParametersError`; nemožno z typu natívnej výnimky automaticky vyvodiť programátorský kontrakt. |
| Päť low-level lookup helperov | `ElisaLookupError`; atmosféry/LD alebo iný caller určujú doménový význam. Existujúce preklady nemeníme. |
| Analytics dataset, príprava pozorovaní a odhad neistôt | `ElisaAnalyticsDataError` pri príprave vstupných dát. |
| Analytics počiatočné parametre a transformácie | `ElisaInitialParametersError`; výber vzorkovania → `ElisaAnalyticsTaskError`; chain-tools → `ElisaMonteCarloError`. |
| Passband vstupy/tabuľky, atlas názov, settings/logging | Už preložené na zodpovedajúcu pevnú vstupnú kategóriu. |
| `utils.is_empty`, parsovanie `content-length` v download manageri | Lokálny fallback (`False` alebo neznáma veľkosť); nejde o propagovanú výnimku kandidáta. |

`ElisaJSONEncoder.default` nemá vlastný explicitný `raise TypeError`, ale deleguje nepodporované objekty štandardnému JSON encoderu, ktorý ju môže vyhodiť. Ide o serializáciu, nie numerickú penalizáciu. Široké `except Exception` v už dohodnutých solverových/LD hraniciach môžu zachytiť aj natívny `TypeError`; tento audit ich politiku nemení. Implicitné TypeError z ľubovoľnej operácie v celom programe nie sú úplne pokryté inventárom explicitných raisov a pomenovaných odchytov.

Numerické konverzie a existujúce doménové odchyty kandidátnych chýb zostali nezmenené; natívny `TypeError` pri konverzii alebo constraintoch sa nemá automaticky považovať za interný kontrakt.

**Revízia zostávajúcich hodnotových výnimiek je navrhnutá v [samostatnom pracovnom audite](/home/mike/Work/projects/elisa/docs/audits/VALUE_ERRORS_CLASSIFICATION_AUDIT.md).** Aktuálny inventár vrátane helperov obsahuje 0 miest: 0 otvorených kontraktov, 0 pevných vstupov/operácií, 0 otvorených clipping kontrol a 0 otvorených zmiešaných prípadov. Sú to počty dnešných explicitných zdrojových miest, nie historické počty riadkov nižšie. Tri potvrdené kontroly `StarContainer.symmetry_points`, `mirror_face_values` a `symmetry_faces` už používajú `raise_elisa_unexpected_error` s komentárom o internom kontrakte; sú odstránené z otvoreného zoznamu. Neceločíselné aj záporné `size` v `surface_area_coverage` sú tiež vyriešené cez `ElisaUnexpectedError`; chýbajúce `partial_coverage` pre vybrané partial plochy je tiež vyriešený interný kontrakt cez `ElisaUnexpectedError`. Výpočet partial coverage odmieta nekonečné/nepoužiteľné výsledky cez `ElisaNumericError` pred prahovaním; degenerovaný eclipse obrys sa prekladá na numeric s príčinou. Kontroly flatteningu v `generate_harmonics` a `complex_displacement` sú vyriešené cez `ElisaUnexpectedError`; prípravu musí garantovať caller. `setup_periastron_critical_potential` tiež používa `ElisaUnexpectedError` pre neznámy interný názov komponentu. Neplatný selector v `resolve_curve_method` je vyriešený ako `ElisaUnexpectedError`. Neznámy komponent v `calculate_potential_gradient` je vyriešený cez `ElisaUnexpectedError`; lokálne selektory používajú nový `BinaryComponent(StrEnum)`. Rovnaká klasifikácia a enum platia aj pre `calculate_polar_potential_gradient_magnitude`. Interné selectory v `calculate_radius` a `critical_potential_static` používajú `ElisaUnexpectedError` a `BinaryComponent`. Verejne dostupný `component_to_list` a dve odmietacie vetvy v surface/wireframe rendereroch používajú `ElisaOperationInputError`. Neceločíselný aj záporný počet bodov v `symmetry_point_reduction` sú vyriešené cez `ElisaUnexpectedError`. Neznámy kind v `normalize_light_curve` je používateľská voľba operácie a používa `ElisaOperationInputError`. Sort selector v `OrbitalSupplements.sort` používa `OrbitalSortKey` a `ElisaOperationInputError`. None referenčná magnitúda v `convert_to_magnitudes` je vyriešená ako kontrakt pevných kalibračných dát cez `ElisaUnexpectedError`. Pevný RV dispatch v `BinarySystem.compute_rv` používa `RVMethod(StrEnum)` a neznámy selector končí cez `ElisaUnexpectedError`; Observer validuje verejný argument cez `ElisaOperationInputError` a settings validujú `RV_METHOD` cez `ElisaSettingsError`. MCMC obe kategórie explicitne propaguje bez penalizácie; LSQ ich nemá v `SYNTHESIS_FAILURES`. Numerické chyby radiometrickej syntézy tým nie sú dotknuté. Ostatné položky zostávajú otvorené. UI/pypex a už vyriešený horizon tool sú z tohto inventára vylúčené.

`_log_noise_scale` má rozdelené kontroly: NaN/Inf v skalárnom `ln_f` je `ElisaNumericError` a odmietne kandidáta v MCMC cez `-inf`. Penalizácia zahŕňa aj matematicky nepoužiteľné stavy fitovania, nielen fyzikálne neprípustné modely. Neskalárny `ln_f` je `ElisaUnexpectedError`, pretože fitovanie mení hodnotu parametra, nie jeho tvar; táto chyba fit preruší. Obe vetvy používajú helpery; stabilizovaný výpočet škálovania neistôt sa nemenil.

`ElisaParametersError` je z pracovnej tabuľky odstránená ako vyriešená kategória: pri vyhodnotení kandidáta ju LSQ
prevedie na penalizačné reziduály a MCMC na `-inf`. Jej význam a hranice zostávajú zaznamenané v súhrne vyššie.

**Constrainty sú rozdelené podľa kontextu vyhodnotenia.** Spoločný `constraints_evaluator` a kontrola odkazov
`check_for_invalid_constraint` používajú `raise_initial=True` pre `ElisaInitialParametersError` a `raise_initial=False` pre
`ElisaParametersError`; kontext prenáša `prepare_properties_set`. Spoločný setup LC/RV raz vyhodnotí počiatočný
vektor v inicializačnom režime. LC aj RV callbacky LSQ/MCMC explicitne volia kandidátny režim: chyba constraintu
potom vedie k penalizačným reziduálom alebo `-inf`, neukončí fit. Rovnaký kandidátny režim používajú vyhodnotenia
posteriorových vzoriek a výsledných modelových hodnôt; ich volajúci tým nezískavajú automatický odchyt/penalizáciu.
Pôvodné chyby pri `eval` (`ValueError`, `NameError`, `TypeError`) a aritmetické chyby vrátane delenia nulou sa
prekladajú príslušným helperom so zachovaným `__cause__`. Kontrola nevyriešených odkazov používa rovnaké rozdelenie.
Tým sú obe pôvodné hot-path položky `ElisaInitialParametersError` vyriešené a odstránené z pracovnej tabuľky.
Vzorce ani správanie NumPy pri NaN/Inf sa týmto krokom nemenia; doménové kontroly výsledných modelových hodnôt
zostávajú na modelových validátoroch.

Overenie rozdelenia: **191 passed / 5 failed** v cielenom behu parametrov, constraintov, fyzikálnej validácie a
adaptívneho setupu; všetkých päť zlyhaní bolo reprodukovaných aj s pôvodným produkčným kódom (štyri serializačné
prípady `confidence_interval`, jeden LC fixture s nepovoleným `system@gamma`). Nové testy overujú rovnakú singularitu
v setupe a v kandidátovi pre LC/RV a LSQ/MCMC, pôvodný `ZeroDivisionError` v `__cause__`, penalizáciu pred syntézou
a úspešné vyhodnotenie následného platného kandidáta. Ruff a `git diff --check` prešli.

| Výnimka / výraz                          | Súbor                                                                                                                                         | Riadok                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Funkcia / metóda                                               | Fáza  | Kontext                                                                                                                                                        |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|

Posledná revízia úplnosti komponentov: `object_params_validity_check` používa aktuálne atribúty, pôvod kwargs iba
na identifikáciu nedodaného NaN sentinelu. Chýbajúca povinná konfigurácia je `ElisaInitialParametersError`, chýbajúci
interný atribút `ElisaUnexpectedError`; numericky/fyzikálne neplatné aktuálne hodnoty sa overujú existujúcimi
property validátormi v interných jednotkách a zostávajú `ElisaParametersError`. Binary aj Single `init()` robia túto
kontrolu pred prestavbou. Optional defaulty a sentinelové vetvy sa nemenia; odvodený potenciál SingleSystem sa
neprevaliduje ako vstupný potenciál binary hviezdy. Nejde o všeobecnú transakčnosť init() pri neskorších zlyhaniach.

## 2. Výnimky mimo produkčnej hot path

Pracovný zoznam po kroku 1 (**2026-10-30**, overené voči HEAD `f92511f9` a nižšie uvedenej oprave): z tejto tabuľky boli
vyradené grafické výnimky a `ElisaMonteCarloError`, spolu **38 zoskupených riadkov / 60 zdrojových miest**. Vyradené sú
aj nepoužívané legacy validačné vetvy: **4 riadky / 4 zdrojové miesta**. Vyriešené vstupné validácie mimo iterácií boli
vyradené v rozsahu **40 riadkov / 74 zdrojových miest**, vrátane troch historických single-system záznamov so starým
typom výnimky. Vyradených je aj **11 riadkov / 11 zdrojových miest** neimplementovaných operácií, ktoré teraz používajú
`raise_elisa_not_implemented_error`. Po normalizácii ďalších **18 explicitných setupových výnimiek v `shared.py`**
zostáva v tabuľke mimo hot path **0 riadkov / 0 zdrojových miest pôvodného inventára**; záznamy vrstvy používateľského
rozhrania sú úplne vyradené. Súhrnné počty vyššie zostávajú historickými počtami úplného inventára; nejde o nový úplný
snapshot zdrojov.

Kontrola ručne vyhadzovaných chýb grafických modulov nenašla ďalší explicitný raise na preklasifikovanie. V
`fit_lc_summary_with_error_propagation` a `fit_rv_summary_with_error_propagation` sa chýbajúca MCMC reťaz zmenila z
`ValueError` na `ElisaMonteCarloError`; aj tieto dve miesta sú už vyradené. Grafické chyby používajú vlastnú doménovú
kategóriu, problémy samotnej reťaze zostávajú `ElisaMonteCarloError` aj pri použití z diagnostického grafu. Kontrola sa
týkala explicitných raise miest, nie všetkých možných implicitných chýb knižníc. Tabuľka hot path je filtrovaná aj o
vyradené kategórie chýb názvov argumentov podľa poznámky pri jej úvode. Opätovná kontrola explicitných grafických chýb
vrátane animácií, diagnostiky reťazcov, Observer plotov a ostatných zdrojových modulov neodhalila ďalšie miesto na
preklasifikovanie.

Filter vyriešených validácií zahŕňa štyri doménové kategórie a chyby neznámych/chýbajúcich názvov. Pri tomto filtri sa
odstraňovali konkrétne miesta mimo hot path a výpočtu cieľovej funkcie; kategórie chýb názvov argumentov boli následne
vyradené aj z hot path. Modelová validácia kandidátov, runtime constrainty, penalizácia, likelihood a LSQ reziduály
zostávajú v inventári; definície zdieľaných raise helperov sú z auditu vylúčené.

Overenie kroku 1: **115 testov prešlo** (`test_summary.py::test_error_propagation_summary_requires_a_monte_carlo_chain`,
`test_chain_io.py`, `test_chain_diagnostics.py`, `test_mcmc_diagnostic_plotting.py`, `test_plot.py`) s `PYTHONPATH=src`
a `/home/mike/Work/envs/elisaui/bin/python`.

V pracovnom inventári výnimiek mimo hot path momentálne nezostávajú žiadne nevyriešené položky. Nejde o nový úplný audit
všetkých implicitných chýb ani o uzavretie odloženej revízie hot path invariantov.

Jednoznačné explicitné validačné výnimky v `analytics/binary_fit/shared.py` používajú existujúce
`raise_elisa_analytics_data_error` alebo `raise_elisa_analytics_task_error`. Rozlišujú chyby pozorovaní/jednotiek,
nepoužiteľných tokov a neistôt od konfigurácie pásiem, identifikátorov, normalizácie, formátu atmosférickej
konfigurácie, kompatibility JD/fáz a vzorkovania. Pôvodne odložené chýbajúce pripravené neistoty a odchyt v
`adaptive_sampling` sú už vyriešené samostatnými krokmi zaznamenanými nižšie. Hlbšie validátory a ich prípadné úniky sa
teraz nemenili. Hot path, penalizácia, likelihood a reziduály zostávajú bez zmeny. Overenie: **128 cielených testov
prešlo** (`test_shared_setup_validation.py`, `test_observation_normalization.py`, `test_observation_preparation.py`,
`test_lc_preparation_and_lsq.py`); Ruff prešiel pre produkčný modul aj všetky nové/upravené testy. Širšia regresia: *
*870 testov prešlo, 4 testy zlyhali**. Ide o
`test_lc_prediction_consistency.py::test_invalid_support_escapes_physical_penalty_handlers` (3 prípady) a
`test_missing_passband_escapes_physical_penalty_handlers`, ktoré očakávajú propagáciu chýb predikcie namiesto
existujúceho MCMC odmietnutia kandidáta. Všetky štyri zlyhania boli reprodukované aj po obnovení všetkých menených
funkcií `shared.py` z HEAD v pamäti samostatného procesu; nejde o regresiu tejto normalizácie. Politika solverov a jej
testy sa v tomto kroku neupravovali.

Normalizácia neimplementovaných operácií mimo hot path: `ElisaNotImplementedError` dedí priamo od `NotImplementedError`;
helper `raise_elisa_not_implemented_error(message, exc=None)` používa spoločný `_raise_error` a zachováva príčinu.
Upravených bolo sedem abstraktných metód `PositionContainer`, `AbstractOrbit.orbital_motion`,
`OrbitalSupplements.to_orbital_position`, generický single-system LC stub a neimplementovaná metóda vyhladzovania
datasetu. Ďalší explicitný raise mimo hot path sa nepotvrdil ako chýbajúca implementácia. Tri pôvodné raise miesta v hot
path zostávajú nezmenené. Overenie: **79 testov prešlo** (`test_not_implemented_errors.py`, `test_analytics_data.py`,
`test_container_serde.py`, `tests_orbital/test_orbit.py`, `test_kinematics.py`); Ruff pre všetkých šesť zmenených
produkčných modulov prešiel.

Štyri explicitné chyby sťahovania a bezpečného rozbaľovania teraz používajú `ElisaSettingsError` a helper
`raise_elisa_settings_error`; tri príslušné riadky boli odstránené. Pôvodná príčina sa zachováva. Implicitné chyby HTTP
knižnice, súborového systému a tar archívov tento krok nemení. Overenie: **22 testov prešlo** (
`test_download_errors.py`, `test_not_implemented_errors.py`).

Numerická normalizácia: explicitné nekonvergencie používajú `raise_elisa_max_iteration_error`, rozpoznané nepoužiteľné
numerické výsledky `raise_elisa_numeric_error`. Neplatné pozorovania/neistoty pri príprave a samostatných reziduáloch
používajú dátovú kategóriu; štruktúrne chyby zostávajú alignment. Radius helpery už nevracajú koreň po hlásenej
nekonvergencii. Lagrange search opakuje iba známe aritmetické zlyhania, ostatné chyby propagujú. Vyriešené miesta boli
odstránené z tabuliek. Overenie: **1005 testov prešlo, 60 zlyhalo**; presne rovnakých 60 testov zlyhalo aj pred úpravou.
Cielený výber kategórií, likelihood a typed exceptions: **113 prešlo, 1 existujúce zlyhanie vynechané**. Ruff pre
upravené numerické moduly a testy prešiel; Pri pôvodnom numerickom kroku mal `mcmc.py` dve existujúce Ruff hlásenia;
následný passband krok presunul návrat za odchyty a označil zachovanú všeobecnú politiku explicitným komentárom.

Vstupné kontroly verejných operácií nad existujúcim modelom používajú `ElisaOperationInputError` a
`raise_elisa_operation_input_error`: súčasné zadanie fázy aj času v binary/single `build_container` a neplatná rovina v
binary `compute_equipotential_boundary`. Tri príslušné riadky boli odstránené z pracovnej tabuľky. Fyzikálne parametre
zostávajú modelovou validačnou vrstvou; numerické zlyhania aj interné invarianty sa týmto krokom nepreklasifikovali.
Zdieľané syntetizačné vetvy ani odchyty solverov sa nemenili. Overenie: **31 testov prešlo** (operation input a
binary/single konštruktory), Ruff prešiel. Širší výber s `test_synthesis_sources.py`: **48 prešlo, 23 zlyhalo**;
všetkých 23 zlyhaní sa reprodukovalo aj s pôvodnými produkčnými modulmi z HEAD.

Nastavenia: explicitné chyby v `conf/settings.py`, `conf/parsers.py` a `conf/logconfig.py` používajú jednotne
`ElisaSettingsError` a `raise_elisa_settings_error`. Patrí sem neplatná konfigurácia, nepovolené zápisy/mazanie
nastavení, formát intervalov a logging policy. Známe chyby načítania/parsing INI, JSON logging konfigurácie a vytvorenia
HOME sa prekladajú na konfiguračnej hranici so zachovanou príčinou; neplatný kandidát nastavení sa nepublikuje. Pôvodná
samostatná download kategória bola zrušená a jej štyri explicitné asset-setup chyby tiež používajú settings kategóriu.
Implicitné sieťové, súborové a tar chyby download managera sa týmto krokom nemenili. Python čítanie neexistujúceho
atribútu aj výnimky z používateľského tela `override()`/`logging_context()` zachovávajú pôvodný význam. Šesť vyriešených
riadkov bolo odstránených z pracovnej tabuľky; definície/helpery sa do inventára nepridávajú. Overenie: **202 testov
prešlo** (`tests/tests_cfg`, `test_download_errors.py`); Ruff pre upravené moduly a testy prešiel.

Passband vstupy: `ElisaPassbandError` dedí iba od `ElisaError` a nepatrí do `SYNTHESIS_FAILURES`. Helper
`raise_elisa_passband_error` normalizuje neznáme názvy, chýbajúce/nečitateľné CSV, chybné stĺpce/vzorky, neplatné pevné
vlnové intervaly a chyby vytvorenia interpolátora. Kontroly sú v `PassbandContainer`, RV passband inicializácii,
Observer bandwidth setup a validácii názvov vo fitting/task vstupe. Pri chybe sa zachováva pôvodná príčina; setter
neprepíše platný kontajner neplatnou tabuľkou. Dodávané kalibrované profily so zápornými konečnými vzorkami zostávajú
podporované; nevnucuje sa nový rozsah throughput 0–1. Kontrola `get_passband_df` bola odstránená z pracovnej tabuľky.

Passband chyba vždy propaguje z MCMC bez `-inf`, záznamu odmietnutia či penalizačného počítadla; explicitne propaguje aj
cez `adaptive_sampling`. Ostatné MCMC kategórie si zachovávajú existujúcu politiku. Nesúlad kľúčov/rozmerov novej
predikcie zostáva `ElisaPredictionAlignmentError`; platné pásma nezodpovedajúce datasetom zostávajú
`ElisaAnalyticsTaskError`. Atmosférické interpolačné zlyhania závislé od kandidáta sa automaticky nepreklasifikovali na
passband chyby. `bolometric` pri nepodporovanom type argumentu už používa `ElisaUnexpectedError`: ide o interný kontrakt, nie
chybu definície filtra. Overenie: **226 cielených testov prešlo** (passband, analytics transform/task a explicitné
validácie), širší výber **528 prešlo, 12 zlyhalo**; presne rovnakých 12 zlyhaní bolo aj pred úpravou. Ruff pre všetky
upravené moduly a testy prešiel.

Chýbajúci súbor schémy v `Registry._get_schema_path` používa `raise_elisa_settings_error`; príslušný explicitný lookup
záznam bol odstránený. Vo fittingu zostáva `_verify=False`: validácia schém sa vedome preskakuje, nepridávajú sa
kontroly do hot path ani nové odchyty kandidátov. Chýbajúca schéma je chyba aplikácie/inštalácie, nie fyzikálnych
parametrov.

PYPEX je samostatná geometrická vrstva: jeho explicitné validačné chyby používajú lokálny
`pypex.base.error.raise_value_error` a zachovávajú presný builtin `ValueError`. Neimportuje ELISA doménové výnimky; tri
vyriešené záznamy boli odstránené z pracovnej tabuľky a definície lokálnych error helperov sa do inventára nepridávajú.

Produkčné prieniky pre coverage volajú PYPEX cez jedinú integračnú vetvu
`binary_system.surface.coverage._calculate_pypex_intersection_areas`. Táto hranica prekladá ľubovoľný bežný `Exception`
z konverzie hull/faces, prienikov a výpočtu plôch na `ElisaNumericError` cez helper, s pôvodnou príčinou. Nesprávny
počet/tvar, NaN/Inf alebo záporné výsledné plochy tiež znamenajú numerické zlyhanie kandidáta. Chýbajúci prienik (
`None`) zostáva platná nula. PYPEX takto odmietne kandidáta v LSQ/MCMC bez zastavenia fitu; procesné
`KeyboardInterrupt`/`SystemExit` propagujú. Priame PYPEX adaptéry v `binary_system.utils` ani používateľský horizon tool
sa neprebaľujú a zachovávajú builtin chyby. Samotný PYPEX krok nemenil druhý backend; následná Sutherland-Hodgman
revízia je zaznamenaná nižšie. Ostatné coverage kontroly zostávajú samostatné. Overenie: **678 testov a 419 subtestov
prešlo, 1 test zlyhal**; rovnaké zlyhanie bolo aj pred úpravou (`test_private_integration_replay` očakáva starý
`ValueError` pri neplatnom settingu namiesto `ElisaSettingsError`). Nové integračné testy overujú celý PYPEX pipeline,
penalizáciu LSQ, MCMC numerickú kategóriu a pokračovanie ďalším platným kandidátom. Ruff pre upravené moduly a nové
testy prešiel.

Sutherland-Hodgman revízia: batch funkcia sa volá `sutherland_hodgman_intersection_areas`; produkčný router, testy aj
benchmarkové skripty používajú tento názov bez legacy aliasu. Skalárne vstupy aj batch majú pomenované kontroly tvaru a
konečnosti. Chybné typy/tvary zostávajú `ValueError` cez `raise_value_error` (samostatný kontrakt určený aj
na budúcu revíziu interných invariantov); NaN/Inf súradnice, natívne floating-point zlyhania a nepoužiteľné výstupné
plochy používajú `raise_elisa_numeric_error`. Dva vyriešené riadky boli odstránené z inventára.

Coverage má samostatný `_calculate_sutherland_hodgman_intersection_areas` adaptér pre referenčnú aj Numba vetvu. Už
existujúci `ElisaNumericError` propaguje bez dvojitého prebalenia; ostatné bežné výnimky z výpočtu prevedie na numerickú
kategóriu s pôvodnou príčinou. Platí tak rovnaká politika odmietnutia kandidáta ako pri PYPEX. Riadiace
`KeyboardInterrupt`/`SystemExit` propagujú. Výber backendu/settings ani clippingové vzorce, tolerancie, poradie
výsledkov, nulové prieniky a existujúce názvy settings sa nemenia. Overenie: **632 testov prešlo, 2 známe zlyhania boli
vynechané**; obe zlyhania boli znovu reprodukované s pôvodnými modulmi z HEAD (`test_private_integration_replay`,
`test_static_physical_range_check_retains_message`). Ruff pre upravené moduly a testy prešiel; upravené benchmarkové
skripty majú platnú syntax.

Samostatne vyriešené pripravené LC neistoty: zavedené `ElisaUnexpectedError` (dnes priamo od builtin `RuntimeError`) a
`raise_elisa_unexpected_error`. `_prepare_observation_errors` sa vykoná pred normalizáciou a garantuje mapu neistôt;
chýbajúca mapa, chýbajúci kľúč datasetu alebo hodnota `None` v `_validated_normalized_errors` preto znamenajú porušený
interný kontrakt. Pôvodný `KeyError` sa zachováva ako príčina. Existujúce nefinite/nekladné dáta ostávajú
`ElisaAnalyticsDataError`. Pôvodný pracovný záznam bol odstránený. Nová kategória nepatrí do `SYNTHESIS_FAILURES` a
explicitne propaguje cez MCMC aj adaptívne vzorkovanie bez zaznamenania penalizácie. Toto nie je plošná implementácia
odloženej revízie ostatných ValueError/RuntimeError invariantov.

Adaptívne vzorkovanie v setupe: `AbstractLCFit.adaptive_sampling` explicitne odchytáva iba `ElisaParametersError`,
`MorphologyError`, `ElisaMaxIterationError` a `ElisaNumericError` (nezávisle od meniaceho sa `SYNTHESIS_FAILURES`) a cez
`raise_elisa_initial_parameters_error` vystavuje `ElisaInitialParametersError` s pôvodnou príčinou, keď z počiatočného
modelu nemožno pripraviť vzorkovanie. Setup sa tým skončí, nejde o penalizáciu iterácie. Známe alignment, passband,
settings a interné invarianty propagujú pôvodné. Zostávajúci `except Exception` vystavuje cez
`raise_elisa_unexpected_error` chybu `ElisaUnexpectedError` s kontextom vyhodnocovania/normalizácie počiatočného modelu
a pôvodnou výnimkou v `__cause__`; neoznačuje ľubovoľné zlyhanie za chybné počiatočné parametre. Obe položky boli
odstránené z pracovnej tabuľky. Regresie pokrývajú známe kategórie syntézy, propagáciu ostatných chýb a opakovanie
skutočného adaptive setupu s platným modelom po zlyhaní.

## Revízia single-star konštrukcie podľa rozšírenej hot path

- Dohodnutá politika `SingleSystem.calculate_radii`: na inicializačnej hranici volanej zo `setup_radii` sa každá bežná
  výnimka z výpočtu polárneho/rovníkového polomeru prekladá cez `raise_elisa_parameters_error` na
  `ElisaParametersError`, s kontextom konkrétneho polomeru a pôvodnou výnimkou v `__cause__`. Nezostáva všeobecný
  `ValueError`. Samostatné nižšie radius helpery si ponechávajú svoje numerické kategórie; táto politika prekladu platí
  pre zostavenie systému a umožňuje odmietnuť kandidáta. Procesné výnimky ako `KeyboardInterrupt` sa neprebaľujú.
- `single_system.utils.resolve_json_kind` používa `raise_elisa_parameters_error` namiesto starého `LookupError` pre
  chýbajúcu modelovú parametrizáciu. Oba pôvodné nesprávne cold záznamy boli odstránené z inventára ako vyriešené.
- Preverené fázy konštrukcie/reinicializácie: `Star` a komponentové validácie, single-system Properties a
  OrbitProperties, orbitálna inicializácia, kritický potenciál, stabilita, polárny/rovníkový/ekvivalentný polomer a
  objem, beta, pulzačné amplitúdy a nastavenie discretizácie. `rotation_period`, `reference_time`, inclination a
  fyzikálne rozsahy používajú modelové validačné výnimky; nekonvergencia a nepoužiteľné polomery v nižších helperoch
  používajú numerické kategórie, pričom `calculate_radii` ich pri inicializácii prekladá na modelovú kategóriu podľa
  vyššie uvedenej politiky. Ich skoršie vyradenie z cold tabuľky sa nepovažuje za dôvod meniť ich na operation/settings
  výnimky; zostávajú M aj pri budúcom fittingu.
- `ElisaOperationInputError` sa v single-system module používa iba pre konfliktné argumenty verejného `build_container`;
  nevyskytuje sa na inicializačnej ceste. `is_property` kontroluje názvy a zachováva samostatný
  `ElisaInvalidKwargsError`.
- Už evidované otvorené kontroly zo spoločnej modelovej/syntetizačnej vetvy zostávajú hot:
  `System.object_params_validity_check` (`TypeError`/`ValueError`), single mesh discretizácia (`ValueError`),
  interné pulse kontrakty a triangulačné hranice. Nejde o nové cold operácie a tento krok ich automaticky
  nepreklasifikoval. Revízia `ElisaUnexpectedError` a všeobecného MCMC odchytu zostáva odložená.
- Regresie overujú skutočnú konštrukciu a `init()`: bežné výnimky z oboch radius helperov sa na inicializačnej hranici
  preložia na `ElisaParametersError` so zachovaným reťazcom príčin; rotačná nestabilita je tiež `ElisaParametersError`,
  ďalší platný model sa po zlyhaní zostaví. Overenie kategórií používa existujúci `SYNTHESIS_FAILURES`, nie hypotetický
  či novozavedený single-star fitter. Overenie: **15 nových regresií prešlo**; širší výber **150 prešlo, 10 zlyhalo**,
  presne rovnakých 10 testov zlyhalo aj pred úpravou (staršie očakávania výnimiek pri neplatnej single-orbit perióde).
  Ruff pre upravené moduly a nové testy prešiel.

## Otvorené hranice po numerickej normalizácii

- Triangulačné kontroly (`binary_system/surface/faces.py`, `single_system/surface/faces.py`) už používajú helper a sú
  odstránené z tabuliek priamych raise miest, ale ich význam zostáva otvorený: chýbajúce body a všeobecné prebalenie
  `QhullError` zatiaľ nerozlišujú numerickú degeneráciu od porušeného interného kontraktu. Ich posúdenie patrí do
  následnej revízie interných invariantov; samotný prepis na helper ich sémantické posúdenie nerieši.
- `_compiled_area` aj Python adaptéry používajú `raise_elisa_numeric_error`. Numba overload bez pôvodnej príčiny je
  definovaný pri výnimkách a registruje sa až pri importe clipping modulu; Python helper naďalej zachováva reťazenie
  príčin. Priame raise miesta boli odstránené z pracovnej tabuľky.
- Chýbajúce pripravené neistoty sú už vyriešené samostatným krokom; ostatné interné stavy a všeobecná revízia kategórií
  `ElisaUnexpectedError` aj odchytov LSQ/MCMC zostávajú otvorené; legacy `ElisaRuntimeError` už bola odstránená. MCMC naďalej zachytáva všeobecný
  `Exception` okrem explicitne propagovaných `ElisaPassbandError` a `ElisaUnexpectedError`; politika ostatných kategórií
  zostáva otvorená.

## Odložený druhý krok — porušenie interného stavu

**Nasledujúca etapa po tomto checkpointe: plošná revízia interných invariantov a hot path odchytov. Bez nového pokynu ju
teraz neimplementovať.** Cold pracovný zoznam je už prázdny; pripravené neistoty a adaptive setup sa vyriešili
samostatne. Tento plán zachováva kontext pre návrat.

Cieľ: rozlíšiť vývojárske kontroly konzistencie vnútorného flowu od očakávaných numerických zlyhaní kandidáta a od
vstupných validačných chýb.

- Prejsť všetky použitia `ValueError` vrátane miest už prepísaných na helpery. Podľa
  volajúcich overiť, ktoré kontroly strážia stav alebo poradie krokov, ktoré správne implementovaný produkčný flow musí
  garantovať. Samotná zriedkavosť chyby nestačí na toto zaradenie.
- Potvrdené porušenia interných invariantov zjednotiť pod `ElisaUnexpectedError` a príslušný raise helper. Ide o chyby
  určené vývojárovi: konzistentný vnútorný kód ich nesmie vyvolať. Nejde o návrh validačnej vrstvy pre koncového
  používateľa ani o bežné odmietnutie kandidáta.
- Konkrétny príklad: `Pulsations can be calculated only on flattened container.` Ak syntetizačný flow garantuje flatten
  pred pulzáciami, nesplnenie tejto podmienky je interná chyba, nie neplatná hodnota fyzikálneho parametra.
- `_validated_normalized_errors` je už vyriešený podľa záznamu vyššie; ďalší chýbajúci pripravený stav posudzovať podľa
  doloženého poradia prípravy, nie iba podľa umiestnenia v setupe.
- Dorevidovať triangulačné kontroly prázdnych/nepoužiteľných bodov a všeobecné prebalenia `QhullError`: oddeliť
  potvrdenú numerickú degeneráciu od porušenia interného kontraktu. Tieto miesta už používajú numerický helper; samotná
  neprítomnosť priameho `raise` preto neznamená dokončenú významovú revíziu.
- Rozpoznaná nekonvergencia a numericky nepoužiteľné výsledky zostávajú `ElisaMaxIterationError` / `ElisaNumericError`;
  chybné vstupné dáta a parametre si zachovávajú príslušnú doménovú kategóriu. Neisté prípady ponechať explicitne
  otvorené v audite, nezavádzať do produkcie pracovnú `_ReviewError`.
- Po zavedení internej kategórie overiť všetky relevantné odchyty vrátane lokálnych fallbackov a transportu výnimiek. *
  *`ElisaUnexpectedError` musí propagovať s diagnostikou; LSQ ani MCMC ho nesmú premeniť na penalizáciu alebo `-inf`.**
  Pre novú explicitnú kategóriu už MCMC a adaptívne vzorkovanie majú passthrough; lokálne prebaľovania a ďalšie budúce
  preklasifikované invarianty treba naďalej preveriť jednotlivo. Budúce zjednotenie politiky odchytov je súvisiaca
  samostatná práca; teraz sa nemení.
- Pri budúcej oprave zachovať pôvodné príčiny, používať raise helpery a vyriešené položky priebežne odstraňovať z
  pracovného inventára. Definície tried a helperov do inventára nezaraďovať.

## Overené prechody a významné hraničné prípady

1. **Modelové validátory sú hot aj bez dnešného fittera.** Single-star `from_json` → `Star` → `SingleSystem.__init__`,
   prípadne `init` → `__init__`, patrí do M vrátane single/orbit transformácií a odvodených
   výpočtov.  [LC model](/home/mike/Work/projects/elisa/src/elisa/analytics/models/lc.py:21) vytvára
   `BinarySystem.from_json(..., _verify=False)` cez
   `prepare_binary`; [RV model](/home/mike/Work/projects/elisa/src/elisa/analytics/models/rv.py:36) vytvára štandardný
   systém alebo community `RadialVelocitySystem`. Konštruktory modelov a ich komponentov overujú parametre. Preto
   `ElisaParametersError` aj kontroly názvov v zdieľanom `utils.py` majú hot použitie.
2. **Inicializačné fittingové hodnoty sú samostatná hranica.** `InitialParameter`, `BinaryInitialParameters` a
   inicializačné Properties patria do prípravy vstupov. Ich výnimky sú v druhej tabuľke. Výnimkou z adresárového
   pravidla sú **`constraints_evaluator` a `check_for_invalid_constraint`**, ktoré sú volané aj pre každého kandidáta
   cez `prepare_properties_set`. Kontext je už oddelený: setup `raise_initial=True` → `ElisaInitialParametersError`,
   kandidát `raise_initial=False` → `ElisaParametersError`. Vyriešené položky sú z pracovnej hot-path tabuľky odstránené.
3. **JSON schémy sa v bežnom kandidátovi preskakujú.** `validate_binary_json` nie je súčasťou štandardnej binary
   kandidátnej konštrukcie s `_verify=False`; single-star `from_json` tiež má `_verify` prepínač, pričom zapnutá
   validácia pri tvorbe modelu patrí do M; `resolve_json_kind` a community prevod sa však vykonávajú. Pri verejnom
   `from_json(_verify=True)` sú schémy kontrolované pred vytvorením modelu.
4. **Atmosféry a LD sú hot helpery.** načítanie/interpolácia tabuliek,
   `_wrap_grid_lookup_errors`, `utils.find_surrounded*`, `ld.get_ld_source_identities` a
   `CustomLimbDarkeningCoefficients.validate_law` sú na produkčnej ceste. Cache miss nie je setup-only. Staré samostatné
   CSV čítače boli odstránené; historický benchmark má vlastný lokálny baseline reader.
5. **Geometrický modul nemusí znamenať hot raise.** PYPEX backend konštruuje polygóny s `_validity=False`; prienik
   používa `mplpath.contains_points`, nie používateľský `is_point_in_polygon`. Colineárne prieniky hrán idú do
   dvojbodového SAT adaptéra. Tieto konkrétne PYPEX guards si zachovávajú builtin kategórie cez lokálny helper a boli
   vyradené z pracovných tabuliek; bežné zlyhania reálnej PYPEX coverage cesty už normalizuje ELISA integračný adaptér.
   Skalárny `optimized_intersection_area` wrapper sa obchádza batch cestou; jeho `_compiled_area` kernel je hot; jeho
   numerické výnimky už používajú helper.
6. **Abstraktné metódy a nepoužívané stubs sú mimo.** `PositionContainer.build*`, `AbstractOrbit.orbital_motion`,
   `OrbitalSupplements.to_orbital_position` a `single_system.lc.compute_general_lightcurve` sa nemajú zamieňať s
   konkrétnymi implementáciami volanými curve routermi. Verejné `build_container` guards tiež nie sú volané routermi,
   ktoré skladajú position containery priamo.
7. **Typy v premenných a opätovné vyhodenie.** `raise error_queue.get()` prenáša pôvodnú worker výnimku s neznámym
   konkrétnym typom. Bare `raise` zachová pôvodnú chybu, nepridáva novú doménovú triedu.
8. **Samotné raise miesto nemusí propagovať von.** Napríklad MCMC `ln_probability` zachytáva bežné výnimky a vracia
   `-np.inf`, s výnimkou explicitne propagovaných `ElisaPassbandError` a `ElisaUnexpectedError`; LSQ má vlastný
   `SYNTHESIS_FAILURES` odchyt. Lokálne fallbacky, odchyty a transport môžu výsledok meniť. Zaradenie do hot path
   neurčuje, či ide o fyzikálne odmietnutie, chybu používateľského vstupu alebo programátorský defekt.

## Vyriešený odkaz na neexistujúcu výnimku pri validácii času

`validate_primary_minimum_time` používa `raise_elisa_parameters_error` namiesto neexistujúceho `error.ValidationError`. Pre `t0 <= 0` teraz vzniká penalizovateľný `ElisaParametersError`, nie náhodný `AttributeError` pri hľadaní triedy. Existujúca podmienka a výpočty konverzie JD/fáz zostali nezmenené. Docstringy časovej vrstvy odkazujú na skutočnú kategóriu.

## Metóda a hranice overenia

- Inventár vznikol parsovaním všetkých 240 Python súborov pod `src/elisa` cez `ast`, nie iba textovým hľadaním `raise`.
- Vyhodnotené boli priame výrazy, class-only raise, prenos z error queue, opätovné vyhodenie a používané raise helpery.
  Počet príkazov `raise` je nezávisle zosúladený s počtom uzlov `ast.Raise`; každé inventarizované miesto patrí presne
  jednej skupine.
- Delenie vychádza zo statického čítania vstupov Observer/modelov, factory volaní, curve routerov, surface/atmosphere/LD
  helperov, callbackov a vyhľadania ich volajúcich. Nie je to dynamický profil ani formálny dôkaz úplného call graphu
  pri ľubovoľných podtriedach, dynamickom dispatchi alebo custom callbackoch.
- Pri nepoužívaných/stub funkciách znamená „mimo“ neprítomnosť doloženého volania v aktuálnej štandardnej produkčnej
  ceste. Výnimky pri vlastných rozšíreniach alebo použití helperov používateľom môžu vzniknúť v inom kontexte.
- Pri vytvorení pôvodného inventára sa nevykonávala syntéza kriviek, optimalizácia ani testy a nemenil sa produkčný kód,
  testy ani pôvodný validačný audit. Neskoršia cielená oprava dvoch summary výnimiek a filtrovanie tabuľky sú
  zaznamenané v kroku 1 vyššie. Implicitné natívne výnimky z volaných knižníc sú mimo tohto inventára explicitných raise
  miest.
