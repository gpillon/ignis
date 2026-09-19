# La codebase come memoria n-gram — risultati

- Data: 2026-09-19
- Branch: `ngram-study`
- Esegue: [`02-storage.md`](02-storage.md) §7, sul veicolo del §8
- Stato: **in corso** — veicolo e Fase 0 chiusi, Fasi 1-4 in esecuzione

Tutti i numeri qui sono misurati su questa macchina. Gli script stanno in
`scripts/`, gli output grezzi in `results/`, l'identità esatta di modello e
ambiente in [`results/env.json`](results/env.json).

---

## 0. Il veicolo — la strada del §8 è chiusa, un'altra è aperta

Il §8 indica NVFP4 come unico formato che entra nel budget di disco. **Non
entra in VRAM**, e la ragione non è il formato ma l'engine:

- transformers 5.17 passa ogni checkpoint compressed-tensors che non sia
  FP8 puro a `apply_quantization_config(..., run_compressed=False)` e poi a un
  forward pre-hook che **decomprime l'intero modello al primo forward**
  (`quantizers/quantizer_compressed_tensors.py`);
- `CompressedLinear`, la classe che decomprimeva a ogni chiamata, in
  compressed-tensors 0.18 solleva `"no longer supported"`.

Quindi `unsloth/Qwen3.8-27B-NVFP4` (21,5 GiB su disco) atterrerebbe come BF16
denso, ~54 GB contro 32,6 GiB di card. Il checkpoint FP8 ufficiale ha invece
un path a kernel nativo, ma a **28,77 GiB** lascia meno di 4 GiB per
attivazioni e logit: troppo stretto.

**La strada aperta**: il BF16 di base quantizzato in ingresso da
bitsandbytes 0.50.2, che spedisce una build cuda130 con `sm_120`.

| | |
|---|---|
| sorgente | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2…` |
| combacia col pin dell'artifact ignis | **sì** (`env.json: matches_pin`) |
| dimensione sorgente | 51,8 GiB — non entra su nessun disco locale, sta sulla share di rete |
| quantizzazione | NF4 + double quant, compute BF16 |
| non quantizzati | `lm_head`, `visual`, `in_proj_a`, `in_proj_b` |
| risultato | **16,49 GiB** su `F:/ai/models/Qwen3.8-27B-nf4` |
| picco VRAM in conversione | 16,62 GiB |
| load: da rete / da locale | 516 s / **9 s** |

I moduli lasciati fuori dalla quantizzazione ricalcano la `ignore` list di
unsloth: `in_proj_a` e `in_proj_b` sono le proiezioni di gating e decay del
Gated DeltaNet, dove 4 bit perturbano lo stato ricorrente molto più che in un
MLP 5120×17408. Quantizzarle metterebbe un confondente fra "nessun segnale" e
"ricorrenza rotta".

Due kernel opzionali non sono installati (`causal_conv1d`,
`flash-linear-attention`): transformers cade sul riferimento PyTorch, che è
**corretto ma lento**. Per una misura di segnale va bene; è il motivo per cui
i tempi di forward qui non dicono nulla sulla performance.

### Determinismo — il controllo α = 0 è leggibile

Due forward identici sullo stesso file danno logit **bit per bit identici**
(`max_abs_diff = 0.0`, 1043 token,
[`results/determinism-27b.json`](results/determinism-27b.json)). Senza questo,
α = 0 non distinguerebbe "il path di iniezione è pulito" da "rumore di fondo".

---

## 1. Il corpus in BPE — il caveat del §4 era pessimista

Il §4 contava gli n-gram word-level e dichiarava un'incertezza di ±2× nel
passaggio a BPE. Col tokenizer vero il fattore è **1,18×**.

| | §4 (word-level) | misurato (BPE) |
|---|---|---|
| file | 802 | **801** |
| byte | 9.071.351 | **9.066.024** |
| token | ~2,7 M (stimati) | **2.599.881** (3,487 B/token) |
| 2-gram distinti | 240.992 | **294.764** |
| 3-gram distinti | 579.715 | **673.833** |
| **righe, n-gram scorrevole** | 820.707 | **968.597** |
| simboli | 11.760 | **12.780** (7.598 nomi distinti) |

L'argomento di storage del §4 non solo regge, si rafforza: a riga = hidden del
27B (5120 × 2 B = 10.240 B), l'n-gram scorrevole costa **9,9 GB** contro i
**131 MB** dell'indice per simbolo — **76×**, non 70×.

I simboli sono contati da un parser vero, non da grep: 801 file, **zero
fallimenti di parsing**. Le trappole che contano sono lessicali, e ognuna è un
test (`scripts/test_symbols.py`, 26 casi): `/* */` annidati in Rust ma non in
C, `r#"…"#` che contiene una graffa spaiata, `'a` lifetime contro literal di
carattere, `10'000` separatore di cifre C++, apostrofo nel testo JSX, e i
buchi `${…}` nelle template literal.

### La regola delle chiavi rare, resa operativa

Il §7 Fase 4 chiede chiavi "rare e distintive". Misurata: almeno 2 token BPE
**e** contenuta in nessun altro nome.

| | |
|---|---|
| nomi distinti | 7.598 |
| ≥ 2 token BPE | 6.792 |
| unica per sottostringa | 6.152 |
| **entrambe (chiavi rare)** | **6.033** (79,4 %) |

Le escluse sono esattamente quelle che il piano temeva: `A`, `ACTIVE`, `ALL`,
`ALIGN`, `ARTIFACT` — nomi che su codice estraneo matchano di continuo.

---

## 2. Fase 0a — go/no-go: gli hidden non sono collassati

200 simboli di primo livello (`fn`, `struct`, `function`, `class`), corpo ≥ 120
caratteri, nome unico nel corpus, al massimo 4 per file.

**Il coseno grezzo non è leggibile**, e la tabella mostra perché: a L2 vale
0,995 fra *qualsiasi* coppia. È l'effetto delle dimensioni ad attivazione
massiva del residual stream, non somiglianza semantica. I numeri che contano
sono quelli **centrati sulla media dell'insieme**, e letti **contro un gruppo
di controllo** di span presi a caso nello stesso file.

| L / pooling | coseno centrato (def) | PC1 (def) | rango eff. (def) | rango eff. (controllo) | coseno grezzo |
|---|---|---|---|---|---|
| L2 / last | −0,002 | 0,270 | 18,3 | 35,2 | +0,995 |
| L2 / mean | +0,000 | 0,605 | 9,5 | 7,2 | +0,998 |
| L5 / last | −0,001 | 0,291 | 17,4 | 37,5 | +0,995 |
| L5 / mean | +0,003 | 0,589 | 9,6 | 6,9 | +0,998 |
| L19 / last | −0,004 | 0,128 | **68,3** | 143,5 | +0,931 |
| L19 / mean | −0,004 | 0,092 | **71,9** | 61,9 | +0,983 |
| L33 / last | −0,004 | 0,116 | **85,0** | 150,2 | +0,796 |
| L33 / mean | −0,004 | 0,078 | **79,4** | 69,9 | +0,961 |
| L47 / last | −0,004 | 0,115 | 82,2 | 149,6 | +0,690 |
| L47 / mean | −0,004 | 0,108 | 68,6 | 59,3 | +0,920 |
| L61 / last | −0,004 | 0,151 | 60,5 | 149,0 | +0,610 |
| L61 / mean | −0,004 | 0,145 | 67,1 | 64,1 | +0,803 |

Rango efficace massimo possibile: 199.

**Verdetto: 0a passa.** Il timore del §3 — l'ultimo token di un corpo (`}`) è
addestrato a predire `\n\n`, quindi i vettori potrebbero essere tutti lo
stesso — è reale ma parziale:

1. **Non sono collassati.** Coseno centrato ≈ 0 a ogni layer e con entrambi i
   pooling; un collasso darebbe rango efficace ≈ 1, qui è 60-85.
2. **Sono però più concentrati del caso.** A L19-L61 con pooling `last` il
   rango efficace è 60-85 contro i 143-150 del controllo: circa **la metà
   della dispersione** di posizioni qualsiasi. Quella metà mancante è
   precisamente l'effetto "fine-definizione" che il §3 sospettava.
3. **Il pooling `mean` si comporta all'opposto.** Sui layer profondi pareggia
   `last` (68-79) ma **supera il controllo** (59-70) invece di starci sotto:
   la media sul corpo non ha la direzione comune del token finale. Sui layer
   superficiali invece collassa (9,5 contro 7,2 del controllo), perché a L2 la
   media di un corpo è poco più della media degli embedding.
4. **I layer superficiali non servono.** A L2 e L5 tutto è concentrato
   (rango 9-37): il residual è ancora dominato dall'embedding del token. Il
   punto di iniezione a layer 2 di Flash-Next non trasferisce a questo uso —
   là l'n-gram *è* informazione sul token corrente, qui è un riassunto.

Non serve ripiegare sulla via di scampo del §7 ("provando prima l'altro
pooling"): entrambi i pooling sopravvivono, su layer diversi.

---

## 3. Fase 0b — associatività query↔valore

Per ogni simbolo: l'hidden nel primo punto d'uso in un **altro** file (query)
contro i 200 vettori di definizione (valori). Recall@1 al caso = 1/200 =
**0,5 %**. Match whole-identifier, così che `kv` non venga cercato dentro
`kv_pages`.

| layer | recall@1 | rango medio | coseno corretto | coseno sbagliati | Δ |
|---|---|---|---|---|---|
| L2 | 0,011 | 93,4 | +0,928 | +0,928 | **0,000** |
| L5 | 0,011 | 86,2 | +0,951 | +0,951 | **0,000** |
| **L19** | **0,413** | **19,2** | +0,782 | +0,764 | **+0,018** |
| L33 | 0,228 | 32,4 | +0,643 | +0,615 | +0,028 |
| L47 | 0,217 | 45,6 | +0,521 | +0,485 | +0,036 |
| L61 | 0,228 | 44,3 | +0,390 | +0,342 | +0,048 |

n query = 92, n valori = 200.

Tre cose, in ordine di importanza:

1. **A L19 il recall@1 è 83× il caso.** Il punto d'uso di un simbolo "sa" quale
   definizione è la sua, con un margine grande. Questa è l'evidenza più forte
   che l'ipotesi del §3 non sia vuota.
2. **Sui layer superficiali non c'è niente.** A L2 e L5 il coseno col valore
   corretto è identico a quello coi 199 sbagliati alla terza cifra. Insieme al
   §2, questo chiude il layer 2: **il punto d'iniezione di Flash-Next non
   trasferisce a questo uso.** Là l'n-gram descrive il token corrente e a L2 il
   residual *è* ancora il token corrente; qui la chiave deve richiamare un
   riassunto, e a L2 quel riassunto non esiste ancora.
3. **Il margine assoluto è piccolo, e cresce con la profondità mentre il
   recall cala.** A L19 il corretto batte gli sbagliati di +0,018 su un
   fondo di 0,78; a L61 il margine è +0,048 su un fondo di 0,39, ma il recall
   è la metà. Cioè: **la discriminazione sta nell'ordinamento, non in una
   soglia assoluta.** È una cattiva notizia per il gate non addestrato del §6 —
   una soglia globale sul coseno taglia i match buoni quasi quanto i cattivi.
   L'asse `soglia coseno` dello sweep va letto aspettandosi poco.

**Conclusione di Fase 0.** Si procede, e lo sweep dei layer va spostato: il
piano proponeva L ∈ {2, ~19/33, ~47}, e **L19 è il candidato migliore su
entrambi i test**. L2 resta nello sweep come controllo negativo dichiarato,
non come speranza.

---

## 4. Fase 1 — l'indice

Una sola passata sul corpus produce tutte e sei le varianti (L, P): il forward
è la parte cara e non dipende da quale tap si legge, quindi costruirle
separatamente rifarebbe gli stessi 659 forward sei volte. Il set di chiavi è
condiviso per costruzione, ed è anche ciò che rende confrontabili gli assi
dello sweep.

| | |
|---|---|
| righe (chiavi distinte indicizzate) | **3.345** |
| nomi scartati perché ambigui (definiti in 2+ posti) | 1.149 |
| nomi scartati (file oltre il budget di token) | 2.945 |
| file held-out | 20 |
| tempo di costruzione, 6 varianti | 11,8 min |
| dimensione per variante | 19,7 MB (3.345 × 5120 × fp16) |

I file held-out non sono campionati ma **scelti**: quelli che usano più simboli
definiti altrove, che è il caso d'uso sotto esame. In testa:
`crates/server/tests/vision_mixed_load_gpu.rs` (217 simboli esterni),
`crates/server/tests/vision_dflash2_gpu.rs` (213),
`kernel/include/ignis_seq_internal.h` (213), `crates/server/src/main.rs` (206).

### Il budget di token è un muro, non una preferenza

| max_tokens | file coperti | forward | picco VRAM |
|---|---|---|---|
| 4096 | 581/801 | **4,11 s** | 20,58 GiB |
| 8192 | 706/801 | **78,85 s** | 32,00 GiB |

A 8192 il picco tocca esattamente la capacità della card e WDDM pagina: **19×
più lento**. L'indice è costruito a 8192 (il tap non calcola `lm_head` e sta
sotto), ma **lo sweep gira a 4096**, quindi i file held-out più lunghi vengono
troncati. La troncatura è identica fra baseline e iniezione, quindi il ΔNLL
resta valido; si perde copertura, non correttezza.

---

## 5. Fasi 2-3 — lo sweep, e un errore nella forma dell'iniezione

**Il controllo α = 0 passa**: con l'hook forzato a girare, l'iniezione a
ampiezza zero riproduce il baseline **bit per bit** su tutte le varianti. Il
path di iniezione non sporca niente.

Lo sweep nella forma del §7 — `residual += α·‖h_t‖·v/‖v‖` con `v` la riga
grezza dell'indice — dà, sulla metrica primaria (ΔNLL sulla finestra di 8
token dopo il match, % contro baseline, tasso di match 3,76 %):

| variante | α=0,1 | α=0,3 | α=1 | α=1, τ mediana | α=1, τ alta |
|---|---|---|---|---|---|
| L19/last | +0,022 | +0,045 | +0,668 | +0,228 | +0,089 |
| L19/mean | +0,048 | +0,124 | +0,624 | +0,232 | +0,003 |
| L47/last | **−0,053** | +0,069 | +4,940 | +1,950 | +0,291 |
| L47/mean | −0,018 | +0,110 | +2,670 | +0,631 | +0,006 |

Danno quasi ovunque, monotono in α. Ma la forma è sbagliata, e il §2 di questo
documento dice perché.

### Quello che si stava davvero iniettando

Misurato sulle 3.345 righe dell'indice:

| variante | coseno grezzo medio | ‖μ‖ / ‖v‖ medio | ‖v−μ‖ / ‖v‖ medio | coseno centrato |
|---|---|---|---|---|
| L2/last | 0,9865 | 0,9935 | **0,123** | +0,031 |
| L19/last | 0,9080 | 0,9530 | **0,308** | +0,002 |
| L19/mean | 0,9748 | 0,9872 | **0,155** | +0,003 |
| L47/last | 0,6242 | 0,7889 | **0,609** | +0,001 |

μ = media delle righe dell'indice.

Normalizzare la riga **grezza** significa che a L19/last il 95 % di ciò che si
somma è la direzione condivisa del layer — quella delle attivazioni massive
già presente in `h_t` — e solo il 31 % è la parte che distingue un simbolo
dall'altro. A L2 la parte distintiva è il 12 %. **α non stava scalando il
simbolo: stava amplificando ciò che il residual aveva già.** Il danno monotono
in α, identico fra `last` e `mean`, e i coseni tutti stretti fra 0,77 e 0,89,
sono tutti coerenti con questo e con nient'altro.

Che le righe centrate abbiano coseno medio ≈ 0 dice che l'informazione
specifica del simbolo c'è ed è quasi ortogonale fra simboli. Era nascosta sotto
la direzione comune.

**Departure dichiarata rispetto al §7**: l'iniezione diventa

```
residual += α · ‖h_t‖ · (v − μ) / ‖v − μ‖
```

e il coseno del gate si calcola anch'esso su vettori centrati. Non è un cambio
di idea: è il §7 applicato dopo aver scoperto, in Fase 0a, che il coseno
grezzo in un residual stream non misura somiglianza. La formula del §7 è stata
scritta prima di quella scoperta.

Il controllo negativo che decide tutto è **la stessa iniezione con
l'accoppiamento chiave→riga permutato**: stesse chiavi, stesse posizioni,
stesse norme, solo la riga sbagliata. Nella forma grezza avrebbe riprodotto
quasi esattamente il danno della forma corretta — e si sarebbe letto come "la
chiave non porta informazione" quando invece confermava l'artefatto. Va
eseguito solo sulla forma centrata.

_Run centrate in esecuzione._

---

## Nota di metodo: due errori di misura, entrambi costosi

Vale la pena registrarli perché non sono specifici di questo esperimento.

**Il coseno grezzo nel residual stream non misura somiglianza.** A L2 due
posizioni *qualsiasi* hanno coseno 0,995. Senza centratura sulla media
dell'insieme e senza un gruppo di controllo, la tabella del §2 avrebbe detto
"tutto collassato" a ogni layer, e l'esperimento si sarebbe fermato sul
go/no-go per un artefatto.

**Su Windows la VRAM si perde per frammentazione, non per dimensione.** Il
working set reale è ~19 GiB, ma con file di lunghezza variabile la
prenotazione dell'allocatore derivava fino a 32,1 GiB su 32,6, e a quel punto
WDDM inizia a paginare: lo stesso forward passa da 1,2 s a ~10 s, un fattore
~10 senza nessun errore visibile. `expandable_segments`, che sarebbe la cura,
**non è supportato su Windows** (`UserWarning` esplicito da PyTorch). La cura
che funziona è `torch.cuda.empty_cache()` dopo ogni file.

Un contributo separato allo stesso problema: chiamare il modello wrapper
invece dello stack di linguaggio calcola `lm_head` su tutte le posizioni —
4096 × 248320 = 2 GB di logit allocati e buttati a ogni forward, quando di
logit non se ne legge nemmeno uno. Da solo, quel cambio porta la VRAM da 31,9
a 18,8 GiB.

---

## Nota sulla riproducibilità

Nessun file Rust è stato toccato: la build del workspace e `cargo test` non
sono coinvolti. Tutto il codice di questo studio è Python sotto
`scripts/`, e gira nel venv `F:/ai/ngram-venv` (fuori dal repo, non
committato). I test del parser e dell'harness girano senza GPU:

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest -q
```
