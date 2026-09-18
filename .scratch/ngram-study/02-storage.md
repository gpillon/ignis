# La codebase come memoria n-gram — rationale e disegno dell'esperimento

- Data: 2026-09-19
- Branch: `ngram-study`
- Dipende da: [`00-fatti.md`](00-fatti.md) (fonti primarie), [`01-strade.md`](01-strade.md) (le tre strade)
- Stato: disegno approvato dall'owner, **esperimento da eseguire**
- Esecutore previsto: sessione separata, **in questo stesso worktree**

Questo documento è autosufficiente: un agente che parte da zero qui dentro
deve poter eseguire l'esperimento leggendo solo questo file e i due che
precedono.

---

## 1. L'idea, in una riga

Indicizzare l'intera codebase in una tabella residente in RAM host, chiave =
nome del simbolo, valore = uno stato nascosto del modello stesso; a runtime,
quando il nome ricompare, sommare quel vettore nel residual. Retrieval a
**O(1)**, senza consumare contesto e senza appesantire i pesi.

## 2. Perché non il KV cache (l'argomento che regge tutto)

Ignis ha già il riuso esatto del KV: retained prefix, prompt checkpoint, spill
KV-RAM (ADR 0029, `crates/core/src/{prefix,checkpoint,retained_slot}.rs`). Per
**un file** quello è già la risposta giusta: esatta, senza training, senza
approssimazione. L'idea di questo documento non lo sostituisce.

Per **la codebase intera** non lo è, e non lo diventerà, per due ragioni
indipendenti:

1. **Non entra nel contesto.** 9,07 MB di sorgenti ≈ 2,7 M token BPE, contro
   una finestra da 262 k (preset `…-262k`). Un ordine di grandezza oltre.
2. **Anche se entrasse**, ogni singolo decode step dovrebbe attraversarla in
   attenzione: **O(len) per token generato**. Il lookup n-gram è **O(1)**.

Questa è la separazione strutturale tra i due meccanismi. Non è una questione
di quanti GB occupano.

## 3. Cosa cambia rispetto a Engram/PLE — dichiararlo, non subirlo

PLE, Engram e Memory Grafting memorizzano lo stato nascosto **dopo aver
consumato** l'n-gram: servono a rappresentare meglio *il token corrente*.

Quello che facciamo qui — chiave = nome del simbolo, valore = stato nascosto
alla **fine della definizione** — è un'altra funzione: **retrieval latente**.
RAG nello spazio del residual. Meccanica identica (chiave → riga → somma
gated), scopo diverso.

Ne segue che c'è un'**ipotesi sotto test, non un dato acquisito**:

> lo stato nascosto dell'ultimo token del corpo di una funzione è un riassunto
> utilizzabile di quel corpo?

Non è ovvio. In un modello causale l'ultimo hidden è addestrato a **predire il
token successivo** (`}` → `\n\n`), non a riassumere ciò che precede. Per
questo il pooling è un asse da spazzare, non una scelta da fare a priori.

## 4. I numeri, misurati su questo repo (2026-09-19)

Corpus: `*.rs *.cu *.h *.cuh *.cpp *.ts *.tsx`, esclusi `target/`,
`node_modules/`, `.git/`.

| | |
|---|---|
| file | 802 |
| byte | 9.071.351 |
| token, word-level (`\w+|[^\w\s]`) | 2.223.942 |
| token BPE (stimato, ~3,4 B/token) | ~2,7 M |
| 2-gram distinti | 240.992 (0,108/token) |
| 3-gram distinti | 579.715 (0,261/token) |
| **righe, n-gram scorrevole** | **820.707** |
| **simboli** | **11.760** |

Simboli contati con: `fn|struct|enum|trait|const|static|type` (Rust, 7.761),
funzioni top-level + `struct|typedef|enum` (C/CUDA, 2.422),
`function|const|class|interface|type` (TS/TSX, 1.574).

**Caveat sui conteggi n-gram**: sono word-level, non BPE. Gli identificatori si
spezzano in sotto-token, quindi gli n-gram distinti in BPE salgono più che
linearmente — assumere **±2×**. Se il tokenizer dell'artifact è raggiungibile
da `crates/artifact`, un conteggio reale sostituisce questo caveat.

### Storage, a riga = hidden del 27B (5120 × BF16 = 10.240 B)

| schema | questo repo | monorepo 100× |
|---|---|---|
| n-gram scorrevole | 8,4 GB (word-level) — 13-17 GB in BPE | **~1 TB** ❌ |
| **chiave = simbolo** | **120 MB** | **12 GB** ✅ |

**70× di differenza, e solo uno dei due scala.** Questo non favorisce
l'indicizzazione per simbolo: la impone. Il resto del documento assume
chiave = simbolo.

### Costo di costruzione

Prefill misurato: **0,0936 ms/token** (width 4096,
[`docs/findings/2026-09-11-prefill-chunk-wall-time.md`](../../docs/findings/2026-09-11-prefill-chunk-wall-time.md)).

- indice completo: 2,7 M token × 0,0936 ms ≈ **4,2 minuti**, una tantum
- un file modificato (~5 k token): **~0,5 s**
- copia tap → host: 11.760 righe × 10 KB = **120 MB** in totale. Irrilevante.

## 5. Decisioni di design già prese

Sono decisioni, non scoperte: l'esperimento non le rimette in discussione.

1. **Chiave = simbolo**, non n-gram scorrevole (§4: 70×).
2. **Occorrenze multiple → vince il sito di definizione.** `decode_round`
   compare una volta come definizione e N volte come call site, ognuno con un
   hidden diverso. Si indicizza la definizione.
3. **Una riga per simbolo.** Niente 8 teste di hash: su un corpus locale gli
   n-gram si enumerano, quindi **match esatto, zero collisioni**. Le 8 teste di
   Flash-Next esistono perché in pretraining sul web non puoi enumerare.
4. **Indicizzazione per file indipendente.** Gli hidden di un file dipendono
   solo da quel file → una modifica invalida e ricalcola solo le sue righe.
   Serve un indice inverso `file → row ids`.
5. **L'esperimento non usa il percorso veloce.** Per misurare il *segnale* non
   serve la performance: gather host-side, copia H2D, fuori dai CUDA graph,
   sync liberi. Il percorso veloce (arena mappata + gather UVA, §
   [`01-strade.md`](01-strade.md)) si costruisce **solo se il segnale c'è**.
   Tenere le due domande separate è la scelta più importante di questo piano.

## 6. Il nodo, e perché è il nodo

L'unica cosa che non è ingegneria ordinaria è **il gate**.

Con α = 1 si somma al residual un vettore a piena ampiezza proveniente da un
altro contesto. È esattamente il meccanismo di degrado che l'Engram Adapter ha
misurato: le baseline **always-on degradano nettamente**, mentre il gate
appreso è ciò che preserva il 99,4-100,1 % della performance fuori dominio
(`00-fatti.md`).

Il modello mentale "storage" è per costruzione always-on — n-gram matcha,
inietti. **Quindi la parte non gratis è precisamente quella che impedisce al
modello di peggiorare.**

Esiste però una via **non addestrata**, suggerita dal paper stesso
("occupancy tracking as a lightweight selectivity prior"):

- sparare solo su chiavi **rare e distintive** — gli identificatori lo sono per
  costruzione;
- solo se il coseno tra `h_t` e la riga supera una soglia;
- con **α piccolo**, 0,1-0,3, non 1.

Se basta, non si addestra niente. Se non basta, si è misurato quanto lavoro
deve fare il gate. **L'esperimento esiste per decidere fra questi due esiti.**

---

## 7. L'esperimento

### Fase 0 — pre-test di struttura (decide se vale la pena proseguire)

Non richiede nessuna iniezione: solo leggere hidden. Due test distinti, con
ruoli diversi — **non confonderli**.

**Test 0a — distintività dei valori. Questo è il go/no-go.**
Estrarre l'hidden di fine-definizione di ~100 simboli e guardare se sono
*sparsi* o *collassati*. Il timore concreto: in un modello causale l'ultimo
token di un corpo (`}`) è addestrato a predire `\n\n`, quindi i 100 hidden
potrebbero essere quasi lo stesso vettore. Misura: coseno medio a coppie, e
varianza spiegata dalla prima componente principale. Se sono collassati, non
c'è niente da recuperare e **l'esperimento finisce qui** — provando prima
l'altro pooling (media sul corpo), che è la via di scampo naturale.

**Test 0b — associatività query↔valore. Questo testa il gate, non l'idea.**
Per ogni simbolo, coseno fra l'hidden nei punti d'uso (**query**) e il valore
corretto, contro il coseno con 99 valori sbagliati. Un recall@1 alto dice che
un gate basato sul coseno può discriminare. Un recall@1 basso **non** uccide
l'idea: il match è già esatto sulla chiave, il coseno serve solo a decidere
*se* iniettare.

Spazzare L in entrambi prima di concludere: la struttura potrebbe esserci solo
in profondità.

**Se 0a fallisce a ogni L e con entrambi i pooling, l'esperimento si ferma ed
è comunque un finding.**

### Fase 1 — costruzione dell'indice

- **Held-out**: escludere dall'indice ~20 file, scelti fra quelli che *usano*
  molti simboli definiti altrove (il caso d'uso è "il modello scrive codice che
  chiama roba che non ha nel contesto"). Registrare l'elenco esatto.
- Per ogni simbolo dei file indicizzati: chiave = i token BPE del nome; valore
  = hidden a layer L con pooling P.
- Formato su disco: un `.bin` di righe contigue + un `.json` con
  `{chiave → row id}`, `{file → [row ids]}`, e il manifest dei parametri
  (L, P, commit del repo, identità del modello).
- Verifica di sanità: ricostruire l'indice due volte deve dare byte identici.

### Fase 2 — iniezione e misura

**Metrica: NLL teacher-forced.** Niente generazione, niente giudizio umano —
un numero. Il repo ha già un precedente teacher-forced (ADR 0014, canary G1).

Per ogni token della sequenza di valutazione:
- calcolare le chiavi (n-gram scorrevole sui token BPE) e cercare match esatto;
- iniettare **sull'ultimo token del nome** che ha fatto match, cioè quando
  l'identificatore è completo;
- se match **e** la chiave supera il filtro di selettività → sommare al
  residual a layer L;
- registrare la NLL per token.

Confronto: stessa sequenza, stesse condizioni, iniezione **on** contro **off**.

#### Normalizzare prima di applicare α

`residual += α · v` **non è confrontabile lungo gli assi dello sweep**: un `v`
ottenuto per media ha una frazione della norma di un singolo hidden, e le norme
del residual crescono con la profondità. α = 0,3 a L = 2 con pooling
"ultimo token" e α = 0,3 a L = 47 con pooling "media" sarebbero due
perturbazioni diverse, e lo sweep misurerebbe la norma invece dell'effetto.

Scalare sempre alla norma locale:

```
residual += α · ‖h_t‖ · v / ‖v‖
```

Così α è una frazione dichiarata dell'ampiezza del residual in quel punto, e i
valori sono comparabili fra layer e fra pooling.

#### Dove si misura il ΔNLL

L'iniezione avviene sul nome; l'effetto ricade sui token **che seguono**. Un
ΔNLL sull'intero file diluisce un effetto locale forte su tutto ciò che non ha
avuto match.

- **primaria**: ΔNLL sulla finestra post-match, i successivi N token, con
  N ∈ {8, 32};
- **secondaria**: ΔNLL sull'intero file;
- intervallo di confidenza bootstrap sui file held-out.

I criteri del §7 si applicano alla **primaria**.

### Fase 3 — lo sweep

Quattro assi, in quest'ordine di priorità:

| asse | valori |
|---|---|
| **layer L** | shallow (2), medio (~19 o 33), profondo (~47) — riusare i tap DFlash2 `[5,19,33,47,61]` se comodo |
| **pooling P** | ultimo token della definizione · media sul corpo |
| **α** | 0 (controllo) · 0,1 · 0,3 · 1,0 |
| **soglia coseno** | nessuna · mediana · alta |

α = 0 deve riprodurre esattamente la baseline: è il controllo che verifica che
il percorso di iniezione non stia già sporcando qualcosa.

### Fase 4 — fuori dominio

Ripetere la misura della Fase 2 con lo stesso indice attivo su un corpus
estraneo. Qui la NLL **non deve salire**: è la misura del danno.

**Il corpus OOD dev'essere codice di un altro progetto, non prosa.** Sulla
prosa il tasso di match è ~0, il ΔNLL esce ~0 per costruzione, e si ottiene un
falso via libera. La perdita vera è un'altra: nomi di simbolo che sono parole
comuni del codice — `step`, `host`, `request`, `type`, `kv`, `next` — che su
codice estraneo matchano di continuo e iniettano hidden specifici di ignis.

Per questo "chiavi rare e distintive" va reso operativo, non lasciato
all'intuizione:

- chiave di **almeno 2 token BPE**, e/o
- chiave **unica** nel corpus indicizzato (nessun altro simbolo la contiene).

**Riportare sempre il tasso di match OOD accanto al ΔNLL OOD**: un ΔNLL OOD
buono con tasso di match dell'1 % non dimostra niente sulla selettività.

### Criteri, dichiarati prima di misurare

Tutte le soglie si riferiscono alla **metrica primaria** (ΔNLL sulla finestra
post-match, §Fase 2), non al ΔNLL sull'intero file.

| esito | condizione | conclusione |
|---|---|---|
| **successo** | ΔNLL in-dominio ≤ −2 % **e** ΔNLL OOD ≤ +0,5 % | c'è segnale senza addestrare; si passa al percorso veloce (`01-strade.md`) |
| **gate necessario** | esiste α con guadagno in-dominio, ma ΔNLL OOD > +2 % | quantificato quanto deve fare il gate; diventa un lavoro di training |
| **nessun segnale** | nessun α/L/P dà ΔNLL in-dominio < −0,5 % | l'ipotesi del §3 è falsa; finding, e si chiude |

Riportare sempre anche il **tasso di match**, in dominio e fuori (quale
frazione di token riceve un'iniezione): un ΔNLL piccolo con tasso di match
dell'1 % significa una cosa molto diversa da un ΔNLL piccolo con tasso del
40 %. Senza il tasso di match, i numeri di ΔNLL non sono interpretabili.

---

## 8. Il veicolo: PyTorch, non il leaf

**Questa è la decisione che fa risparmiare più tempo, e va presa per prima.**

Le Fasi 0-4 hanno bisogno **del modello**, non di ignis. In PyTorch:
`output_hidden_states=True` regala la Fase 0, e un forward hook che somma
`α·‖h‖·v/‖v‖` all'uscita del layer L è ~50 righe. In ignis servirebbero invece
un nuovo ABI tap→host, un kernel di iniezione dietro un flag, il plumbing Rust
e i test relativi: giorni, non ore, e tutti spesi **prima** di sapere se il
segnale esiste.

I pesi sono su HF: `conversion.json` fissa `Qwen/Qwen3.8-27B` (revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) e `unsloth/Qwen3.8-27B-NVFP4`.

**Quindi: Python per il segnale, ignis intatto finché il segnale non è
dimostrato.** Solo dopo, e solo se c'è, si porta nel leaf.

### Primo task dell'esecutore: verificare che il veicolo esista

L'ambiente **non è pronto**, ma l'owner ha detto esplicitamente che
aggiornarlo va bene. Stato al 2026-09-19:

| | stato | serve |
|---|---|---|
| torch | `2.13.0+cpu`, `cuda False` | build CUDA per Blackwell (sm_120) |
| transformers | `4.51.3` | versione che conosca l'architettura Qwen3.8 (linear/full attention alternata, MoE) |
| modello in locale | **assente** (`hf-repos/` vuoto, cache HF senza Qwen3.8) | da scaricare |
| **spazio su F:** | **48 GB liberi su 1,9 TB (98 % pieno)** | **questo non si aggiorna** |
| VRAM | 32,6 GiB | 27B a 4 bit ≈ 16 GiB: entra comodo |

Conseguenza dello spazio: **BF16 (~54 GB) è fuori discussione, NVFP4 (~16 GB)
è l'unica opzione che entra.** Da verificare che transformers sappia caricare
quel formato (passa da `compressed-tensors`); se non lo sa, il veicolo Python
cade e si torna al leaf — ed è bene scoprirlo al primo task, non al terzo
giorno. Controllare comunque lo spazio prima di scaricare, e **chiedere
all'owner prima di cancellare `target/` di altri worktree**.

### Scorciatoia consigliata: mettere a punto su un modello piccolo

Il Test 0a — "gli hidden di fine-definizione sono distinti o collassati?" — è
una proprietà generale dei modelli causali, non specifica del 27B. Girarlo
prima su un Qwen piccolo (2-4 B, pochi GB di disco, minuti di runtime) serve a
due cose: mettere a punto tutto il codice di estrazione e di misura a costo
quasi nullo, e ottenere un primo segnale. Se collassa anche lì, è
un'indicazione forte prima di spendere 16 GB di disco.

**Ma il risultato che conta è quello sul 27B**: il modello piccolo è un filtro
e un banco di prova, non la risposta.

### Cosa serve costruire, in ogni caso

- un **parser dell'estensione delle definizioni** (brace matching): serve a
  entrambi i pooling, ed è il pezzo di tooling più noioso;
- il costruttore d'indice (simbolo → riga) e il caricatore;
- l'harness NLL teacher-forced on/off con le finestre post-match.

### Se invece si finisce nel leaf

C'è già: la pipeline di prefill; **i tap per layer**, che DFlash2 fa già su
`[5,19,33,47,61]` per la sua `feature_projection`
(`crates/core/src/speculation.rs`, `kernel/src/dflash2_drafter.h`); l'arena
pinned host (ADR 0030); il precedente teacher-forced (ADR 0014).

Va scritto: esposizione host di un tap a layer L, somma al residual dietro un
flag e fuori dai graph, più indice e harness come sopra.

### Non serve, in nessuno dei due casi

Arena mappata, kernel di gather UVA, indirizzi device-resident, cattura nel
graph, endpoint HTTP, indice inverso incrementale, plugin opencode. Tutto
questo viene **dopo**, e solo se il segnale c'è.

## 9. Vincoli operativi

- **GPU in esclusiva**: `make gpu-status` prima di ogni run; la 5090 regge un
  run alla volta e il perdente muore senza diagnostica (ADR 0006,
  `docs/agents/testing.md`). Vale anche per uno script PyTorch: occupa la card
  come qualunque altra cosa.
- **Spazio su F: prima di scaricare qualsiasi cosa**: 48 GB liberi su 1,9 TB.
  Chiedere all'owner prima di cancellare `target/` di altri worktree.
- **Ogni modifica di codice porta un test**, e `cargo test` deve passare
  workspace-wide prima di dirsi finito.
- **Mai `cargo fmt`**: il repo non è rustfmt-clean, i diff restano semantici.
- **Ordine di lavoro**: veicolo → Fase 0 → Fasi 1-4. **Committare e pushare
  su `ngram-study` alla fine di ogni fase**, non solo alla fine: è il canale
  attraverso cui la sessione di studio rilegge i risultati.
- **`.nsys-rep` non si committa**: le catture del profiler portano l'ambiente
  del processo, quindi le API key.

## 10. Dove vanno i risultati

- output grezzi, log, csv → `.scratch/ngram-study/results/`
- sintesi leggibile → `.scratch/ngram-study/03-risultati.md`, con: parametri
  esatti, tabella dello sweep, tasso di match, e l'esito secondo i criteri del
  §7
- se il risultato è durevole e riusabile → promuoverlo in `docs/findings/`
  seguendo `docs/agents/findings.md` (kind `experiment`), **con la riga
  nell'indice nello stesso commit**
- **committare sul branch `ngram-study` e pushare**: è il canale attraverso cui
  la sessione di studio rilegge i risultati
