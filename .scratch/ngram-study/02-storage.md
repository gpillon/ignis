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

### Fase 0 — pre-test di struttura (economico, decide se vale la pena)

**Costa poche ore e può far risparmiare giorni di lavoro sul leaf.** Non
richiede nessuna iniezione: solo leggere hidden.

Domanda: gli stati nascosti hanno la struttura che l'idea presuppone?

Procedura:
1. Scegliere ~100 simboli del repo con definizione e almeno 3 usi altrove.
2. Estrarre, a layer L, l'hidden di fine-definizione di ciascun simbolo
   (**il valore**) e l'hidden nei punti d'uso (**la query**).
3. Misurare, per ogni simbolo, il coseno query↔valore corretto contro il
   coseno query↔valore di 99 simboli sbagliati.

**Criterio**: se il valore corretto non è sistematicamente più vicino del
casuale (es. recall@1 non significativamente sopra 1/100), l'iniezione non può
funzionare a quel layer, e non vale la pena scrivere il percorso di iniezione.
Spazzare L prima di concludere: la struttura potrebbe esserci solo in
profondità.

**Se la Fase 0 fallisce a ogni L, l'esperimento si ferma qui ed è comunque un
finding.**

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
- se match **e** la chiave supera il filtro di selettività → sommare α · riga
  al residual a layer L;
- registrare la NLL per token.

Confronto: stessa sequenza, stesse condizioni, iniezione **on** contro **off**.

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

Ripetere la misura della Fase 2 su un corpus **estraneo** al repo (prosa, o
codice di un altro progetto), con lo stesso indice attivo. Qui la NLL **non
deve salire**: è la misura del danno.

### Criteri, dichiarati prima di misurare

| esito | condizione | conclusione |
|---|---|---|
| **successo** | ΔNLL in-dominio ≤ −2 % **e** ΔNLL OOD ≤ +0,5 % | c'è segnale senza addestrare; si passa al percorso veloce (`01-strade.md`) |
| **gate necessario** | esiste α con guadagno in-dominio, ma ΔNLL OOD > +2 % | quantificato quanto deve fare il gate; diventa un lavoro di training |
| **nessun segnale** | nessun α/L/P dà ΔNLL in-dominio < −0,5 % | l'ipotesi del §3 è falsa; finding, e si chiude |

Riportare sempre anche il **tasso di match** (quale frazione di token riceve
un'iniezione): un ΔNLL piccolo con tasso di match dell'1 % significa una cosa
molto diversa da un ΔNLL piccolo con tasso del 40 %.

---

## 8. Cosa c'è già nel repo, e cosa va scritto

**C'è già:**
- la pipeline di prefill che produce gli hidden;
- **i tap per layer**: DFlash2 li fa già su `[5,19,33,47,61]` per la sua
  `feature_projection` (`crates/core/src/speculation.rs`,
  `kernel/src/dflash2_drafter.h`) — il meccanismo esiste nel leaf;
- l'arena pinned host (ADR 0030);
- il precedente teacher-forced (ADR 0014).

**Va scritto (per l'esperimento, non per la produzione):**
- l'esposizione host di un tap a layer L su richiesta;
- la somma `residual += α · v` a layer L, dietro un flag, fuori dai graph;
- il costruttore d'indice e il caricatore;
- l'harness di misura NLL teacher-forced on/off.

**Non serve per l'esperimento** (solo dopo, se il segnale c'è): arena mappata,
kernel di gather UVA, indirizzi device-resident, cattura nel graph, endpoint
HTTP, indice inverso incrementale, plugin opencode.

## 9. Vincoli operativi

- **GPU in esclusiva**: `make gpu-status` prima di ogni run; la 5090 regge un
  run alla volta e il perdente muore senza diagnostica (ADR 0006,
  `docs/agents/testing.md`).
- **Ogni modifica di codice porta un test**, e `cargo test` deve passare
  workspace-wide prima di dirsi finito.
- **Mai `cargo fmt`**: il repo non è rustfmt-clean, i diff restano semantici.
- **Spazio su F:** controllare prima di una build fresca — le `target/` per
  worktree riempiono il disco.
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
