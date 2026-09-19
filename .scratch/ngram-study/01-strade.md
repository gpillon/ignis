# Le tre strade, e quale è misurabile oggi

- Data: 2026-09-18
- Dipende da: [`00-fatti.md`](00-fatti.md)
- Stato: **superato il 2026-09-19 per la parte operativa** — vedi
  [`03-risultati.md`](03-risultati.md) e
  [il finding](../../docs/findings/2026-09-19-codebase-as-ngram-memory.md)

> **Il microbenchmark proposto qui sotto non va eseguito, e il percorso veloce
> non va costruito.** La sua premessa era che il segnale ci fosse: l'esito
> misurato è *nessun segnale*. Non è stato confutato — è rimasto senza scopo.
>
> Portata precisa della chiusura: cade la **strada C** nella forma testata,
> cioè un retrofit **additivo e non addestrato** — l'unica che l'hardware di
> questa macchina consentiva. La **strada A** (servire Flash-Next) resta
> bloccata dall'hardware, come qui descritto. La **strada B** (adapter
> addestrato) resta aperta, e ora ha il suo primo dato: l'informazione è
> recuperabile, a L19 su 64, con recall@1 0,413 contro 0,005 del caso.
>
> Resta valida senza riserve l'analisi dei vincoli (§ hardware, § ninfer, §
> config), che è il motivo per cui questo documento non viene riscritto.

## A — Servire Qwen3.8-Flash-Next in ignis

Il modello ha l'n-gram nativo: niente da addestrare, i pesi esistono.

**Bloccata dall'hardware, non dal software.**

| | serve | c'è |
|---|---|---|
| pesi backbone 125 B | ~62 GiB a NVFP4 | 32,6 GiB VRAM |
| tabella n-gram | 95,4 GiB BF16 / ~24 GiB NVFP4 | 63,8 GiB RAM, di cui una fetta già impegnata dal tier KV host |

Anche quantizzando tutto al massimo, i pesi del backbone non entrano in VRAM
senza expert offload — che consuma la stessa RAM host che serve alla tabella.
E sarebbe comunque un port di architettura nuova (Gated DeltaNet, Qwen Sparse
Attention, MoE, MTP) in cui **l'n-gram è il pezzo più piccolo**. Il lavoro
sarebbe il backbone, non la memoria condizionale.

Non raccomandata su questo box.

## B — Innestare una tabella n-gram su Qwen3.8-27B

Il meccanismo di innesto esiste già: `tools/artifact/graft_dflash2_module.py`
ha aggiunto 66 oggetti al `.ninfer` v1 lasciando i 1.259 di base
bit-identici. Un modulo n-gram sarebbe lo stesso gesto.

**Il blocco non è il formato, è che i pesi non esistono.** Vanno addestrati
(ricetta Engram Adapter, backbone congelato). Il training non è il mestiere di
ignis, e né compute né volume dati sono dichiarati nell'abstract del paper.

Resta un'opzione reale ma di un altro progetto. Se interessa, il primo passo è
leggere il PDF completo dell'Engram Adapter per il costo di training, non
scrivere codice in ignis.

## C — Il substrato di serving

**Questa è la parte che il repo possiede davvero, ed è misurabile oggi senza
un solo peso addestrato.**

La meccanica è identica in A e in B: dagli ultimi 2-3 token ricavi 16 indirizzi
deterministici, peschi 16 righe da una tabella in RAM host, le fondi nel
residual a layer 2. Se quel gesto costa poco su questa macchina, sia A sia B
diventano discorsi sensati; se costa molto, nessuno dei due lo è.

### La domanda che decide tutto

> La gather si nasconde dentro il decode round di ignis, che è catturato in un
> CUDA graph da 1.166 nodi con solo 0,80 ms di idle su 15,81 ms?

SGLang misura −0,07 % su H200/Linux e **non discute affatto l'interazione con i
CUDA graph**. Ignis cattura il round. Il numero di SGLang non trasferisce per
ipotesi.

### I sette nodi concreti

1. **Gli indirizzi devono nascere sul device.** In decode il token appena
   campionato vive in VRAM; l'host non lo conosce fino al sync. L'hash
   (multiplicative-XOR su 2 e 3 token) va calcolato in un kernel, non
   sull'host. Così non aggiunge sync e resta catturabile.
2. **La memoria pinned di ignis non è dichiarata accessibile da un kernel.**
   L'unico `cudaHostAlloc` del leaf è in `kernel/vendor/src/core/arena.cu:279`
   e usa `cudaHostAllocDefault`: page-locked per DMA veloce. È
   `cudaHostAllocMapped` il flag che *garantisce* per contratto la mappatura
   nello spazio di indirizzi del device; UVA dà l'uguaglianza dei puntatori,
   non la mappatura. Se serva davvero il flag su questo driver è da
   confermare — è il **passo 0 del benchmark**. In ogni caso la conclusione di
   design non cambia: una seconda arena mappata, senza toccare la vendored
   (nessun conflitto con ADR 0010).
3. **Il costo in nodi di graph è trascurabile.** fork + kernel hash + kernel
   gather + join ≈ 4 nodi su 1.166, a 0,36-0,73 µs/nodo ≈ 2-3 µs di
   submission.
4. **La finestra di overlap è ampia.** 15,81 ms su ~64 layer ≈ 0,25 ms per
   layer; fino all'iniezione a layer 2 ci sono ~0,5 ms. Va nascosta la latenza
   di 16 letture random su PCIe, non una banda.
5. **Lo stato per sequenza è minuscolo**: i due token id precedenti. Va nel
   seq pool per slot, esattamente dove sta già la finestra del drafter
   (precedente: GitHub #152).
6. **Convivenza con DFlash2.** SGLang tiene il lookup attivo in prefill,
   decode e verifica del target, e lo toglie solo nel draft. Tradotto per
   ignis: il verify round fa la gather per M+1 posizioni per slot (indirizzi
   derivati dai draft, quindi anch'essi device-resident), il drafter no.
7. **Il budget RAM è condiviso** con il tier KV host, che ADR 0030 riserva al
   load. Una tabella e una KV-RAM arena competono per gli stessi 63,8 GiB.

### Il rischio specifico di questa macchina

Windows/WDDM. Il repo ha già visto WDDM paginare sotto pressione (lavoro
#207/ADR 0030). Lo zero-copy da kernel verso host memory mappata è la
primitiva più esposta a quel comportamento, e su Windows non ha lo stesso
profilo che su Linux. È precisamente ciò che una misura risolve e una
discussione no.

## Proposta: il microbenchmark, prima di qualsiasi design

Standalone, sul modello di `graphlaunch_bench.cu` già usato per l'anatomia del
round. Nessun modello caricato, nessun peso, nessuna dipendenza da A o B.

**Passo 0** — verificare se un kernel legge un'arena `cudaHostAllocDefault` su
questo driver, o se serve `cudaHostAllocMapped`. Decide la forma dell'arena
prima di misurare qualsiasi latenza.

**Setup**
- arena host mappata di dimensione variabile (8 / 16 / 24 GiB), riempita di
  righe sintetiche;
- 16 row id casuali per token, già in VRAM (niente sync);
- kernel di gather UVA che scrive le 16 righe in un buffer device.

**Misure**, al variare di quattro parametri
- **dimensione della tabella** (8 / 16 / 24 GiB): isola l'effetto TLB/pagine
  su un working set che non entra in nessuna cache;
- **larghezza della riga**: 160 è la scelta di Flash-Next, ma un retrofit sul
  27B (hidden 5120) sceglierebbe la propria — a 1280 per tabella, come Engram,
  la gather per token è 40 KB invece di 5 KB. Va spazzata, non assunta;
- **larghezza del batch** (1 lane vs N lane): gli accessi random si
  ammortizzano o si sommano?
- **formato riga** (BF16 / FP8 / NVFP4): si vede se quantizzare la tabella
  compra latenza oltre che spazio.

Per ciascuna combinazione: latenza del kernel isolato, e quanto ne resta
scoperto sotto un carico fittizio di ~0,5 ms su un altro stream, dentro e
fuori un graph catturato.

**Esito, in un verso o nell'altro, è un finding**
- si nasconde → il −0,07 % di SGLang è plausibile anche qui, e A/B diventano
  valutabili nel merito;
- non si nasconde → si sa il perché e quanto, e si è risparmiato un port.

Costo stimato: una sessione. Richiede la card in esclusiva (ADR 0006), quindi
`make gpu-status` prima.

## Raccomandazione

**C prima**, poi si decide. È l'unico dei tre che produce un numero invece di
una discussione, non richiede pesi che non esistono, e il suo risultato è la
premessa di entrambi gli altri.
