# VRAM ignis vs ninfer: perché ignis cresce e va in "Shared GPU memory"

2026-09-17, branch `issue-191` @ 726a1b7, build release del 17/09 01:17.

Fonti:
- analisi del codice di ignis e di `F:/ai/q38/ninfer` @ a00648cb (`gpillon/coding`)
- una misura live di ignis
- per ninfer, nessuna esecuzione: i numeri vengono dal codice e dal record
  `server_start` in `F:/ai/q38/logs/traces-dflash2_20260903_033450.jsonl`
  (preset `qwen3.8-27b-nvfp4full-ram-dflash2-262k.bat`)

Unità: **GiB/MiB binari**, come Task Manager (che però scrive "GB"). Il log
di ninfer usa GB decimali; qui sono convertiti. La scheda ha 32607 MiB, cioè
31.84 GiB.

## Conclusione

**Non è un leak. ignis e ninfer allocano la memoria in modo diverso.**

1. **Al load pesano uguale.**
   - ignis con la configurazione di `make`: 27.65 GiB.
   - ninfer (somma delle sue riserve più il contesto CUDA): circa 27.5 GiB.
   - In ignis la vision costa circa 1.25 GiB in più (workspace sommato allo scratch invece che condiviso), ma il pool KV è più
     piccolo e le due cose si compensano.
   - **Tutta la differenza che vedi nasce dopo il load.**
2. **ninfer riserva tutto allo start e non chiede più memoria al driver.**
   - `request_memory.h:15-17` lo vieta esplicitamente.
   - Checkpoint e stato salvato occupano slot fissi (2 × max_concurrency)
     nel blocco sequenze.
   - La KV-RAM (`--kv-ram-capacity`) è un solo `cudaHostAlloc` pinned da
     16 GiB (preset dflash2) oppure 8 GiB (preset MTP), fatto allo start.
   - Per questo vedi una dedicata fissa e una shared fissa.
   - *Da verificare:* la shared di ninfer in Task Manager dovrebbe valere
     circa 16 GB con il preset dflash2 e circa 8 GB con quello MTP.
3. **ignis alloca durante le richieste.**
   - **Prefix publish:** un `cudaMalloc` per la `clone_image`
     (`kernel/src/seq_prefix.cu:207`), circa 148 MiB di stato GDN più 80 MiB
     con dflash2, cioè ~228 MiB.
   - **Checkpoint capture:** un `cudaMalloc` per `image` e `tail_page`
     (`kernel/src/seq_checkpoint.cu:244-246`).
   - **Evict o spill su KV-RAM:** un `cudaHostAlloc` pinned della taglia del
     blob (`kernel/src/seq.cu:901`).
4. **La tua ipotesi sui checkpoint è giusta a metà.**
   - **I checkpoint hanno un tetto.** Il retained pool vale
     `(free − 1 GiB)/2` (`crates/core/src/checkpoint.rs:1153`), cioè 710 MiB
     in questa run: al massimo circa 3 immagini.
   - **I retained prefix e le catene di prefix (#187) non hanno tetto.**
     - Nessun budget li conta.
     - Li libera solo una carenza di pagine KV
       (`crates/core/src/concrete.rs:1038-1071`).
     - Il pool KV è da 4 GiB, quindi la carenza arriva tardi.
     - Ne nasce uno per ogni blocco system+tools e uno per ogni turno che
       riparte da un checkpoint.
5. **Su Windows/WDDM un `cudaMalloc` oltre la VRAM fisica non fallisce.**
   - Il driver sposta allocazioni nella RAM di sistema ("Shared GPU memory")
     e le riporta indietro quando servono.
   - `cudaMemGetInfo` non se ne accorge.
   - L'effetto è paging: TTFT lento e irregolare, lo stesso di #204.

## Misura live di ignis (run 2)

**Configurazione:** `make start` di default: 262K, hq-e8-2b, dflash2/7,
`--vision`, `--prompt-reuse on`, `--kv-host-pool-bytes 8G`.

**Carico:**
- replay della trace #191 (157 richieste, `--max-gap 10`)
- in parallelo, 4 conversazioni `@agent` multi-turn con storia crescente
  (prompt da 50K a 87K token)

**Campionamento:** ogni 3 s, contatori WDDM per processo (gli stessi di Task
Manager), in `ignis-run2-mem.final.csv`.

**Durata:** fermata dopo circa 14 minuti, con 55 richieste completate.

| Momento | Dedicata processo | Shared processo | Commit processo | Dedicata adapter (desktop incluso) |
|---|---:|---:|---:|---:|
| prima del load (solo desktop) | – | – | – | 1.60 GiB |
| subito dopo il load | 27.65 GiB | 74 MiB | 27.72 GiB | 29.24 GiB |
| dopo la 1ª richiesta (\*) | 28.32 GiB | 74 MiB | 28.39 GiB | 29.26 GiB |
| 01:45–01:48 | 29.1–29.8 GiB | 76–460 MiB | 29.51 → 30.21 GiB | 30.8–31.2 GiB |
| 01:57 (picco del commit) | 30.24 GiB | 652 MiB | **30.88 GiB** | 31.25 GiB |
| 01:58:37, evict su KV-RAM | 29.22 GiB | **976 MiB** | 30.17 GiB | 30.64 GiB, shared adapter con picco a 2.04 GiB |

(\*) I +670 MiB della prima richiesta sono in parte una tantum: il lazy init
CUDA al primo uso, più il primo prefix e il primo checkpoint. Quel pezzo non
cresce più.

**Crescita del commit dopo il load:** +3.16 GiB in 14 minuti. Stima, non
misurata evento per evento (a livello INFO il log non emette publish o
capture): 3 checkpoint più circa 10 prefix e link di catena da ~228 MiB
fanno ~3 GiB, compatibile con la crescita.

**Paging:**
- Desktop (1.6 GiB) più ignis superano 31.84 GiB poco dopo le 01:45.
- In 143 campioni su 167 lo shared del processo supera 200 MiB.
- Diverse volte **il commit resta fermo mentre dedicata e shared si
  scambiano**, per esempio 30477/460 → 30860/76 MiB con commit a 30937 MiB.
- Sono le stesse allocazioni che WDDM sfratta e riporta: è paging, non
  memoria nuova.

**Startup** (`ignis-run2-server.log`):
- `kv_pool` 4.0 GiB (7281 pagine)
- `vision_reserved_bytes` 2.38 GiB, in un'arena separata
- `retained_pool.free_vram_bytes` 2.39 GiB e `budget_bytes` 710 MiB

## Confronto delle riserve

| Voce | ninfer (preset dflash2, 550K token, 6 lane) | ignis (make default, 262K) |
|---|---|---|
| Pesi | 17.31 GiB | ~17.3 GiB, più 282 MiB di pesi vision |
| KV | 5.39 GiB di payload per 550K token, dentro un blocco sequenze da 7.28 GiB | pool da 4.0 GiB |
| Workspace | 2.07 GiB, uno solo: `max(prefill, round, dflash, vision_encode)` | scratch di prefill **più** un'arena vision separata da 2.38 GiB (**somma**, non max) |
| Checkpoint e stato salvato | slot fissi nel blocco sequenze | `cudaMalloc` per checkpoint e per publish, prefix senza tetto |
| KV-RAM | un `cudaHostAlloc` da 16 GiB allo start | un `cudaHostAlloc` per blob, a runtime |
| Libera dopo lo start | 2.53 GiB, che non tocca più | 2.39 GiB, poi erosa dalle richieste |

## Bug trovato durante la misura (separato dalla VRAM, blocca il gate #191)

**Causa.**
- Tutte le 157 richieste della trace #191 hanno **due messaggi `system`
  consecutivi** agli indici 0 e 1.
- Il secondo è quello iniettato dall'hook ("CAVEMAN MODE ACTIVE…").
- In ignis il template fallisce con "System message must be at the
  beginning".
- A quel punto `crates/server/src/artifact_template.rs:180` restituisce
  `RenderedPrompt::default()`, cioè un prompt vuoto, invece di un 400.

**Effetto.**
- La richiesta viene ammessa con 0 token.
- Il decode entra in un loop caldo di `ignis.runtime.leaf_error`
  ("ignis_program_decode: sequence is null or was not prefilled").
- Risultato: 345K righe e 105 MB di log in circa 60 s.
- Dati in `ignis-run1-*`.

**ninfer.** ninfer ha servito le stesse richieste quando la trace è stata
registrata. Il suo parser tiene i system consecutivi come turni separati e
ordinati (`docs/serving.md:417-419`, `tests/test_anthropic_schema.cpp:192-205`,
lato Anthropic). Sul path OpenAI (verificato dopo) ninfer non usa il template
jinja: rende il secondo system come blocco `<|im_start|>system` separato al
suo posto, e il prefisso condiviso finisce dopo il primo. **Conta per #191:**
il prefisso in token deve coincidere con quello di ninfer, altrimenti il
matching del reuse cambia.

**Workaround per la run 2.** I system sono stati uniti con `\n\n` in
`trace-merged-system.jsonl` (24 MB, non committato: si rigenera unendo i
system della trace #191). Vale solo per questa misura di memoria, non per
un confronto con ninfer.

## Fix candidati (da decidere, niente implementato)

1. **Slab device fisso per le immagini di checkpoint e prefix.**
   - Dimensionato al load (N immagini) e contabilizzato, come il blocco
     sequenze di ninfer.
   - Esaurito lo slab, la vittima viene sfrattata o spillata invece di fare
     un altro `cudaMalloc`.
   - È il fix che toglie la crescita.
2. **KV-RAM come un'unica arena pinned allocata allo start.**
   - `HostPinnedArena` è già vendorizzata (`kernel/vendor/src/core/arena.cu`)
     ma ignis non la usa.
   - Porta la shared a un valore fisso e toglie `cudaHostAlloc` da ogni
     spill.
3. **Workspace vision dentro lo scratch di prefill (max, non somma).**
   - Risparmio circa **1.25 GiB**, cioè `min(scratch di prefill ~1.34 GB,
     workspace vision 2.22 GB)`: lo scratch è stimato dal codice, non
     loggato. L'output vision da 320 MiB resta separato.
   - Era già un follow-up di #177.
4. **Budget esplicito con headroom per il desktop**, invece di un retained
   pool derivato dalla VRAM libera letta una volta al load.
5. **Template.**
   - Un render o un tokenize fallito deve dare un 400, mai un prompt vuoto.
   - Rendere i system consecutivi come fa ninfer.

**Mitigazione immediata senza codice:** `make start VISION=0` libera circa
2.38 GiB (tutta la riserva vision) se non servono immagini. Allontana il paging ma non ferma la crescita
dei prefix.

## File

- `ignis-run2-mem.final.csv`: contatori WDDM in MiB, per adapter e per processo, più `nvidia-smi`
- `ignis-run2-server.log`: log del server della run 2
- `ignis-run2-parallel.jsonl`: le 4 conversazioni `@agent`
- `parallel_lane_load.py`: generatore di carico parallelo
- `ignis-run1-leaf-error-sample.log`, `ignis-run1-server.err`: il bug del template
- `gpumem.ps1`: il sampler (`powershell -ExecutionPolicy Bypass -File gpumem.ps1 -ProcName <exe> -Out <csv>`)

## Esito (2026-09-17)

Decisioni prese dopo il grilling: ADR 0030, spec
`.scratch/vram-budget/specs/01-vram-budget.md`, master #207 con le slice
#208–#215. Differenze rispetto ai fix candidati sopra:
- il risparmio vision è ~1.25 GiB, non 2.4;
- i system consecutivi vengono uniti con `merge` (default) oppure rifiutati
  con `strict`; per i `developer` c'è una policy separata;
- confronto con gli altri motori:
  `research-multi-system-messages.md`.
