# N-gram embedding come memoria condizionale — i fatti

- Data: 2026-09-18
- Branch: `ngram-study`
- Stato: esplorazione, nessuna misura propria ancora

Questo documento raccoglie solo ciò che è verificato da fonte primaria, per
separare i fatti dalle ipotesi di design (file `01-strade.md`).

## Di cosa si tratta (e cosa NON è)

Non è speculative decoding. Non è il drafter DFlash2 già presente in ignis.

È un **asse di sparsità aggiuntivo**: una tabella di embedding indicizzata da
un hash dell'n-gram locale (gli ultimi 2-3 token). Il modello ci deposita
conoscenza statica — entità, collocazioni, pattern frequenti — che altrimenti
occuperebbe parametri densi o esperti MoE. Il costo per token è **una gather di
poche righe**, non FLOPs. Poiché l'indirizzo è deterministico e noto in
anticipo, la tabella può stare in **RAM host** e non in VRAM.

Due incarnazioni indipendenti, entrambe del 2026:

| | Qwen3.8-Flash-Next | DeepSeek Engram |
|---|---|---|
| Uscita | 2026-08-26 | 2026-01 |
| Nome del modulo | N-gram Embedding (PLE) | Engram |
| Taglia tabella | 51,2 B param | 5,7 B (27B) / 18,5 B (40B) |
| Ordini n-gram | 2 e 3 | 2 e 3 (max N=3) |
| Teste per ordine | 8 | 8 |
| Iniezione | layer 2 | layer 2 e 15 |
| Offload host | sì, nativo | sì, "negligible overhead", picco 2,8 % |

## Qwen3.8-Flash-Next — config primaria

Da `huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`:

```json
"ngram_size": 3,
"ngram_vocab_size_base": 20000000,
"make_ngram_vocab_size_divisible_by": 128,
"split_ngram_parts": 128,
"heads_per_ngram": 8
```

Modello: 125 B totali, 6 B attivati, **+51 B tabella n-gram**, +4 B modulo MTP.
Backbone ibrido: 3× Gated DeltaNet + 1× Qwen Sparse Attention, ripetuti su 48
layer, MoE dopo ogni blocco di attenzione.

### Aritmetica della tabella (verificata contro il totale dichiarato)

- 2 ordini (bigram, trigram) × 8 teste = **16 tabelle**
- 16 × 20.000.000 righe = 320 M righe
- 51,2e9 / 320e6 = **160 di dimensione per riga**
- Controprova: 16 × 20e6 × 160 = 51,2e9 ✅ coincide con i 51 B dichiarati
- SGLang dichiara **95,4 GiB in BF16** ✅ coincide con 51,2e9 × 2 B

Costo per token (**il numero che conta**):

| formato | byte/riga | byte/token (16 righe) |
|---|---|---|
| BF16 | 320 | **5.120 B** |
| FP8 | 160 | 2.560 B |
| NVFP4 (+scale) | ~88 | ~1.408 B |

Occupazione totale: BF16 95,4 GiB · FP8 47,7 GiB · NVFP4 ~24-25 GiB.

Inferenza, non fatto: 16 × 160 = 2.560 è la larghezza del vettore concatenato
prima della fusione. Non ho la hidden size di Flash-Next da fonte primaria, per
cui non so se 2.560 coincida con essa o vada proiettato.

## Come lo si serve davvero — SGLang, day-0

Fonte: `lmsys.org/blog/2026-08-26-qwen-flash-next/`. È il riferimento
implementativo più vicino al problema di ignis, perché è codice di serving.

1. **La tabella sta in pinned host memory**, shard per rank (vocab-parallel).
2. La gather è un **kernel Triton via UVA**: legge direttamente la host memory
   mappata e scrive le righe in un piccolo buffer BF16 su GPU. Nessuna copia
   host-side, nessun `cudaMemcpy` esplicito.
3. **Uno stream CUDA dedicato** sovrappone la gather con il **primo decoder
   block**; l'iniezione avviene a layer 2.
4. Indirizzamento: 8 teste 2-gram su `(x_{t-1}, x_t)`, 8 teste 3-gram su
   `(x_{t-2}, x_{t-1}, x_t)` → **16 row id per token**.
5. Stato per richiesta: **i due token id precedenti**, e nient'altro.
6. Il PLE resta attivo in **prefill, decode e verifica del target**; viene
   disattivato **solo nel draft MTP a un layer**.
7. Effetto misurato su H200: throughput **-0,07 % (media geometrica)**,
   -23,46 GiB di pesi per GPU, **+78,54 % di capacità KV**.
8. L'interazione con i CUDA graph **non è discussa** nella fonte.

Il punto 8 è il buco che riguarda ignis direttamente (vedi `01-strade.md`).

## Il retrofit su modello congelato

- **Engram Adapter** (arXiv 2608.29327): riusa la memoria condizionale come
  adapter post-hoc su LLM **congelato**, provato su Qwen3-4B e Qwen3-8B.
  Matching multi-canale su pattern n-gram locali, occupancy tracking, gate
  scalare appreso. Risultato: accuratezza in-dominio migliore mantenendo il
  99,4-100,1 % della performance fuori dominio (le baseline "always-on"
  degradano nettamente). Compute e volume dati di training non dichiarati
  nell'abstract.
- **Memory Grafting** (arXiv 2605.20948): hidden state congelati di un modello
  donatore usati come memoria n-gram, recuperati per longest-match suffix.
- **Lngram** (arXiv 2605.24869): memoria condizionale n-gram in spazio latente.

Sono tutti lavori di **training**, non di serving.

## Vincoli duri già accertati

### Il modello che ignis serve non ha la tabella

Ignis serve `qwen3_8_27b_nvfp4full-v2.ninfer` = **Qwen3.8-27B** NVFP4 + graft
DFlash2. Il manifest `…nvfp4full.ninfer.conversion.json` fissa la base a
`Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Verificato sulla config primaria di quel modello
(`huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json`): **nessun campo
`ngram`, `n_gram`, `heads_per_ngram`, `ngram_vocab_size_base`**. La config
riporta `hidden_size 5120`, `num_hidden_layers 64`, `vocab_size 248320`,
`intermediate_size 17408` — gli stessi valori già cablati in
`crates/core/src/speculation.rs`, quindi è il modello giusto.

**Qwen3.8-27B non ha modulo n-gram.** Non è una questione di supporto engine:
i pesi non esistono.

### ninfer non ha codice n-gram

Zero occorrenze su tutto `F:/ai/q38/ninfer` (`*.cpp *.cu *.h *.hpp *.cuh
*.py`) per `ngram|n_gram|engram`, e zero anche per
`flash_next|flashnext|flash-next|per_layer_embed|PLE` su `src/` e `include/` —
la seconda passata serve perché SGLang chiama il modulo "PLE", quindi una
ricerca per solo "ngram" avrebbe potuto mancare un'implementazione battezzata
col nome del modello.

Ignis non ha un riferimento live/live da cui misurare, contrariamente a ogni
lavoro precedente del repo.

### L'hardware

| risorsa | disponibile | serve per Flash-Next |
|---|---|---|
| VRAM | 32,6 GiB (RTX 5090) | 125 B pesi — non entra nemmeno a NVFP4 (~62 GiB) |
| RAM host | **63,8 GiB** DDR4-3200 (35 GiB liberi) | tabella 95,4 GiB BF16 — **non entra** |

La tabella entrerebbe solo quantizzata: FP8 47,7 GiB (stretto, e resterebbe
poca RAM per il tier KV host che il fork usa già), NVFP4 ~24-25 GiB.
Ma i pesi del backbone restano il vincolo bloccante.

### Il budget di latenza, per contrasto

Un decode round di ignis a una lane è **15,81 ms** di device time con 0,80 ms
di idle (4,8 %), 1.166 nodi di graph
(`docs/findings/2026-09-18-decode-round-anatomy.md`).

Contro questo: 5.120 B per token di gather n-gram. Su PCIe Gen5 x16 la banda è
irrilevante di ordini di grandezza. **Il costo non è banda, è latenza di 16
letture random su una tabella di decine di GiB** — e la domanda è se quella
latenza si nasconde dentro un graph catturato.

## Fonti

- [Qwen3.8-Flash-Next — model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next — config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json)
- [Qwen3.8-Flash-Next — repo](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [SGLang day-0 support](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)
- [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
- [Engram — Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372)
- [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)
- [Engram Adapter](https://arxiv.org/html/2608.29327)
- [Memory Grafting](https://arxiv.org/abs/2605.20948v1)
- [Lngram](https://arxiv.org/pdf/2605.24869)
