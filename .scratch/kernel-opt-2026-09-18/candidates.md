# Candidati di ottimizzazione nel program layer del kernel leaf — 2026-09-18

Solo lettura del codice, **nessuna misura su GPU**. Tutto quanto segue sta in
`kernel/src/` (program layer, nostro, ADR 0009) — nessun file vendored toccato
(ADR 0010).

Geometria di riferimento: 64 layer (16 GQA / 48 GDN), hidden 5120, 24 q-head su
4 kv-head da 256, default `KV_FORMAT=hq-e8-2b`, `MAX_CONTEXT=262144`,
`PREFILL_CHUNK=1024`.

## A. Il memset del workspace di attention è lavoro morto

`kernel/src/gqa_layer.cu:201` (prefill eager) e `:426` (decode graph) azzerano
**tutto** il workspace transiente prima di ogni chiamata A1, a ogni layer GQA.

Motivazione storica (commento P2-02/#84): "una fresh, zeroed workspace impedisce
che partial di split inattivi vengano ridotti". Verificato nel vendored:

- `gqa_attention_decode.cuh:217` — il reducer calcola `active_split_count` **sul
  device** dalla finestra reale (`last_pos + 1`) e legge solo `split <
  active_split_count`.
- `gqa_attention_decode_bf16.cuh:151` — per gli split inattivi la route **hq
  chiama `write_neutral()`**; solo la route bf16 "relies on the engine's
  zero-initialized partial workspace".
- `gqa_attention_prefill_hq_routes.cuh:73-99` — i piani scratch hq
  (`scratch_k`/`scratch_v`) vengono **scritti per intero** su `band_rows` prima
  che il kernel FA2 li legga. Azzerarli è puro spreco.

Conseguenze sotto il formato di default (hq-e8-2b):

- **Prefill.** Il workspace hq prompt è ≈ `span × 4096` byte
  (`span = min(key visibili, 262144)`), cioè il pezzo dominante. Un prompt da
  32K in chunk da 1024: Σ 16 layer × (i+1)·1024 · 4096 B ≈ **33.8 GB** scritti
  per il solo azzeramento. A ~1.6 TB/s ≈ 21 ms su ~3 s di prefill → **~0.7%**.
  Cresce linearmente col contesto.
- **Decode.** Envelope fisso a `max_context_tokens` → 85 split sempre →
  ≈ 1.06 MB × B per layer → **~17 MB × B per round**, e sotto hq la stessa
  regione viene riscritta subito dopo da `write_neutral()`.

Fix: togliere il memset sulla route U8 (provabilmente morto). Su BF16 (formato
oracolo, non critico per le prestazioni) lasciarlo, perché il commento vendored
dichiara esplicitamente di dipendere dallo zero-init.

## B. La copia del residual per layer serve solo al ping-pong

`gqa_layer.cu:217,441` e `gdn_layer.cu:231,409,552`: `cudaMemcpyAsync(out_residual,
in_residual, hidden*columns*2)` a ogni layer, perché `decode_graph.cu` fa
`std::swap(left, right)`. Ma `in_residual` è letto solo dal primo rmsnorm, e
`linear_add` accumula in place: con `in == out` la copia sparisce e il risultato
è identico.

Costo prefill: 64 layer × 10.5 MiB (1024 token) letti+scritti ≈ **1.34 GB per
chunk** ≈ 0.9%. Decode: 64 nodi di grafo in meno per round, e un buffer residual
invece di due.

## C. Lo split QKV del GDN è tre copie 2D stridate

`gdn_layer.cu:175,371`: l'uscita fusa `[conv_channels, T]` di
`causal_conv1d_silu` viene spezzata in query/key/value con tre
`cudaMemcpy2DAsync` stridate, per layer. 48 layer × 3 = **144 copie per
round/chunk**. Da verificare: se il wrapper di `gated_delta_net_snapshot`
richiede tensori contigui, e come lo risolve il reference (op fusa non
vendorizzata?).

## D. L'ipotesi che lega tutto: 224 nodi non-compute per decode round

GQA 16 × (1 memset + 1 memcpy) = 32; GDN 48 × (3 memcpy2D + 1 memcpy) = 192.
**224 nodi memset/memcpy per round**, interposti fra kernel vendored che sono
quasi tutti PDL-chained (`pdl::launch_dependent` / `pdl::sync` / `pdl::publish`
in linear, linear_add, linear_swiglu, attn_input_proj, gdn_input_proj,
gated_delta_net recurrent, rmsnorm, rope, qk_norm_rope, gqa decode). Un nodo
memcpy/memset non ha trigger programmatico: ogni occorrenza degrada
plausibilmente l'edge PDL successivo a dipendenza piena di stream.

Collegamento alla misura nota: `docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md`
riporta ignis al 25-27% di utilizzo memoria contro 37-44% di ninfer.

Verifica proposta (carta esclusiva, `make gpu-status` prima): una traccia nsys di
un singolo decode round a B=1 e B=8, contando i tipi di nodo e i gap fra kernel.

## Follow-up separato (decisione di design, non ticket)

Il decode graph passa `max_visible_keys = 262144` sempre → 85 split anche a 1K di
contesto. Il vendored prevede esplicitamente che "graph calls pass their
target-private replay interval": grafi catturati per fascia di contesto
(≤4K/≤16K/≤64K/max) porterebbero gli split da 85 a 16-64 nei casi tipici.
