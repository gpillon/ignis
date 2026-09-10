import json, statistics

for f in ['reference-1.json','reference-2.json','ignis-1.json','ignis-2.json']:
    d = json.load(open(f'.scratch/g3-gate-110-113-rerun/{f}', encoding='utf-8'))
    ttfts = [p['ttft_ms'] for p in d['itl']['prefillers'] if not p['void']]
    tok = d['itl']['prefillers'][0]['computed_prefill_tokens']
    mean_ttft = statistics.mean(ttfts)
    median_ttft = statistics.median(ttfts)
    tokps = tok / (mean_ttft / 1000)
    chunk_ms = mean_ttft / (tok / 1024)
    print(f"{d['label']:10s} {f:20s} n={len(ttfts):2d} mean_ttft={mean_ttft:8.1f}ms median={median_ttft:8.1f}ms  prefill_tok/s={tokps:8.1f}  per_1024_chunk={chunk_ms:6.2f}ms")
