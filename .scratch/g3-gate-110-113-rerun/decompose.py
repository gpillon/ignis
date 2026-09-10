import json, statistics

def analyze(path):
    d = json.load(open(path, encoding='utf-8'))
    itl = d['itl']
    windows = [(p['started_ms'], p['first_token_ms']) for p in itl['prefillers']]
    blocked = []
    free = []
    for lane in itl['lanes']:
        times = lane['token_times_ms']
        for a, b in zip(times, times[1:]):
            mid = (a + b) / 2
            iv = b - a
            if any(w0 <= mid <= w1 for w0, w1 in windows):
                blocked.append(iv)
            else:
                free.append(iv)
    return {
        'label': d['label'],
        'n_blocked': len(blocked),
        'n_free': len(free),
        'blocked_mean': statistics.mean(blocked) if blocked else None,
        'blocked_median': statistics.median(blocked) if blocked else None,
        'free_mean': statistics.mean(free) if free else None,
        'free_median': statistics.median(free) if free else None,
    }

for f in ['reference-1.json','reference-2.json','ignis-1.json','ignis-2.json']:
    r = analyze(f'.scratch/g3-gate-110-113-rerun/{f}')
    print(f, r)
