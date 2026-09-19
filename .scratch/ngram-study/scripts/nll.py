"""Teacher-forced NLL, with the index on and off.

`02-storage.md` §7 Fase 2 picks NLL over generation because it is a number, not
a judgement, and the repo already has a teacher-forced precedent (ADR 0014).

Two details the plan is explicit about and that the code has to honour:

* The injection happens on the symbol *name*; the effect lands on the tokens
  that follow.  So the primary metric is the NLL over the N tokens after each
  match, N in {8, 32}, and the whole-file number is secondary — averaging over
  a file dilutes a strong local effect across everything that never matched.
* The match rate is reported next to every delta.  A -1% delta with a 1% match
  rate and a -1% delta with a 40% match rate mean different things, and without
  the rate neither is interpretable.

Logits are never materialised for the whole sequence: 248320 vocab entries at
4k positions is 2 GB in bf16 and 4 GB once upcast, so the head runs in position
chunks and only the gathered target log-probabilities are kept.
"""

import re

import torch

from inject import Injector
from modeladapter import lm_head


def build_matches(tokenizer, text, keys, key_rows, offsets=None, ids=None,
                  max_tokens=None):
    """Find every whole-identifier occurrence of an indexed key, and map it to
    the token that *completes* the name.

    Matching on text rather than on token ids is deliberate: `decode_round` and
    ` decode_round` tokenize differently, so a sliding window over ids would
    miss half the occurrences depending on the preceding character.  The name
    is found in the source, and `offset_mapping` turns the character span into
    the last token overlapping it.
    """
    if ids is None:
        enc = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True,
                        truncation=max_tokens is not None,
                        max_length=max_tokens)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]

    limit = offsets[-1][1] if offsets else 0
    starts = [a for a, _ in offsets]

    positions, rows, names = [], [], []
    seen = set()
    for key in keys:
        pat = re.compile(r"(?<![A-Za-z0-9_$])%s(?![A-Za-z0-9_$])"
                         % re.escape(key))
        for m in pat.finditer(text):
            if m.end() > limit:
                break
            last = _last_token_overlapping(offsets, starts, m.start(), m.end())
            if last is None or last in seen:
                continue
            seen.add(last)
            positions.append(last)
            rows.append(key_rows[key])
            names.append(key)
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    return (ids, offsets,
            [positions[i] for i in order],
            [rows[i] for i in order],
            [names[i] for i in order])


def _last_token_overlapping(offsets, starts, a, b):
    import bisect
    hi = bisect.bisect_left(starts, b)
    for i in range(min(hi, len(offsets)) - 1, -1, -1):
        s, e = offsets[i]
        if s == e:
            continue
        if s < b and e > a:
            return i
        if e <= a:
            break
    return None


@torch.no_grad()
def token_nll(model, input_ids, chunk=512):
    """Negative log-likelihood of each real next token, length T-1."""
    base = model.model
    hidden = base(input_ids=input_ids, use_cache=False).last_hidden_state[0]
    head = lm_head(model)
    targets = input_ids[0, 1:]
    out = torch.empty(targets.shape[0], dtype=torch.float32)
    for a in range(0, targets.shape[0], chunk):
        b = min(a + chunk, targets.shape[0])
        logits = head(hidden[a:b]).float()
        lp = torch.log_softmax(logits, dim=-1)
        out[a:b] = -lp.gather(1, targets[a:b].unsqueeze(1)).squeeze(1).cpu()
    return out


@torch.no_grad()
def run_file(model, tokenizer, text, keys, key_rows, vectors, layer, alpha,
             cos_threshold=None, max_tokens=None, chunk=512, force_hook=False,
             center=None):
    """One file, one (alpha, threshold): returns per-token NLL and the match set.

    `force_hook` runs the injection path even at alpha = 0.  That is the only
    way the §7 Fase 3 control means anything: without it, alpha = 0 skips the
    hook entirely and trivially equals the baseline, testing nothing.  With it,
    the clone, the cosine and the `index_add` all run, and a zero amplitude
    still has to come back bit for bit identical.
    """
    ids, offsets, positions, rows, names = build_matches(
        tokenizer, text, keys, key_rows, max_tokens=max_tokens)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    if positions:
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        vec = vectors.index_select(0, torch.tensor(rows)).to(device)
    else:
        pos = torch.zeros(0, dtype=torch.long, device=device)
        vec = torch.zeros(0, vectors.shape[1], device=device)

    with Injector(model, layer, pos, vec, alpha, cos_threshold,
                  force=force_hook, center=center) as inj:
        nll = token_nll(model, input_ids, chunk=chunk)
        injected = inj.injected if alpha != 0.0 else 0
        cosines = inj.cosines
    torch.cuda.empty_cache()        # see the note in hiddens.hiddens_for_spans
    return {
        "nll": nll, "tokens": len(ids), "positions": positions,
        "names": names, "injected": injected,
        "cosines": None if cosines is None else cosines.tolist(),
    }


def window_mean(nll, positions, width):
    """Mean NLL over the `width` tokens after each match.

    `nll[i]` is the cost of predicting token i+1, so the token right after a
    match at position p is scored by `nll[p]`.  Windows are unioned, not summed
    per match, so two matches close together are not counted twice.
    """
    picked = set()
    for p in positions:
        for k in range(p, min(p + width, len(nll))):
            picked.add(k)
    if not picked:
        return None, 0
    idx = torch.tensor(sorted(picked))
    return float(nll.index_select(0, idx).mean()), len(picked)


def bootstrap_ci(per_file_deltas, iters=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over files — the unit of resampling is the file,
    because tokens inside one file are anything but independent."""
    import random
    if not per_file_deltas:
        return None
    rng = random.Random(seed)
    n = len(per_file_deltas)
    means = []
    for _ in range(iters):
        means.append(sum(per_file_deltas[rng.randrange(n)]
                         for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return {"mean": sum(per_file_deltas) / n, "lo": lo, "hi": hi, "n_files": n}
