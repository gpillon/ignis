# The draft head's ranking

`ranking.totals.i64` is the frequency ranking that picks the 27B's draft head's 131,072 rows
(`checkpoint.draft_ids`, ported from ninfer's `qwen3_6/common/draft_head.py`): the default of the converter's
`--ranking`.

- **Source:** `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` in
  <https://github.com/Neroued/ninfer>, commit `eea471ce5600402914091ee4f0aa8d859a459966` ("chore(freq-corpus):
  track final frequency statistics", 2026-07-06), git blob `fc230ca6eb288d35592c33a185dd70c4ce1b9f80`:
  11,919,360 bytes, sha256 `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5`. Its manifest
  (`ranking.train.manifest.json`) gives the layout, `int64 LE; row 0 = total[vocab]; rows 1..5 = strata[vocab]
  in strata_order`: 6 rows of the padded vocabulary's 248,320 ids, the model's teacher-forced argmax counted over
  the assistant tokens of ninfer's frequency corpus.
- **This file:** row 0 of the 6, the totals, unchanged: 248,320 little-endian int64, 1,986,560 bytes, sha256
  `1e8fb35cd76655acac1cc422e2ef36e7dbedae567e04527ae433cf30b7dbd59f`. It is the only row the converter reads;
  the five strata rows, whose sums the totals are, are left out.
- **Pins:** both sha256s are in `checkpoint.py`'s `PINNED`, so `--ranking` takes this file (the default) or
  ninfer's whole one, and both give the same rows; any other file is refused. `test_ranking.py` checks this
  file's size and sha256; `test_real_sources.py`, where ninfer's checkout is on the machine, that it is the first
  1,986,560 bytes of ninfer's file and that both give the base checkpoint's draft head the same ids.
- **Licence:** Apache-2.0, ninfer's `LICENSE`, whose text is ignis's own `LICENSE` (and `kernel/vendor/LICENSE`,
  ninfer's copy). ninfer's `LICENSE` came in at `433af9c0` (2026-07-19), after `eea471ce`; the source file is
  the same blob at the upstream master `9e163eee` (2026-09-18), which carries it. ninfer has no `NOTICE` file,
  so there are no attribution notices to carry (Apache-2.0 §4(d)). This file is a changed form of the source,
  its row 0 only, as stated above (§4(b)). The root `NOTICE` names it.
