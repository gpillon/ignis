# A request after up to 60 s of GPU idle runs as fast as a warm one: the reported clock lags, the work does not

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: runtime / GPU clocks under WDDM (driver 596.36), short-request latency, warm-up mitigations
- Related: the 2026-09-22 point-latency decomposition (a first request after a pause
  seen at 326 ms and 307 MHz, against 22–26 ms at 2,917 MHz),
  `.scratch/sota-research-2026-09-24/SINTESI.md` (clock floor / warm-up item), raw
  material in `.scratch/clocks-2026-09-24/`
- Superseded by: none

## Question

The RTX 5090 drops to a few hundred MHz within seconds of idling. A coding agent's
requests arrive in bursts with gaps between them. Does the first request after a gap
pay for the clock ramp? If so, would a client-side mitigation buy it back without
admin rights (`nvidia-smi -lgc` needs them)? The two candidates are a 1-token priming
request just before, and a 1-token keep-alive ping every second.

## Evidence

**Setup.**
- Server `4039aa9` with the `make config` flags. The shell was not elevated.
- `clocks.py` sends the same ~300-token coding prompt, greedy, no thinking, 64 tokens.
  The prompt is identical every time, so prompt reuse is in the same state on every
  request.
- Cells:
  - five warm requests back to back;
  - one request after 2, 5, 10, 20, 60, 20 and 5 s of idle;
  - three "primed" requests (20 s idle, a 1-token request, then the real one);
  - two "keepalive" requests, after 20 s of a 1-token ping every 1 s.
- `nvidia-smi` sampled SM clock and power every ~62 ms.

| cell | TTFT (ms) | total (ms) | decode tok/s | reported SM clock at start |
|---|---|---|---|---|
| warm (×5) | 37–63 | 224–249 | 336–341 | 2,887–2,895 MHz |
| after 2–60 s idle (×7) | 38–64 | 224–256 | 314–342 | 330–2,197 MHz |
| primed (×3) | 38–84 | 227–275 | 330–333 | 307–502 MHz |
| keepalive (×2) | 38–54 | 225–242 | 335–338 | 352–712 MHz |

- After idle, the *reported* clock takes 430–760 ms to reach 2,500 MHz, longer than the
  whole 250 ms request. Warm, it takes 2–44 ms.
- Idle power is 67.6 W. With the keep-alive ping it is 71.6 W, and the reported clock
  median rises only from 412 to 664 MHz.

## Finding

**Observed.**
- A request that starts after 2–60 s of idle completes in the same time as a warm one:
  total 224–256 ms against 224–249 ms, and the same decode rate.
- The TTFT spread (37–84 ms) is the same in every cell.
- Priming and keep-alive change nothing measurable. Keep-alive costs +4 W for as long
  as the GPU would otherwise be idle.

**Inference.**
- nvidia-smi's clock reading lags the clock the kernels actually run at: a request
  finished at full speed while its samples still showed a few hundred MHz.
- The 2026-09-22 "326 ms at 307 MHz" first request was not the clock. That
  decomposition also found prompt reuse bimodal (a miss re-prefills the image), which
  explains a 5–10× outlier by itself.

## Implications

- No clock floor, warm-up at admission, or keep-alive is needed for request latency on
  this driver. SINTESI's clock item is closed as not a problem.
- nvidia-smi's SM clock is not evidence of what a short request ran at. Latency
  investigations should time the work itself.

## Limits and unknowns

- One driver (596.36), Windows WDDM, one prompt shape (300 tokens in, 64 out). A cold
  *prefill* of tens of thousands of tokens was not tested after idle; the ramp, if
  real, would be amortized there anyway.
- Idle gaps up to 60 s. Longer idles, deeper power states and display-off states were
  not tested.
- n = 2–7 per cell.
