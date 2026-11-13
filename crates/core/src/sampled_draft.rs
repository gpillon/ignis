//! The **sampled draft**'s rule (spec #574, GitHub #575): the host reference
//! of the arithmetic a sampling lane of the 27B's DFlash2 verify round
//! follows, which the device kernels are held to. The engine never calls it:
//! a round under the sampled draft runs the leaf's kernels
//! (`kernel/src/sampled_draft.cuh`, GitHub #576, ADR 0090), which a GPU test
//! holds to the same oracle through the `device` module.
//!
//! Under sampling the drafter draws each column's draft d from its own
//! distribution q over the column's candidates, instead of proposing a fixed
//! one, and the accept compares the two distributions: it takes d when
//! `u·q(d) < p(d)`, p being the target's sampling distribution at the column
//! (the sampler's own, renormalized over its support) and u the accept's
//! uniform, so d is taken with probability min(1, p(d)/q(d)). On a refusal
//! the lane's token is drawn, with the refusal's own uniform, from
//! max(p − q, 0) renormalized over the target's support, and the lane's
//! accepted run ends there. A draft the target gives no probability to is
//! always refused. This is the standard rule of speculative sampling: the
//! token a column emits is distributed exactly as p, whatever q is, so q
//! moves how often a draft is taken and never what text can come out.
//!
//! q is the softmax of the selector's transition scores for the column, given
//! the draft chosen at the column before it, at the drafter temperature, cut
//! by the drafter top-p and renormalized ([`drafter_distribution`]); the
//! draft is drawn from it with the draft's own uniform ([`draw_draft`]). At a
//! drafter temperature of 0 the draft is the highest-scored candidate, the
//! selector walk's fixed draft ([`crate::dflash2_readout::Dflash2Proposal::walk`]).
//!
//! The arithmetic is f32, as on the device. Every random number enters as an
//! explicit uniform in [0, 1), the value the sampler's counter-based draw
//! gives on the device, so a test can sweep them:
//! `crates/core/tests/sampled_draft_oracle.rs` holds this reference to a
//! double-precision oracle over a grid of the draft's, the accept's and the
//! refusal's uniforms.
//!
//! The replay of recorded turns forces a recording through rounds under the
//! rule; [`forced_verdict`] is that round's verdict (GitHub #577), held to the
//! same oracle by `crates/core/tests/sampled_draft_forced_oracle.rs`.

/// A column's distribution over token ids, in the order a draw from it walks
/// them: the target's support in the sampler's order, or the drafter's
/// candidates in the proposal head's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution<'a> {
    ids: &'a [i32],
    probs: &'a [f32],
}

impl<'a> Distribution<'a> {
    /// `probs[i]` is the probability of `ids[i]`.
    ///
    /// # Panics
    ///
    /// When the two differ in length.
    pub fn new(ids: &'a [i32], probs: &'a [f32]) -> Self {
        assert_eq!(ids.len(), probs.len(), "a distribution pairs each id with its probability");
        Self { ids, probs }
    }

    /// The probability of `token`: the sum over the entries that hold it, 0
    /// when none does.
    pub fn prob(&self, token: i32) -> f32 {
        self.ids.iter().zip(self.probs).filter(|&(&id, _)| id == token).map(|(_, &p)| p).sum()
    }
}

/// The two uniforms a draft column's accept reads, each a draw of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnUniforms {
    /// The accept's: the draft is taken when `accept·q(d) < p(d)`.
    pub accept: f32,
    /// The refusal's: the inverse-CDF draw from max(p − q, 0).
    pub refusal: f32,
}

/// What the accept says of one draft column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The draft is taken: the lane emits it and goes on to its next column.
    Accepted,
    /// The draft is refused: the lane emits this token, drawn from what the
    /// target's distribution has left over the drafter's, and its accepted
    /// run ends.
    Refused(i32),
}

/// The accept's verdict on `draft` at one column: taken when
/// `uniforms.accept · q(draft) < p(draft)`, with p the `target`'s sampling
/// distribution and q the `drafter`'s; otherwise refused, with the token
/// drawn by [`uniforms.refusal`](ColumnUniforms::refusal) from
/// max(p − q, 0), renormalized over the target's support and walked in its
/// order.
///
/// # Panics
///
/// When a refusal finds no positive mass in the target, whose support is
/// never empty.
pub fn accept_verdict(target: Distribution<'_>, drafter: Distribution<'_>, draft: i32, uniforms: ColumnUniforms) -> Verdict {
    if uniforms.accept * drafter.prob(draft) < target.prob(draft) {
        return Verdict::Accepted;
    }
    let residual = |j: usize| (target.probs[j] - drafter.prob(target.ids[j])).max(0.0);
    let j = inverse_cdf(target.ids.len(), residual, uniforms.refusal)
        // A refusal means q(d) > p(d), so p has mass left over q; only
        // rounding leaves none, and then the draw is from p itself.
        .or_else(|| inverse_cdf(target.ids.len(), |j| target.probs[j], uniforms.refusal))
        .expect("the target's support holds positive mass");
    Verdict::Refused(target.ids[j])
}

/// The **teacher-forced** verdict under the sampled draft (GitHub #577): at a
/// column whose recorded token is `recorded` and whose draft is that token
/// (the forced round drafts the recording wherever it is one of the
/// drafter's candidates), whether the round takes it.
///
/// A live round that emits a token x has drafted x, and taken it, with
/// probability min(p(x), q(x)) / p(x): the rest of the time it refused
/// another draft and drew x from what p has left over q. The forced verdict
/// takes the recorded token with that probability, so the run a replay
/// commits is the recording's and the rounds it walks are distributed as live
/// rounds are. It gets there through the live round's own draws first:
/// `live` is the [`accept_verdict`] on `live_draft`, the draft the lane's
/// draw would have proposed, under the column's own uniforms. When that live
/// column emits the recorded token, the forced verdict is the live one --
/// taken when it took its draft, refused when it drew the token on a refusal
/// -- so a recording of the lane's own rounds is forced back into exactly
/// those rounds. When it emits another token, which is what a recording made
/// elsewhere mostly meets, the `coin`, a uniform of its own, takes the
/// recorded token when `coin·p(x) < q(x)`. Over the live draws and the coin
/// the token is taken with probability min(p, q)(x) + (1 − p(x))·min(1,
/// q(x)/p(x)) = min(p(x), q(x)) / p(x). A recorded token the target gives no
/// probability to (a recording sampled by other numerics) is taken with the
/// drafter's own probability of it, `coin < q(x)`.
///
/// `target` and `drafter` are the column's p and q, as [`accept_verdict`]
/// takes them; q is the drafter's distribution after the recorded token
/// before (the forced round drafts along the recording).
pub fn forced_verdict(
    target: Distribution<'_>,
    drafter: Distribution<'_>,
    recorded: i32,
    live_draft: i32,
    live: Verdict,
    coin: f32,
) -> bool {
    match live {
        Verdict::Accepted if live_draft == recorded => return true,
        Verdict::Refused(token) if token == recorded => return false,
        _ => {}
    }
    let p = target.prob(recorded);
    let q = drafter.prob(recorded);
    if p > 0.0 { coin * p < q } else { coin < q }
}

/// The drafter's distribution q over a column's candidates: the softmax of
/// `scores` at `temperature`, cut to the smallest prefix of the candidates in
/// descending score (ties toward the smaller index) that holds `top_p` of the
/// mass, the sampler's inclusive top-p, and renormalized over it. `scores` is
/// the column's row of the selector's transition scores after the previous
/// column's draft (`Dflash2Proposal::scores`, `[pred][succ]`); `top_p` of 1
/// or more cuts nothing, and the prefix holds at least the first candidate.
///
/// A `temperature` of 0 or below puts all the mass on the first highest
/// score: the selector walk's draft, which the distribution tends to as the
/// temperature does.
///
/// # Panics
///
/// When `scores` is empty.
pub fn drafter_distribution(scores: &[f32], temperature: f32, top_p: f32) -> Vec<f32> {
    assert!(!scores.is_empty(), "a draft column has candidates");
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut q = vec![0f32; scores.len()];
    if !(temperature > 0.0) {
        q[order[0]] = 1.0;
        return q;
    }
    let max = scores[order[0]];
    for (qi, &s) in q.iter_mut().zip(scores) {
        *qi = ((s - max) / temperature).exp();
    }
    let mut kept = order.len();
    if top_p < 1.0 {
        let goal = top_p * q.iter().sum::<f32>();
        let mut cum = 0.0f32;
        for (k, &i) in order.iter().enumerate() {
            cum += q[i];
            if cum >= goal {
                kept = k + 1;
                break;
            }
        }
    }
    let mass: f32 = order[..kept].iter().map(|&i| q[i]).sum();
    for &i in &order[kept..] {
        q[i] = 0.0;
    }
    for &i in &order[..kept] {
        q[i] /= mass;
    }
    q
}

/// The draft drawn from `q` (a [`drafter_distribution`]) with its own
/// `uniform`: the inverse CDF over the candidates in their order. Returns the
/// candidate's index; a candidate with no mass is never drawn.
///
/// # Panics
///
/// When `q` holds no positive mass.
pub fn draw_draft(q: &[f32], uniform: f32) -> usize {
    inverse_cdf(q.len(), |i| q[i], uniform).expect("a drafter distribution holds positive mass")
}

/// The inverse CDF over `len` weights in their order: the first index at
/// which the running sum of the positive weights passes `uniform` times
/// their total, or the last positive one when rounding leaves the goal at the
/// total. `None` when no weight is positive.
fn inverse_cdf(len: usize, weight: impl Fn(usize) -> f32, uniform: f32) -> Option<usize> {
    let positive = |j: usize| weight(j).max(0.0);
    let total: f32 = (0..len).map(positive).sum();
    if !(total > 0.0) {
        return None;
    }
    let goal = uniform * total;
    let mut cum = 0.0f32;
    let mut picked = None;
    for j in (0..len).filter(|&j| positive(j) > 0.0) {
        cum += positive(j);
        picked = Some(j);
        if goal < cum {
            break;
        }
    }
    picked
}

/// The leaf's device kernels of the same rule over explicit uniforms (GitHub
/// #576, `kernel/include/ignis_sampled_draft.h`): **test-only**, the seam a
/// GPU test sweeps over the oracle's grid
/// (`crates/core/tests/sampled_draft_gpu.rs`), as the CPU suite sweeps the
/// functions above. No serving path calls it: the verify round draws its
/// uniforms from the sampler's counter-based RNG on the device.
#[cfg(feature = "cuda")]
pub mod device {
    use super::Verdict;

    /// What the leaf writes at a grid point where the draft was taken.
    const ACCEPTED: i32 = -1;

    mod ffi {
        use std::os::raw::c_char;

        unsafe extern "C" {
            pub fn ignis_sampled_draft_verdicts(
                target_ids: *const i32,
                target_probs: *const f32,
                target_count: u32,
                drafter_ids: *const i32,
                drafter_probs: *const f32,
                drafter_count: u32,
                draft: i32,
                grid: *const f32,
                cells: u32,
                out_tokens: *mut i32,
            ) -> i32;
            pub fn ignis_sampled_draft_forced_verdicts(
                target_ids: *const i32,
                target_probs: *const f32,
                target_count: u32,
                drafter_ids: *const i32,
                drafter_probs: *const f32,
                drafter_count: u32,
                recorded: i32,
                live_draft: i32,
                live_taken: i32,
                live_token: i32,
                grid: *const f32,
                cells: u32,
                out_taken: *mut u8,
            ) -> i32;
            pub fn ignis_sampled_draft_draws(q: *const f32, count: u32, grid: *const f32, cells: u32, out_candidates: *mut i32) -> i32;
            pub fn ignis_sampled_drafter_distribution(scores: *const f32, count: u32, temperature: f32, top_p: f32, out_q: *mut f32) -> i32;
            pub fn ignis_sampled_draft_last_error() -> *const c_char;
        }
    }

    fn last_error() -> String {
        let message = unsafe { std::ffi::CStr::from_ptr(ffi::ignis_sampled_draft_last_error()) };
        message.to_string_lossy().into_owned()
    }

    fn check(rc: i32) -> Result<(), String> {
        if rc == 0 { Ok(()) } else { Err(last_error()) }
    }

    /// The device accept's verdict on `draft` at every point of `grid` x
    /// `grid`, the accept's uniform `grid[i]` and the refusal's `grid[j]` at
    /// `[i * grid.len() + j]`; the target's and the drafter's distributions as
    /// [`super::accept_verdict`] takes them.
    pub fn verdicts(
        target_ids: &[i32],
        target_probs: &[f32],
        drafter_ids: &[i32],
        drafter_probs: &[f32],
        draft: i32,
        grid: &[f32],
    ) -> Result<Vec<Verdict>, String> {
        if target_ids.len() != target_probs.len() || drafter_ids.len() != drafter_probs.len() {
            return Err("a distribution pairs each id with its probability".to_owned());
        }
        let mut tokens = vec![0i32; grid.len() * grid.len()];
        check(unsafe {
            ffi::ignis_sampled_draft_verdicts(
                target_ids.as_ptr(),
                target_probs.as_ptr(),
                target_ids.len() as u32,
                drafter_ids.as_ptr(),
                drafter_probs.as_ptr(),
                drafter_ids.len() as u32,
                draft,
                grid.as_ptr(),
                grid.len() as u32,
                tokens.as_mut_ptr(),
            )
        })?;
        Ok(tokens.into_iter().map(|token| if token == ACCEPTED { Verdict::Accepted } else { Verdict::Refused(token) }).collect())
    }

    /// GitHub #577: the device's teacher-forced verdict on `recorded` at each
    /// coin uniform of `grid`, the live column's verdict on `live_draft`
    /// being `live` ([`super::forced_verdict`]); the distributions as
    /// [`verdicts`] takes them.
    #[allow(clippy::too_many_arguments)]
    pub fn forced_verdicts(
        target_ids: &[i32],
        target_probs: &[f32],
        drafter_ids: &[i32],
        drafter_probs: &[f32],
        recorded: i32,
        live_draft: i32,
        live: Verdict,
        grid: &[f32],
    ) -> Result<Vec<bool>, String> {
        if target_ids.len() != target_probs.len() || drafter_ids.len() != drafter_probs.len() {
            return Err("a distribution pairs each id with its probability".to_owned());
        }
        let (live_taken, live_token) = match live {
            Verdict::Accepted => (1, live_draft),
            Verdict::Refused(token) => (0, token),
        };
        let mut taken = vec![0u8; grid.len()];
        check(unsafe {
            ffi::ignis_sampled_draft_forced_verdicts(
                target_ids.as_ptr(),
                target_probs.as_ptr(),
                target_ids.len() as u32,
                drafter_ids.as_ptr(),
                drafter_probs.as_ptr(),
                drafter_ids.len() as u32,
                recorded,
                live_draft,
                live_taken,
                live_token,
                grid.as_ptr(),
                grid.len() as u32,
                taken.as_mut_ptr(),
            )
        })?;
        Ok(taken.into_iter().map(|t| t != 0).collect())
    }

    /// The candidate the device draws from `q` at each uniform of `grid`
    /// ([`super::draw_draft`]).
    pub fn draws(q: &[f32], grid: &[f32]) -> Result<Vec<usize>, String> {
        let mut drawn = vec![0i32; grid.len()];
        check(unsafe { ffi::ignis_sampled_draft_draws(q.as_ptr(), q.len() as u32, grid.as_ptr(), grid.len() as u32, drawn.as_mut_ptr()) })?;
        Ok(drawn.into_iter().map(|i| i as usize).collect())
    }

    /// The device's drafter distribution ([`super::drafter_distribution`]).
    pub fn drafter_distribution(scores: &[f32], temperature: f32, top_p: f32) -> Result<Vec<f32>, String> {
        let mut q = vec![0f32; scores.len()];
        check(unsafe { ffi::ignis_sampled_drafter_distribution(scores.as_ptr(), scores.len() as u32, temperature, top_p, q.as_mut_ptr()) })?;
        Ok(q)
    }
}
