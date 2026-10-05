//! The G1 run of spec flash-next/04 acceptance 4: every canary of the
//! Flash-Next fixture (`references/g1_flash_next.json`, layout.md §11) fed
//! once, teacher-forced, through [`SpanLogits`], and its argmax rows scored
//! against the expected-argmax column by [`crate::oracle::score_canary`].
//!
//! One prefill per canary gives every prediction: row `p − 1 + i` (`p` the
//! recorded prompt's length) is the distribution after the prompt and the
//! canary's first `i` tokens, the prefix `expected_argmax[i]` was recorded
//! after.

use super::kld::DenseRow;
use super::{SpanLogits, for_each_chunk};
use crate::oracle::{
    Fixture, FixturePrompt, TeacherForcedResult, meets_g1_floor, overall_teacher_forced_agreement, score_canary,
};

/// The suite's result: every canary's (with its mismatches) and the overall
/// agreement judged against the G1 floor.
#[derive(Debug, Clone, PartialEq)]
pub struct G1Run {
    pub results: Vec<TeacherForcedResult>,
    pub overall: f64,
    pub pass: bool,
}

/// The engine's teacher-forced argmax at each of `prompt`'s positions.
pub fn predictions(engine: &mut dyn SpanLogits, prompt: &FixturePrompt) -> Result<Vec<u32>, String> {
    let prompt_tokens = prompt.prompt_tokens(|_| {
        Err(format!("canary {}: no recorded prompt_token_ids to feed (the fixture renders none here)", prompt.id))
    })?;
    if prompt_tokens.is_empty() {
        return Err(format!("canary {}: an empty recorded prompt", prompt.id));
    }
    let n = prompt.token_ids.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    // The last canary token's own row predicts past the canary: never scored.
    let fed: Vec<u32> = prompt_tokens.iter().chain(&prompt.token_ids[..n - 1]).copied().collect();
    let first_scored = prompt_tokens.len() - 1;
    let vocab = engine.vocab();
    let mut out = Vec::with_capacity(n);
    for_each_chunk(engine, &fed, |first, rows| {
        for (i, row) in rows.chunks_exact(vocab).enumerate() {
            if first + i >= first_scored {
                let row = DenseRow::new(row).map_err(|e| format!("position {}: {e}", first + i))?;
                out.push(row.argmax());
            }
        }
        Ok(())
    })
    .map_err(|e| format!("canary {}: {e}", prompt.id))?;
    Ok(out)
}

/// Runs and scores every canary of `fixture`, all positions of each.
pub fn run(engine: &mut dyn SpanLogits, fixture: &Fixture) -> Result<G1Run, String> {
    let mut results = Vec::with_capacity(fixture.prompts.len());
    for prompt in &fixture.prompts {
        prompt.expected_tokens()?;
        let predicted = predictions(engine, prompt)?;
        results.push(score_canary(prompt, &predicted, prompt.token_ids.len())?);
    }
    let overall = overall_teacher_forced_agreement(&results);
    Ok(G1Run { results, overall, pass: meets_g1_floor(overall) })
}
