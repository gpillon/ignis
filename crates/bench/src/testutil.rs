//! Test doubles the measurement modules' tests share: an engine with prompt
//! reuse (GitHub #426), which their whole runs are measured against.

use crate::client::{Endpoint, Outcome, Request};
use crate::ttft::PromptTemplate;

/// The artifact's chat template in shape: `<|im_start|>user\n` before the
/// content, `<|im_end|>\n<|im_start|>assistant\n` and the four
/// thinking-off tokens after it.
pub(crate) const CHAT_HEADER: usize = 3;
pub(crate) const CHAT_FOOTER: usize = 9;

/// A stand-in for an engine that serves any repeated prefix from a
/// cache, around another endpoint whose outcomes it reports the usage
/// of: the spec's mock (GitHub #426), which covers ignis's **prompt
/// reuse** and more.
///
/// A request claims, of every prompt admitted before it, the longer of
/// two heads (GLOSSARY: prompt checkpoint, KV page):
///
/// - the earlier prompt's **prompt checkpoint** at its generation
///   opener, everything up to the template's `<|im_start|>assistant\n`,
///   when the request's prompt starts with it: for one user message with
///   thinking off, the prompt less its last four tokens
///   (`<think>`, `\n\n`, `</think>`, `\n\n`). The 5090 run's C=4 sample
///   0, the C=1 prompt again, reused 8,188 of its 8,192 tokens that way;
/// - the whole 64-token **KV pages** of the head the two prompts share,
///   short of the request's last token: any repeated prefix, ignis's
///   chained prefix at an opener's page floor among them (8,128 tokens
///   of an 8,192-token prompt, what each of the run's first four
///   32,768-token prefillers reused), and any shorter head too.
///
/// The longest claim wins, as on the server, and nothing is ever given
/// up: the server keeps less, which can only void fewer. Usage is the
/// server's: the prompt as the template encodes it, the claimed head as
/// `cached_tokens`.
pub(crate) struct PromptReuse<'t, E> {
    inner: E,
    template: &'t dyn PromptTemplate,
    admitted: std::sync::Mutex<Vec<Vec<u32>>>,
}

impl<'t, E: Endpoint> PromptReuse<'t, E> {
    const PAGE_TOKENS: usize = 64;
    const AFTER_OPENER: usize = 4;

    pub(crate) fn new(inner: E, template: &'t dyn PromptTemplate) -> Self {
        Self { inner, template, admitted: std::sync::Mutex::new(Vec::new()) }
    }

    /// Admit `req`: the prompt tokens it counts and the head it claims.
    fn admit(&self, req: &Request) -> Result<(u32, u32), String> {
        let tokens = self.template.encode_user_message(&req.prompt)?;
        let mut admitted = self.admitted.lock().unwrap();
        let cached = admitted.iter().map(|earlier| Self::claim(&tokens, earlier)).max().unwrap_or(0);
        let prompt = tokens.len() as u32;
        admitted.push(tokens);
        Ok((prompt, cached as u32))
    }

    /// The head `tokens` claims of an `earlier` prompt.
    fn claim(tokens: &[u32], earlier: &[u32]) -> usize {
        let shared = tokens.iter().zip(earlier).take_while(|(a, b)| a == b).count();
        let opener = earlier.len().saturating_sub(Self::AFTER_OPENER);
        let checkpoint = if opener > 0 && shared >= opener { opener } else { 0 };
        let pages = shared.min(tokens.len().saturating_sub(1)) / Self::PAGE_TOKENS * Self::PAGE_TOKENS;
        checkpoint.max(pages)
    }

    fn report(mut outcome: Outcome, (prompt, cached): (u32, u32)) -> Outcome {
        outcome.prompt_tokens = Some(prompt);
        outcome.cached_prompt_tokens = Some(cached);
        outcome
    }
}

impl<E: Endpoint> Endpoint for PromptReuse<'_, E> {
    fn complete(&self, req: &Request) -> Result<Outcome, String> {
        let usage = self.admit(req)?;
        self.inner.complete(req).map(|outcome| Self::report(outcome, usage))
    }

    fn complete_observed_while(
        &self,
        req: &Request,
        observer: &mut dyn FnMut(f64) -> bool,
        suppress_eos: &[u32],
    ) -> Result<Outcome, String> {
        let usage = self.admit(req)?;
        self.inner
            .complete_observed_while(req, observer, suppress_eos)
            .map(|outcome| Self::report(outcome, usage))
    }
}
