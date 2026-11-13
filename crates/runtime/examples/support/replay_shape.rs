//! The 27B shape the DFlash2 replay examples load at (GitHub #340), and the
//! command-line flags that move it. Kernel-free, so its tests run on CPU
//! (`tests/replay_shape.rs` includes this file); `replay_load.rs` plans and
//! loads it.
//!
//! The shape follows the artifact's body. The trellis one runs `rtx50-16gb`'s
//! values in the 16 GB class's budget, as a real 16 GB card would (owner,
//! 2026-10-24) -- the profile's values restated (`ignis_server`'s
//! `config/profile.rs`, which this crate cannot read), so a change to that
//! profile is made here too. The NVFP4 one runs the same engine shape --
//! context, chunk, rope, window, head -- on the 5090's derived budget at its
//! eight lanes, so a replay on both differs only in the body and the card.
//! GitHub #577: on either body the rounds run under a named draft rule
//! (`--draft-rule`), today's fixed draft unless the flag names the sampled
//! draft, at its default drafter settings.

use ignis_artifact::Body27b;
use ignis_core::{DraftRule, DrafterSampling, ProposalHead, RopeScaling, Speculation, SpeculativeBackend, VramMode};

/// The 16 GB class's VRAM budget (spec 27b-16gb/05), every byte the process
/// holds included.
pub const SIXTEEN_GIB: u64 = 17_179_869_184;
/// `vram.qwen38.headroom_bytes` of the built-in profiles, `1200M`.
pub const QWEN38_HEADROOM: u64 = 1200 << 20;

/// What a load is asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadShape {
    pub vram: VramMode,
    pub max_context: u32,
    /// The decode lanes: the pool's slots and the widest round.
    pub lanes: u32,
    pub prefill_chunk: u32,
    /// The DFlash2 window (full head); 0 for a plain load.
    pub window: u32,
    pub rope: RopeScaling,
    /// GitHub #577: the rounds' draft rule (ADR 0090), at its default drafter
    /// settings.
    pub draft_rule: DraftRule,
}

/// The flags that move a replay's shape off its body's, taken out of the
/// command line before the positional arguments are read: `--lanes N`,
/// `--prefill-chunk N` and (GitHub #577) `--draft-rule fixed|sampled`. Flags
/// rather than the server's environment (`IGNIS_MODEL_PREFILL_CHUNK`,
/// `IGNIS_SPEC_DRAFT_RULE`): a variable the shell happens to carry must not
/// move a leg unseen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeFlags {
    pub lanes: Option<u32>,
    pub prefill_chunk: Option<u32>,
    pub draft_rule: Option<DraftRule>,
}

impl ShapeFlags {
    /// The flags in `args`, removed from it; every other argument stays, in
    /// order.
    pub fn take(args: &mut Vec<String>) -> Result<Self, String> {
        let mut flags = Self::default();
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].clone();
            if !["--lanes", "--prefill-chunk", "--draft-rule"].contains(&flag.as_str()) {
                i += 1;
                continue;
            }
            args.remove(i);
            if i == args.len() {
                return Err(format!("{flag} needs a value"));
            }
            let raw = args.remove(i);
            let number = || raw.parse().map_err(|e| format!("{flag} `{raw}`: {e}"));
            match flag.as_str() {
                "--lanes" => flags.lanes = Some(number()?),
                "--prefill-chunk" => flags.prefill_chunk = Some(number()?),
                _ => flags.draft_rule = Some(DraftRule::parse(&raw).map_err(|e| format!("{flag}: {e}"))?),
            }
        }
        Ok(flags)
    }
}

impl LoadShape {
    /// The shape `body`'s replay runs at: `rtx50-16gb`'s plan in 16 GiB for
    /// the trellis body, and the same engine at eight lanes on the 5090's
    /// derived budget for the NVFP4 one.
    pub fn for_body(body: Body27b) -> Self {
        let sixteen_gb = Self {
            vram: VramMode::Explicit { budget_bytes: SIXTEEN_GIB, allow_oversubscription: false },
            max_context: 262_144,
            lanes: 2,
            prefill_chunk: 1024,
            window: 7,
            rope: RopeScaling::NONE,
            draft_rule: DraftRule::Fixed,
        };
        match body {
            Body27b::Trellis => sixteen_gb,
            Body27b::Nvfp4 => Self { vram: VramMode::Derived { headroom_bytes: QWEN38_HEADROOM }, lanes: 8, ..sixteen_gb },
        }
    }

    /// [`LoadShape::for_body`] at `flags`' lanes and prefill chunk, on the
    /// NVFP4 body only: its `rtx5090` profile serves 2048-token chunks since
    /// GitHub #341, and a smoke at the trellis body's two lanes pairs the two
    /// on more than the body. The trellis body loads as a 16 GB card does, at
    /// two lanes and 1024-token chunks, and is refused any other. Either body
    /// runs `flags`' draft rule (GitHub #577).
    pub fn new(body: Body27b, flags: ShapeFlags) -> Result<Self, String> {
        let shape = Self { draft_rule: flags.draft_rule.unwrap_or_default(), ..Self::for_body(body) };
        let moved = Self {
            lanes: flags.lanes.unwrap_or(shape.lanes),
            prefill_chunk: flags.prefill_chunk.unwrap_or(shape.prefill_chunk),
            ..shape
        };
        if body == Body27b::Trellis && moved != shape {
            return Err(format!(
                "the trellis body loads as a 16 GB card does, at {} lanes and {}-token chunks, not {} and {}",
                shape.lanes, shape.prefill_chunk, moved.lanes, moved.prefill_chunk
            ));
        }
        Ok(moved)
    }

    /// The load's DFlash2 speculation, full head, under the shape's draft
    /// rule; none for a plain load.
    pub fn speculation(&self) -> Option<Speculation> {
        (self.window > 0).then(|| {
            Speculation::new(SpeculativeBackend::Dflash2, self.window)
                .unwrap_or_else(|e| panic!("window {}: {e}", self.window))
                .with_proposal_head(ProposalHead::Full)
                .with_draft_rule(self.draft_rule, DrafterSampling::default())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_flags_come_out_of_the_command_line_wherever_they_stand() {
        let mut line = args(&["exe", "--lanes", "2", "ARTIFACT", "MANIFEST", "--prefill-chunk", "2048", "OUT"]);
        let flags = ShapeFlags::take(&mut line).unwrap();
        assert_eq!(flags, ShapeFlags { lanes: Some(2), prefill_chunk: Some(2048), draft_rule: None });
        assert_eq!(line, args(&["exe", "ARTIFACT", "MANIFEST", "OUT"]));

        let mut line = args(&["exe", "ARTIFACT", "--draft-rule", "sampled", "MANIFEST"]);
        let flags = ShapeFlags::take(&mut line).unwrap();
        assert_eq!(flags, ShapeFlags { draft_rule: Some(DraftRule::Sampled), ..ShapeFlags::default() });
        assert_eq!(line, args(&["exe", "ARTIFACT", "MANIFEST"]));

        let mut line = args(&["exe", "ARTIFACT"]);
        assert_eq!(ShapeFlags::take(&mut line).unwrap(), ShapeFlags::default());
        assert_eq!(line, args(&["exe", "ARTIFACT"]));
    }

    #[test]
    fn a_flag_without_a_number_is_refused() {
        let err = ShapeFlags::take(&mut args(&["exe", "ARTIFACT", "--lanes"])).unwrap_err();
        assert!(err.contains("--lanes needs a value"), "{err}");
        let err = ShapeFlags::take(&mut args(&["exe", "--prefill-chunk", "big", "ARTIFACT"])).unwrap_err();
        assert!(err.contains("--prefill-chunk `big`"), "{err}");
        let err = ShapeFlags::take(&mut args(&["exe", "--draft-rule", "coupled", "ARTIFACT"])).unwrap_err();
        assert!(err.contains("--draft-rule") && err.contains("fixed or sampled"), "{err}");
        let err = ShapeFlags::take(&mut args(&["exe", "ARTIFACT", "--draft-rule"])).unwrap_err();
        assert!(err.contains("--draft-rule needs a value"), "{err}");
    }

    /// Owner, 2026-10-24: every trellis run loads as a 16 GB card would.
    #[test]
    fn the_trellis_body_loads_as_a_sixteen_gb_card_and_takes_no_other_shape() {
        let shape = LoadShape::new(Body27b::Trellis, ShapeFlags::default()).unwrap();
        assert_eq!(
            (shape.vram, shape.max_context, shape.lanes, shape.prefill_chunk, shape.window, shape.rope),
            (
                VramMode::Explicit { budget_bytes: 17_179_869_184, allow_oversubscription: false },
                262_144,
                2,
                1024,
                7,
                RopeScaling::NONE
            )
        );
        // Its own values, named, are its shape.
        let named = ShapeFlags { lanes: Some(2), prefill_chunk: Some(1024), draft_rule: None };
        assert_eq!(LoadShape::new(Body27b::Trellis, named).unwrap(), shape);
        for flags in [ShapeFlags { lanes: Some(8), ..ShapeFlags::default() }, ShapeFlags { prefill_chunk: Some(2048), ..ShapeFlags::default() }] {
            let err = LoadShape::new(Body27b::Trellis, flags).unwrap_err();
            assert!(err.contains("as a 16 GB card does"), "{err}");
        }
    }

    #[test]
    fn the_nvfp4_body_runs_the_trellis_engine_shape_and_moves_by_flag() {
        let trellis = LoadShape::for_body(Body27b::Trellis);
        let shape = LoadShape::new(Body27b::Nvfp4, ShapeFlags::default()).unwrap();
        assert_eq!(shape, LoadShape { vram: VramMode::Derived { headroom_bytes: 1200 << 20 }, lanes: 8, ..trellis });
        let moved = LoadShape::new(Body27b::Nvfp4, ShapeFlags { lanes: Some(2), prefill_chunk: Some(2048), draft_rule: None }).unwrap();
        assert_eq!(moved, LoadShape { lanes: 2, prefill_chunk: 2048, ..shape });
    }

    #[test]
    fn the_window_is_dflash2_with_the_full_head_and_none_without_one() {
        let shape = LoadShape::for_body(Body27b::Trellis);
        let speculation = shape.speculation().expect("a window of 7");
        assert_eq!((speculation.backend(), speculation.draft_tokens()), (SpeculativeBackend::Dflash2, 7));
        assert_eq!(speculation.proposal_head(), ProposalHead::Full);
        assert_eq!(LoadShape { window: 0, ..shape }.speculation(), None);
    }

    /// GitHub #577: either body runs either rule, today's unless named, the
    /// sampled draft at its default drafter settings.
    #[test]
    fn the_rounds_run_under_the_named_draft_rule_on_either_body() {
        for body in [Body27b::Trellis, Body27b::Nvfp4] {
            let today = LoadShape::new(body, ShapeFlags::default()).unwrap();
            assert_eq!(today.speculation().unwrap().draft_rule(), DraftRule::Fixed);
            let flags = ShapeFlags { draft_rule: Some(DraftRule::Sampled), ..ShapeFlags::default() };
            let sampled = LoadShape::new(body, flags).unwrap();
            assert_eq!(sampled, LoadShape { draft_rule: DraftRule::Sampled, ..today });
            let speculation = sampled.speculation().unwrap();
            assert_eq!((speculation.draft_rule(), speculation.drafter_sampling()), (DraftRule::Sampled, DrafterSampling::default()));
        }
    }
}
