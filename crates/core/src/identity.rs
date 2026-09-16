//! What a retained blob is, and what a retained entry is *called*
//! (GitHub #189, ADR 0029; spec `.scratch/kv-reuse/specs/01-cross-request-reuse.md`
//! §"Identity and the Tier 2 seam").
//!
//! Two independent things live here, and keeping them apart is the point:
//!
//! - a **blob identity** — the *compatibility* fact: which artifact, KV
//!   format, blob layout and drafter the state was produced under. It answers
//!   "may these bytes be written into a sequence at all", and a mismatch is a
//!   refusal before a single byte moves.
//! - a **match key** — the *content* fact: a hash chain over the prompt's
//!   token ids, with each media item's identity mixed in at its placeholder
//!   span. It answers "is this retained state the history this request is
//!   asking for", and it is what a retained entry is addressed by.
//!
//! Neither is ever a [`RequestId`](crate::types::RequestId). A request id is a
//! *backend handle* — the name the compute seam gives the device bytes a
//! finished request left behind — and ADR 0029's rule is about the key: a
//! wrong session id could hand one conversation another's state, a content
//! match can only hand over identical history.
//!
//! **Why a chain rather than a hash of the whole prompt.** A claimant does not
//! ask "is this exact prompt retained", it asks "is any retained *prefix* of
//! my prompt retained". A chain answers every prefix length in one forward
//! pass ([`PromptContent::keys_at`]), and — the reason it matters for Tier 2 —
//! it lets a blob name its own content in sixteen bytes instead of carrying
//! sixty thousand token ids into KV-RAM or onto disk.
//!
//! **Why not a cryptographic hash.** The chain runs once per token of every
//! admitted prompt, on the scheduler thread; SHA-256 per token would cost tens
//! of milliseconds on a 60K-token conversation, against a 128-bit mix's
//! fraction of one. A false match needs a *preimage* on a key the asker can
//! never observe — the keys are internal, and to compute a colliding prompt
//! one would first have to know the tokens one is trying to collide with, at
//! which point one already has the history. Accidental collision over the few
//! dozen entries a pool holds is nil at 128 bits. The algorithm is versioned
//! ([`MATCH_KEY_VERSION`]) and the version is mixed into the chain's seed, so
//! changing it turns every old key into a **miss**, never a wrong hit.
//!
//! **The media slot** ([`MediaKey`]) is filled by GitHub #193. Its shape is
//! settled here so that slice adds a frontend adapter rather than a new key:
//! an item's identity is absorbed at its span's **first** placeholder token,
//! so every prefix that covers so much as one placeholder has already
//! committed to the whole item — two prompts whose token ids are equal and
//! whose images differ cannot share a key anywhere past that point, whatever
//! rule governs where a checkpoint is cut.

use crate::kv_format::KvFormat;
use crate::speculation::Speculation;
use crate::types::TokenId;
use crate::vision::MediaItem;

/// The match-key algorithm's version, mixed into the chain's seed.
///
/// A change here is a **miss**, never a wrong hit: no key computed under one
/// version can equal one computed under another, so retained state produced by
/// an older ignis is simply never matched.
pub const MATCH_KEY_VERSION: u32 = 1;

// ── The compatibility identity ───────────────────────────────────────────────

/// A content hash of the artifact a load runs on.
///
/// **Structural, not a hash of the weights.** The v2 container deliberately
/// carries no per-tensor digest (`ignis_artifact::checksum`'s module doc), and
/// hashing tens of gigabytes at startup to get one is not a trade this feature
/// can make. What [`ignis_artifact::Reader::content_hash`] hashes instead is
/// the container's declared structure — its identity, its size, and every
/// object's name, kind, numeric format, layout, offset and length. A
/// re-quantization that changed weight bytes while preserving every name,
/// format and offset would pass it. That is a proxy, named as one here so no
/// caller mistakes it for a payload digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactHash([u8; 32]);

impl ArtifactHash {
    /// The hash of an artifact nobody measured — what a backend that retains
    /// nothing reports. It equals only itself, so it never makes two
    /// *different* loads look compatible: a backend with no artifact has no
    /// blobs to be handed either.
    pub const UNKNOWN: Self = Self([0; 32]);

    /// Wrap a 32-byte digest.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The digest's bytes (a blob header writes these).
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for ArtifactHash {
    /// The first eight bytes, hex — enough to tell two artifacts apart in a
    /// refusal message without filling a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0[..8] {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// The compatibility identity of retained state: what a blob was produced
/// under, carried in the blob header *and* in the device entry (ADR 0029).
///
/// Four facts, and the test for whether a fifth belongs is not "does this
/// change performance" but "would state produced under one value be wrong to
/// write into a sequence built under the other". The operator knobs that
/// decide *how much* is retained or *how fast* it is produced — the bind
/// address, the prefill chunk width, the concurrency, every byte budget,
/// `--prompt-reuse` itself — are deliberately absent: a device pool budget
/// derived from free VRAM varies from one start to the next, and an identity
/// that moved with it would refuse every blob after a reboot for no reason.
///
/// Vision is absent for the same test: a `--vision` load binds an encoder and
/// reserves its workspace, but the sequence state a checkpoint holds has the
/// same sections and the same meaning either way. What an image changes is
/// *content*, and content is the [`MatchKey`]'s job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobIdentity {
    /// The artifact the state was produced by.
    pub artifact: ArtifactHash,
    /// The KV cache format the pages are written in (ADR 0022).
    pub kv_format: KvFormat,
    /// The leaf's snapshot blob layout version
    /// (`ignis_seq_snapshot_format_version`). Read from the backend, never
    /// restated in Rust: the section table lives inside the leaf, and a
    /// version this crate held its own copy of would not move when the table
    /// did (ADR 0024).
    pub layout_version: u32,
    /// The drafter bound at load: its backend and its draft window, or `None`
    /// for a load with no drafter. Both halves matter — a blob taken on a pool
    /// with per-slot drafter sections cannot be written into one without them,
    /// and a different window is a different section.
    pub drafter: Option<Speculation>,
}

impl BlobIdentity {
    /// The identity of a backend that produces no retained state: no artifact,
    /// the default format, layout version 0, no drafter.
    ///
    /// It is a real value rather than an `Option<BlobIdentity>` so that every
    /// read of an identity is the same read. A pool holding it retains
    /// entries under it and matches them exactly as any other pool does — what
    /// keeps a CPU-only `Compute` behaving as it did before identities
    /// existed — and can never accept a blob from a real load, because
    /// [`ArtifactHash::UNKNOWN`] is not any real artifact's hash.
    pub const UNSET: Self = Self {
        artifact: ArtifactHash::UNKNOWN,
        kv_format: KvFormat::HqE8_2b,
        layout_version: 0,
        drafter: None,
    };

    /// The identity of state produced by a load of `artifact` under
    /// `kv_format` with `drafter`, whose leaf writes blob `layout_version`.
    ///
    /// The one place the identity is assembled. Every caller that has a load's
    /// options in front of it — the server's engine shape, the CUDA leaf —
    /// reaches this function, so which of a load's many knobs are part of the
    /// identity is decided here and nowhere else.
    pub fn of_load(
        artifact: ArtifactHash,
        kv_format: KvFormat,
        drafter: Option<Speculation>,
        layout_version: u32,
    ) -> Self {
        Self {
            artifact,
            kv_format,
            layout_version,
            drafter,
        }
    }

    /// Whether state carrying `blob`'s identity may be written into a sequence
    /// built under this one.
    ///
    /// The refusal names the field that differs, because "refused" alone is
    /// indistinguishable from a bug at three in the morning: an operator who
    /// restarted with `--kv-format bf16` and lost every checkpoint should read
    /// *why* in one line.
    pub fn accepts(&self, blob: &BlobIdentity) -> Result<(), IdentityMismatch> {
        let field = if self.artifact != blob.artifact {
            IdentityField::Artifact
        } else if self.kv_format != blob.kv_format {
            IdentityField::KvFormat
        } else if self.layout_version != blob.layout_version {
            IdentityField::LayoutVersion
        } else if self.drafter != blob.drafter {
            IdentityField::Drafter
        } else {
            return Ok(());
        };
        Err(IdentityMismatch {
            field,
            expected: self.describe(field),
            found: blob.describe(field),
        })
    }

    /// One field of this identity, spelled for a refusal message.
    fn describe(&self, field: IdentityField) -> String {
        match field {
            IdentityField::Artifact => self.artifact.to_string(),
            IdentityField::KvFormat => self.kv_format.as_str().to_string(),
            IdentityField::LayoutVersion => self.layout_version.to_string(),
            IdentityField::Drafter => match self.drafter {
                None => "none".to_string(),
                Some(spec) => format!("{} window {}", spec.backend().as_str(), spec.draft_tokens()),
            },
        }
    }
}

/// Which part of a [`BlobIdentity`] a refusal is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityField {
    /// The artifact the state was produced by.
    Artifact,
    /// The KV cache format.
    KvFormat,
    /// The leaf's blob layout version.
    LayoutVersion,
    /// The drafter's presence or its draft window.
    Drafter,
}

impl IdentityField {
    /// The field's name, for a log line.
    pub fn as_str(self) -> &'static str {
        match self {
            IdentityField::Artifact => "artifact",
            IdentityField::KvFormat => "kv_format",
            IdentityField::LayoutVersion => "layout_version",
            IdentityField::Drafter => "drafter",
        }
    }
}

/// A blob refused: which field disagreed, and what each side said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityMismatch {
    /// The disagreeing field.
    pub field: IdentityField,
    /// What this load has.
    pub expected: String,
    /// What the blob was taken under.
    pub found: String,
}

impl std::fmt::Display for IdentityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "retained state refused: {} is {} here, {} in the blob",
            self.field.as_str(),
            self.expected,
            self.found
        )
    }
}

/// What a retained blob carries so a tier below the device can hold it without
/// holding its history: the identity it was taken under, and the content it
/// covers.
///
/// This is the whole of the Tier 2 seam. KV-RAM (#190) keeps one of these
/// beside each host allocation; a KV-disk tier would write exactly this and
/// nothing more, because a blob that names its own artifact, layout and
/// content is a blob a *later process* can decide about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobHeader {
    /// What the state was produced under.
    pub identity: BlobIdentity,
    /// The content the state covers.
    pub key: MatchKey,
    /// Prompt tokens the key covers.
    pub tokens: u32,
}

// ── The content match key ────────────────────────────────────────────────────

/// A retained entry's content name: 128 bits of hash chain over the prompt
/// tokens it covers, with each media item's identity mixed in.
///
/// Opaque by construction — there is no way back to the tokens, which is the
/// property that lets a blob header travel to another tier without carrying a
/// conversation with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchKey {
    lo: u64,
    hi: u64,
}

impl MatchKey {
    /// The key of the empty prompt — the chain's seed.
    pub fn empty() -> Self {
        MatchKeyChain::new().key()
    }

    /// The key's sixteen bytes, little-endian (a blob header writes these).
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.lo.to_le_bytes());
        out[8..].copy_from_slice(&self.hi.to_le_bytes());
        out
    }

    /// The inverse of [`MatchKey::to_bytes`].
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        lo.copy_from_slice(&bytes[..8]);
        hi.copy_from_slice(&bytes[8..]);
        Self {
            lo: u64::from_le_bytes(lo),
            hi: u64::from_le_bytes(hi),
        }
    }
}

impl std::fmt::Display for MatchKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}{:016x}", self.hi, self.lo)
    }
}

/// One media item's identity at its placeholder span: the media slot of the
/// match key (GitHub #193 fills it).
///
/// Deliberately **not** [`MediaItem`] itself. The item carries its patch rows,
/// megabytes of them, and an identity computed over those would be both slow
/// and wrong — two encodes of one image must name the same item. What decides
/// what the placeholders mean is the processor's content digest (GitHub #176:
/// the SHA-256 of the acquired bytes) and the grid they were packed into, and
/// those two are all that is here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaKey {
    /// The item's first placeholder token, as a prompt offset.
    pub begin: u32,
    /// Placeholder tokens the item covers.
    pub count: u32,
    /// The processor's per-item content digest (GitHub #176).
    pub digest: [u8; 32],
    /// The patch grid the item was packed into, `[t, h, w]`: the same bytes
    /// resized differently are a different item, because the placeholders they
    /// expand to are different.
    pub grid: [u32; 3],
}

impl From<&MediaItem> for MediaKey {
    fn from(item: &MediaItem) -> Self {
        Self {
            begin: item.token_span.begin as u32,
            count: item.token_span.count as u32,
            digest: item.content_digest,
            grid: [item.grid.t, item.grid.h, item.grid.w],
        }
    }
}

/// The content a match key is computed over: a prompt's token ids and the
/// media items inside it.
///
/// Invariant the caller keeps: `media` is in prompt order, spans ascending and
/// non-overlapping, every span inside `tokens` — which is exactly the
/// invariant [`crate::vision::Multimodal`] already checks at the seam where a
/// prepared prompt becomes a request.
#[derive(Debug, Clone, Copy)]
pub struct PromptContent<'a> {
    tokens: &'a [TokenId],
    media: &'a [MediaKey],
}

impl<'a> PromptContent<'a> {
    /// A text-only prompt — today's every request.
    pub fn text(tokens: &'a [TokenId]) -> Self {
        Self { tokens, media: &[] }
    }

    /// A prompt with media items inside it (GitHub #193).
    pub fn new(tokens: &'a [TokenId], media: &'a [MediaKey]) -> Self {
        Self { tokens, media }
    }

    /// The prompt's length in tokens.
    pub fn tokens(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// The content key of this prompt's first `at` tokens.
    pub fn key_at(&self, at: u32) -> MatchKey {
        self.keys_at(&[at])[0]
    }

    /// The content keys of this prompt's prefixes at each of `lengths`, in
    /// `lengths`' own order, computed in **one** forward pass.
    ///
    /// This is why the key is a chain: a pool asks about every retained
    /// entry's length at once, and answering each one separately would walk
    /// the prompt once per entry. A length past the prompt's end answers the
    /// key of the whole prompt — no such entry can be a prefix of it, and the
    /// caller that would ask has already excluded them by length.
    pub fn keys_at(&self, lengths: &[u32]) -> Vec<MatchKey> {
        let mut order: Vec<usize> = (0..lengths.len()).collect();
        order.sort_unstable_by_key(|&i| lengths[i]);
        let mut out = vec![MatchKey::empty(); lengths.len()];
        let mut chain = MatchKeyChain::new();
        let mut next = 0;
        while next < order.len() && lengths[order[next]] == 0 {
            out[order[next]] = chain.key();
            next += 1;
        }
        let mut item = 0;
        for at in 0..self.tokens.len() as u32 {
            // The item is absorbed *before* its first placeholder, so every
            // prefix covering one placeholder has committed to all of it.
            while item < self.media.len() && self.media[item].begin == at {
                chain.push_media(&self.media[item]);
                item += 1;
            }
            chain.push_token(self.tokens[at as usize]);
            while next < order.len() && lengths[order[next]] == at + 1 {
                out[order[next]] = chain.key();
                next += 1;
            }
        }
        let whole = chain.key();
        for &i in &order[next..] {
            out[i] = whole;
        }
        out
    }

    /// Whether a prefix of `at` tokens would end **inside** a media item's
    /// placeholder span.
    ///
    /// Such a point is not a legal boundary for retained state (GitHub #193):
    /// half an image's placeholders are warm and the encoder output the rest
    /// needs is gone. The key itself stays well defined there — the item was
    /// absorbed at its first placeholder — so this is a question a caller asks
    /// and acts on, not a failure mode of the key.
    pub fn splits_media(&self, at: u32) -> bool {
        self.media
            .iter()
            .any(|m| m.begin < at && at < m.begin + m.count)
    }
}

/// The match key's hash chain, absorbed forward one token at a time.
///
/// Exposed because a producer that streams its prompt (a frontend that
/// tokenizes as it renders) can carry one of these instead of the prompt, and
/// because a blob's key is nothing but the chain's value at its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchKeyChain {
    lo: u64,
    hi: u64,
    tokens: u32,
}

/// The chain's seeds — the first two 64-bit words of pi's fractional part,
/// nothing-up-my-sleeve constants.
const SEED_LO: u64 = 0x243F_6A88_85A3_08D3;
const SEED_HI: u64 = 0x1319_8A2E_0370_7344;

/// Domain tags, so a token can never be absorbed as a media word or the
/// reverse.
const TAG_TOKEN: u64 = 0x01;
const TAG_MEDIA: u64 = 0x02;

/// splitmix64's finalizer: a bijection with full avalanche on 64 bits.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

impl Default for MatchKeyChain {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchKeyChain {
    /// A chain over the empty prompt, seeded with [`MATCH_KEY_VERSION`].
    pub fn new() -> Self {
        Self {
            lo: SEED_LO ^ MATCH_KEY_VERSION as u64,
            hi: SEED_HI,
            tokens: 0,
        }
    }

    /// Absorb one 64-bit word. The two lanes are chained through each other,
    /// so colliding the pair takes work in both: `hi` depends on every `lo`
    /// the chain has passed through, not on the words directly.
    fn absorb(&mut self, word: u64) {
        self.lo = mix(self.lo ^ word);
        self.hi = mix(self.hi.rotate_left(31) ^ self.lo);
    }

    /// Absorb one prompt token.
    pub fn push_token(&mut self, token: TokenId) {
        self.absorb(TAG_TOKEN ^ ((token as u64) << 8));
        self.tokens += 1;
    }

    /// Absorb one media item's identity, at its first placeholder token.
    ///
    /// Does not advance the token count: the item's placeholders are ordinary
    /// prompt tokens and are pushed after it.
    pub fn push_media(&mut self, item: &MediaKey) {
        self.absorb(TAG_MEDIA);
        self.absorb(item.count as u64);
        for word in item.digest.chunks_exact(8) {
            self.absorb(u64::from_le_bytes(word.try_into().expect("eight bytes")));
        }
        for axis in item.grid {
            self.absorb(axis as u64);
        }
    }

    /// The key of everything absorbed so far.
    pub fn key(&self) -> MatchKey {
        MatchKey {
            lo: self.lo,
            hi: self.hi,
        }
    }

    /// Prompt tokens absorbed so far.
    pub fn tokens(&self) -> u32 {
        self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculation::SpeculativeBackend;

    fn artifact(byte: u8) -> ArtifactHash {
        ArtifactHash::from_bytes([byte; 32])
    }

    fn identity() -> BlobIdentity {
        BlobIdentity::of_load(artifact(1), KvFormat::HqE8_2b, None, 2)
    }

    fn dflash2(draft_tokens: u32) -> Speculation {
        Speculation::new(SpeculativeBackend::Dflash2, draft_tokens).unwrap()
    }

    /// A media item over `count` placeholders at `begin`, with a synthetic
    /// digest — the shape GitHub #193 will build from a real processor.
    fn media(begin: u32, count: u32, digest: u8) -> MediaKey {
        MediaKey {
            begin,
            count,
            digest: [digest; 32],
            grid: [1, 8, 8],
        }
    }

    #[test]
    fn a_key_names_the_tokens_it_covers() {
        let a: Vec<TokenId> = (1..=100).collect();
        let mut b = a.clone();
        b[57] = 999;
        assert_eq!(
            PromptContent::text(&a).key_at(100),
            PromptContent::text(&a).key_at(100),
            "the same content is the same key"
        );
        assert_ne!(
            PromptContent::text(&a).key_at(100),
            PromptContent::text(&b).key_at(100),
            "one different token is a different key"
        );
    }

    #[test]
    fn a_prefix_key_is_the_key_that_prefix_would_have_on_its_own() {
        // The whole point of a chain: an entry retained over turn N's prompt
        // and a claimant asking about the first 100 tokens of turn N+1's
        // longer prompt compute the same value, without either holding the
        // other's tokens.
        let long: Vec<TokenId> = (1..=250).collect();
        for at in [0, 1, 99, 100, 250] {
            assert_eq!(
                PromptContent::text(&long).key_at(at),
                PromptContent::text(&long[..at as usize]).key_at(at),
                "the key at {at} does not depend on what follows it"
            );
        }
    }

    #[test]
    fn different_lengths_of_one_prompt_are_different_keys() {
        let tokens: Vec<TokenId> = (1..=64).collect();
        let prompt = PromptContent::text(&tokens);
        let mut seen = Vec::new();
        for at in 0..=64 {
            let key = prompt.key_at(at);
            assert!(!seen.contains(&key), "length {at} repeats an earlier key");
            seen.push(key);
        }
    }

    #[test]
    fn keys_at_answers_every_length_in_the_callers_order() {
        let tokens: Vec<TokenId> = (1..=200).collect();
        let prompt = PromptContent::text(&tokens);
        let lengths = [150, 0, 200, 17];
        let keys = prompt.keys_at(&lengths);
        for (n, &at) in lengths.iter().enumerate() {
            assert_eq!(keys[n], prompt.key_at(at), "length {at} out of order");
        }
    }

    #[test]
    fn a_length_past_the_prompt_answers_the_whole_prompt() {
        // No retained entry longer than the prompt can be a prefix of it, so
        // the caller has already excluded them; the chain still has to answer
        // something rather than index out of bounds.
        let tokens: Vec<TokenId> = (1..=10).collect();
        let prompt = PromptContent::text(&tokens);
        assert_eq!(prompt.key_at(99), prompt.key_at(10));
    }

    #[test]
    fn two_prompts_with_equal_tokens_and_different_media_never_match() {
        // Spec §11 / ADR 0029: "I am never answered about a picture I did not
        // send." The token ids are identical — the placeholders are the same
        // token repeated — and only the images differ.
        let tokens: Vec<TokenId> = (1..=40).collect();
        let mine = [media(10, 16, 0xAA)];
        let yours = [media(10, 16, 0xBB)];
        let mine = PromptContent::new(&tokens, &mine);
        let yours = PromptContent::new(&tokens, &yours);
        assert_ne!(mine.key_at(40), yours.key_at(40));
        assert_ne!(
            mine.key_at(40),
            PromptContent::text(&tokens).key_at(40),
            "an image is not the same as no image"
        );
    }

    #[test]
    fn a_media_item_is_committed_at_its_first_placeholder() {
        // The slot's shape, and why #193 can fill it without redesigning the
        // key: before the item's span the two prompts are the same history and
        // *should* match; from its first placeholder on they never can, so no
        // rule about where a checkpoint may be cut is load-bearing for
        // correctness.
        let tokens: Vec<TokenId> = (1..=40).collect();
        let mine = [media(10, 16, 0xAA)];
        let yours = [media(10, 16, 0xBB)];
        let mine = PromptContent::new(&tokens, &mine);
        let yours = PromptContent::new(&tokens, &yours);
        assert_eq!(mine.key_at(10), yours.key_at(10), "before the image");
        for at in 11..=26 {
            assert_ne!(mine.key_at(at), yours.key_at(at), "at {at}, inside it");
        }
    }

    #[test]
    fn the_same_bytes_in_a_different_grid_are_a_different_item() {
        let tokens: Vec<TokenId> = (1..=40).collect();
        let small = [media(10, 16, 0xAA)];
        let large = [MediaKey {
            grid: [1, 16, 16],
            ..media(10, 16, 0xAA)
        }];
        assert_ne!(
            PromptContent::new(&tokens, &small).key_at(40),
            PromptContent::new(&tokens, &large).key_at(40)
        );
    }

    #[test]
    fn two_images_in_one_prompt_are_committed_in_order() {
        let tokens: Vec<TokenId> = (1..=40).collect();
        let forward = [media(5, 8, 0xAA), media(20, 8, 0xBB)];
        let swapped = [media(5, 8, 0xBB), media(20, 8, 0xAA)];
        assert_ne!(
            PromptContent::new(&tokens, &forward).key_at(40),
            PromptContent::new(&tokens, &swapped).key_at(40)
        );
    }

    #[test]
    fn a_prefix_ending_inside_an_image_is_named_as_one() {
        let tokens: Vec<TokenId> = (1..=40).collect();
        let items = [media(10, 16, 0xAA)];
        let prompt = PromptContent::new(&tokens, &items);
        assert!(!prompt.splits_media(10), "the token before the first patch");
        assert!(prompt.splits_media(11), "one placeholder in");
        assert!(prompt.splits_media(25));
        assert!(!prompt.splits_media(26), "the token after the last patch");
        assert!(!PromptContent::text(&tokens).splits_media(11));
    }

    #[test]
    fn a_key_survives_the_round_trip_a_blob_header_makes() {
        let tokens: Vec<TokenId> = (1..=100).collect();
        let key = PromptContent::text(&tokens).key_at(100);
        assert_eq!(MatchKey::from_bytes(key.to_bytes()), key);
        assert_eq!(key.to_string().len(), 32, "128 bits of hex");
    }

    #[test]
    fn an_identity_accepts_itself() {
        assert_eq!(identity().accepts(&identity()), Ok(()));
    }

    #[test]
    fn a_blob_from_another_artifact_is_refused() {
        let other = BlobIdentity {
            artifact: artifact(2),
            ..identity()
        };
        let refused = identity().accepts(&other).expect_err("refused");
        assert_eq!(refused.field, IdentityField::Artifact);
        assert!(
            refused.to_string().contains("artifact"),
            "the refusal names the field: {refused}"
        );
    }

    #[test]
    fn a_blob_in_another_kv_format_is_refused() {
        let other = BlobIdentity {
            kv_format: KvFormat::Bf16,
            ..identity()
        };
        assert_eq!(
            identity().accepts(&other).expect_err("refused").field,
            IdentityField::KvFormat
        );
    }

    #[test]
    fn a_blob_from_another_layout_version_is_refused() {
        // The leaf's section table changed under it: the bytes are the right
        // model's, and they mean something else now.
        let other = BlobIdentity {
            layout_version: 3,
            ..identity()
        };
        assert_eq!(
            identity().accepts(&other).expect_err("refused").field,
            IdentityField::LayoutVersion
        );
    }

    #[test]
    fn a_blob_from_another_drafter_configuration_is_refused() {
        let with_drafter = BlobIdentity {
            drafter: Some(dflash2(4)),
            ..identity()
        };
        // Presence, both ways.
        assert_eq!(
            identity().accepts(&with_drafter).expect_err("refused").field,
            IdentityField::Drafter
        );
        assert_eq!(
            with_drafter.accepts(&identity()).expect_err("refused").field,
            IdentityField::Drafter
        );
        // And the window, at equal presence: a wider draft window is a
        // different per-slot section.
        let wider = BlobIdentity {
            drafter: Some(dflash2(6)),
            ..identity()
        };
        let refused = with_drafter.accepts(&wider).expect_err("refused");
        assert_eq!(refused.field, IdentityField::Drafter);
        assert!(
            refused.expected.contains('4') && refused.found.contains('6'),
            "the refusal spells both windows: {refused}"
        );
    }

    #[test]
    fn the_identity_is_assembled_from_the_load_and_nothing_else() {
        // The four facts, in one place. A knob that is not one of these
        // cannot reach the identity, because there is no argument to pass it
        // through.
        let built = BlobIdentity::of_load(artifact(1), KvFormat::Bf16, Some(dflash2(3)), 2);
        assert_eq!(built.artifact, artifact(1));
        assert_eq!(built.kv_format, KvFormat::Bf16);
        assert_eq!(built.drafter, Some(dflash2(3)));
        assert_eq!(built.layout_version, 2);
    }

    #[test]
    fn an_unset_identity_never_accepts_a_real_load() {
        // A CPU-only backend retains state under `UNSET` and matches it
        // happily; what it must never do is accept bytes from a real one.
        assert_eq!(BlobIdentity::UNSET.accepts(&BlobIdentity::UNSET), Ok(()));
        assert!(BlobIdentity::UNSET.accepts(&identity()).is_err());
        assert!(identity().accepts(&BlobIdentity::UNSET).is_err());
    }
}
