//! The image leaf's runner (spec image-model/04 and 05, GitHub #321): the
//! server's adapter between [`ImageRunner`], the calls an image job makes,
//! and `ignis_core::image_leaf::ImageLeaf`, the leaf's own. It holds what the
//! leaf leaves to its caller (spec image-model/03 §Seams): the artifact's
//! tokenizer, templates and image processor ([`EncoderFrontend`]) on the
//! host, and the job's condition prefixes, by [`ConditionId`].
//!
//! - **The count.** `condition_tokens` is [`EncoderFrontend::count`]'s
//!   prefix, the very count the leaf's `encode_condition` refuses by, or why
//!   there is none ([`ignis_core::image::ConditionUnfit`]). A condition size the encoder's
//!   processor would not keep (a panorama past 200:1) is refused, in the
//!   processor's words. A condition within the plan's budget that the leaf's
//!   arena has no room to encode (`ImageLeaf::condition_fits`), or whose
//!   images leave the target no RoPE position ([`target_rope_layout`]), has
//!   no room: the server's fit (`images::budget::fit`) scales the images for
//!   those limits too. With no image to scale, it is refused.
//! - **The latents.** A job keeps its target's latents on the host,
//!   channel-major ([`TargetLatents`]); the leaf keeps the started image's in
//!   its slot, packed `[y · w + x][c]` ([`packed`]). Each step sends them up
//!   and brings them back; an image starts (its RoPE table after the prefix)
//!   at its first step over a prefix, and the decode reads the latents the
//!   job hands it.
//! - **The schedule.** A step is the leaf's step from the σ to the σ_next the
//!   job names ([`step_of`]): the plan's sigmas are the leaf's sampler's.
//! - **Errors** ([`image_error`]): the leaf's refusal is the request's
//!   ([`ImageError::Refused`], a 400), its failure the server's
//!   ([`ImageError::Leaf`], a 500).
//! - **Release.** A prefix is freed by `release_condition` or, when the
//!   runner is dropped holding it (its model unloaded or switched away
//!   mid-job, so the job's own release finds no runner), by the runner's
//!   `Drop`: the leaf's own drop frees its weights and arena, not prefixes.

use ignis_core::image::{ImageError, ImageSize, TargetLatents, LATENT_CHANNELS, LATENT_SCALE};
use ignis_core::image_leaf::ImageLeafError;

#[cfg(feature = "cuda")]
pub use device::LeafRunner;

/// The largest target the load makes, in latent cells: the largest of
/// `sizes` (`image.sizes`, spec image-model/05 §Sizes) within `cap`
/// (`image.max_target_cells`), or the cap when none is. One number sizes the
/// leaf's arena (`ImageLeafOptions::max_target_tokens`) and is planned as
/// [`ignis_core::image::ImagePlan::max_target_cells`], so a size the routes
/// serve is one the arena holds, and a size past the cap — not served, the
/// load warns — never grows the arena; the defaults' largest is 2400x1792,
/// 16,800 cells, the cap's default.
pub fn max_target_cells(sizes: &[ImageSize], cap: u32) -> usize {
    let cap = cap as usize;
    sizes.iter().map(|size| size.latent_cells()).filter(|&cells| cells <= cap).max().unwrap_or(cap)
}

/// The image leaf's plan at load (spec image-model/04 §Load and plan) as
/// one `ignis.image.plan` event, as the text models' plans are: the VRAM
/// budget and its mode, what it holds — the weights, the arena (sized for
/// `max_target_cells`), the condition budget in tokens and bytes — the
/// memory free at start and the load's time; its message the plan's line.
pub fn log_plan(plan: &ignis_core::image_leaf::ImagePlan, max_target_cells: usize, load_s: f64) {
    tracing::info!(
        name: "ignis.image.plan",
        mode = plan.mode.as_str(),
        free_at_start_bytes = plan.free_at_start_bytes,
        budget_bytes = plan.budget_bytes,
        weights_bytes = plan.weights_bytes,
        arena_bytes = plan.arena_bytes,
        condition_budget_tokens = plan.condition_budget_tokens,
        condition_budget_bytes = plan.condition_budget_bytes,
        max_target_cells,
        load_s,
        "{}",
        plan.line()
    );
}

/// The joint sequence a target starts over, for the leaf's RoPE rule
/// (`ignis_core::image_dit::rope_positions`, which `ImageLeaf::start_image`
/// runs): a prefix of `prefix` tokens with condition images of `images`
/// (sides multiples of 16), laid out text first, then each image's block,
/// then a one-cell target block at the prefix's end — its tokens, the
/// blocks' starts and their (height, width) in latent cells.
///
/// The leaf's own layout interleaves the text and the images
/// (`PrefixLayout`), but the rule refuses the same prefixes either way: a
/// text token moves the position one and an image block its longer side, so
/// the target's block starts at the same position, and every position
/// before it is below that. The target's shape moves only its own rows'
/// height and width positions, within the table for any side
/// `image.sizes` allows.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn target_rope_layout(prefix: u32, images: &[ImageSize]) -> (u32, Vec<u32>, Vec<(u32, u32)>) {
    let mut shapes: Vec<(u32, u32)> = images.iter().map(|size| (size.height / LATENT_SCALE, size.width / LATENT_SCALE)).collect();
    let cells: u32 = shapes.iter().map(|&(height, width)| height * width).sum();
    let mut starts = Vec::with_capacity(shapes.len() + 1);
    let mut start = prefix - cells;
    for &(height, width) in &shapes {
        starts.push(start);
        start += height * width;
    }
    starts.push(prefix);
    shapes.push((1, 1));
    (prefix + 1, starts, shapes)
}

/// Cells a transpose handles at a time: each channel's run of writes (or reads)
/// is this many contiguous floats, and the strided side stays inside the L1.
/// A power-of-two grid (2048x2048 is 128 x 128 = 16,384 cells, a 64 KiB stride
/// between channels) otherwise puts all 64 channels' writes on the same cache
/// sets: `channels` took 14-22 ms a step there, and 0.4 ms at 2752x2752.
const TRANSPOSE_TILE: usize = 128;

/// `latents` as the leaf's image slot holds them: `packed[y · w + x][c] =
/// latents[c][y][x]`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn packed(latents: &TargetLatents) -> Vec<f32> {
    let cells = latents.size.latent_cells();
    let mut out = vec![0.0; LATENT_CHANNELS * cells];
    for start in (0..cells).step_by(TRANSPOSE_TILE) {
        let end = (start + TRANSPOSE_TILE).min(cells);
        for (c, channel) in latents.values.chunks_exact(cells.max(1)).enumerate() {
            for (cell, &value) in (start..end).zip(&channel[start..end]) {
                out[cell * LATENT_CHANNELS + c] = value;
            }
        }
    }
    out
}

/// The leaf's `packed` latents of a `size` target, channel by channel.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn channels(size: ImageSize, packed: &[f32]) -> TargetLatents {
    let cells = size.latent_cells();
    let mut latents = TargetLatents::zeros(size);
    for start in (0..cells).step_by(TRANSPOSE_TILE) {
        let end = (start + TRANSPOSE_TILE).min(cells);
        for c in 0..LATENT_CHANNELS {
            let run = &mut latents.values[c * cells + start..c * cells + end];
            for (value, cell) in run.iter_mut().zip(start..end) {
                *value = packed[cell * LATENT_CHANNELS + c];
            }
        }
    }
    latents
}

/// The step of `sigmas` (the leaf's sampler's, its terminal 0 included) that
/// goes from `sigma` to `sigma_next`, compared bit for bit: the job reads
/// them off the plan, which copies them from the sampler.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn step_of(sigmas: &[f32], sigma: f32, sigma_next: f32) -> Option<usize> {
    sigmas.windows(2).position(|pair| pair[0] == sigma && pair[1] == sigma_next)
}

/// Whether `latents` are a target the leaf can make: sides in whole latent
/// cells, a value per channel per cell.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn check_target(latents: &TargetLatents) -> Result<(), ImageError> {
    let ImageSize { width, height } = latents.size;
    if width % LATENT_SCALE != 0 || height % LATENT_SCALE != 0 || latents.values.len() != LATENT_CHANNELS * latents.size.latent_cells() {
        return Err(ImageError::Leaf(format!(
            "{} latents for a {} target: its sides must be multiples of {LATENT_SCALE}, with {LATENT_CHANNELS} values a latent cell",
            latents.values.len(),
            latents.size
        )));
    }
    Ok(())
}

/// The leaf's error as the runner's: its refusal is the request's (a RoPE
/// position past the table, a size past the arena: spec image-model/04),
/// anything else the server's.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn image_error(error: ImageLeafError) -> ImageError {
    match error {
        ImageLeafError::Refused(why) => ImageError::Refused(why),
        ImageLeafError::Failed(why) => ImageError::Leaf(why),
        ImageLeafError::Plan(plan) => ImageError::Leaf(plan.to_string()),
    }
}

#[cfg(feature = "cuda")]
mod device {
    use std::collections::BTreeMap;
    use std::path::Path;

    use ignis_artifact::{CudaDevice, Reader};
    use ignis_core::image::{ConditionId, ConditionUnfit, ImageCondition, ImageError, ImagePlan, ImageRunner, ImageSize, RgbaImage, TargetLatents, LATENT_SCALE};
    use ignis_core::image_dit::rope_positions;
    use ignis_core::image_leaf::{Condition, ConditionImage as LeafImage, ConditionText, ImageLeaf, ImageLeafOptions};

    use super::{channels, check_target, image_error, max_target_cells, packed, step_of, target_rope_layout};
    use crate::image_encoder_input::{ConditionImage, EncoderFrontend};

    /// The image leaf behind [`ImageRunner`] (the module's doc says how).
    pub struct LeafRunner {
        leaf: ImageLeaf,
        frontend: EncoderFrontend,
        plan: ImagePlan,
        /// The prefixes held, each until `release_condition` or the drop.
        conditions: BTreeMap<ConditionId, Condition>,
        next: u64,
        /// The image the leaf's slot holds: the prefix it started over, its size.
        started: Option<(ConditionId, ImageSize)>,
    }

    // The leaf, its device and each prefix wrap raw device pointers with no
    // synchronization of their own. Sound because every call comes through
    // `ImageModel::with_runner`'s one `Mutex`, never two at once (as
    // `ignis_runtime::CudaModel` under the engine's). The calls run on
    // whichever blocking thread holds the lock: the leaf's `ignis_device`
    // ops select the device there, but its kernel launches (and the FP8
    // GEMM's `require_prepared`) run on the thread's current device, device
    // 0 on a thread that never chose one. Sound for device 0 only, which is
    // the one `load` takes; another device would have to be selected at
    // every call first.
    unsafe impl Send for LeafRunner {}

    impl LeafRunner {
        /// The image leaf for `artifact` under `config` (spec image-model/04
        /// §Load and plan): the artifact's front end, then the leaf on device
        /// 0 — the server's one card, as the text models' — in `config.vram`'s
        /// budget, its arena sized for the largest of `image.sizes`. Refused,
        /// naming why, when the artifact's front end does not read or the
        /// plan does not fit.
        pub fn load(artifact: &Path, config: &crate::config::Config) -> Result<Self, String> {
            let reader = Reader::open(artifact).map_err(|e| format!("open the image artifact: {e}"))?;
            let frontend = EncoderFrontend::from_reader(&reader)?;
            let image = &config.settings().image;
            let max_target = max_target_cells(&image.sizes, image.max_target_cells);
            let options = ImageLeafOptions {
                vram: config.vram,
                // `image.condition_budget_tokens` lowers the budget per
                // request (`images::budget`), so a PATCH of it holds live.
                condition_budget_cap: None,
                max_target_tokens: u32::try_from(max_target).unwrap_or(u32::MAX),
                min_condition_budget_tokens: image.min_condition_budget_tokens,
                prefix_chunk: None,
            };
            let dev = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
            let start = std::time::Instant::now();
            let leaf = ImageLeaf::load(&reader, dev, &options).map_err(|e| e.to_string())?;
            super::log_plan(leaf.plan(), max_target, start.elapsed().as_secs_f64());
            let sampler = leaf.sampler();
            let plan = ImagePlan {
                condition_budget_tokens: leaf.plan().condition_budget_tokens,
                sample_sigmas: sampler.sigmas[..sampler.steps()].to_vec(),
                max_target_cells: max_target,
            };
            Ok(Self { leaf, frontend, plan, conditions: BTreeMap::new(), next: 0, started: None })
        }

        /// The leaf's own plan, in bytes: what the load reserved, for
        /// `/metrics` (GitHub #324).
        pub fn leaf_plan(&self) -> &ignis_core::image_leaf::ImagePlan {
            self.leaf.plan()
        }
    }

    impl ImageRunner for LeafRunner {
        fn plan(&self) -> &ImagePlan {
            &self.plan
        }

        /// The front end's count, refused for a size the processor would
        /// not keep or a prompt the template refuses. Within the budget it
        /// meets the limits beside it — the arena's room for the condition
        /// (`ImageLeaf::condition_fits`), a RoPE position for the target
        /// after it ([`target_rope_layout`]) — and past either has no room
        /// while there are images to scale down, else is refused.
        fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ConditionUnfit> {
            let count = self.frontend.count(prompt, images).map_err(ConditionUnfit::Refused)?;
            if count.prefix > self.plan.condition_budget_tokens {
                return Ok(count.prefix);
            }
            // Within the budget, which is below 2^20 tokens: each count is a u32.
            let sizes: Vec<(u32, u32)> = images.iter().map(|size| (size.width, size.height)).collect();
            let (tokens, starts, shapes) = target_rope_layout(count.prefix as u32, images);
            let room = self
                .leaf
                .condition_fits(count.encoder_rows as u32, count.pads as u32, &sizes, count.prefix as u32)
                .and_then(|()| rope_positions(tokens, &starts, &shapes).map(|_| ()));
            match room {
                Ok(()) => Ok(count.prefix),
                Err(why) if images.is_empty() => Err(ConditionUnfit::Refused(why)),
                Err(why) => Err(ConditionUnfit::NoRoom(why)),
            }
        }

        /// The count first: a size the processor would not keep, or a prefix
        /// past the budget, is refused before any image is prepared.
        fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError> {
            let sizes: Vec<ImageSize> = condition.images.iter().map(|image| image.size).collect();
            let count = self.frontend.count(&condition.prompt, &sizes).map_err(ImageError::Refused)?;
            let budget = self.plan.condition_budget_tokens;
            if count.prefix > budget {
                return Err(ImageError::OverBudget { tokens: count.prefix, budget });
            }
            let images: Vec<ConditionImage> = condition
                .images
                .iter()
                .map(|image| ConditionImage { width: image.size.width, height: image.size.height, rgba: &image.pixels })
                .collect();
            let input = self.frontend.input(&condition.prompt, &images).map_err(ImageError::Refused)?;
            // The processor kept every size (the count): each grid is its image in 16-pixel patches.
            let leaf_images: Vec<LeafImage> = input
                .images
                .iter()
                .zip(&condition.images)
                .map(|(patches, image)| LeafImage {
                    grid: patches.grid,
                    pixels: &patches.pixels,
                    width: image.size.width,
                    height: image.size.height,
                    rgba: &image.pixels,
                })
                .collect();
            let text = ConditionText {
                input_ids: &input.input_ids,
                positions: &input.positions,
                drop_idx: input.drop_idx,
                image_pad_rows: &input.image_pad_rows,
            };
            // Held from here: nothing fallible between the encode and the map.
            let prefix = self.leaf.encode_condition(&text, &leaf_images).map_err(image_error)?;
            let id = ConditionId(self.next);
            self.next += 1;
            self.conditions.insert(id, prefix);
            Ok(id)
        }

        fn denoise_step(
            &mut self,
            condition: ConditionId,
            latents: &mut TargetLatents,
            sigma: f32,
            sigma_next: f32,
            velocity: Option<&mut Vec<f32>>,
        ) -> Result<(), ImageError> {
            let prefix = self.conditions.get(&condition).ok_or(ImageError::NoSuchCondition(condition))?;
            let step = step_of(&self.leaf.sampler().sigmas, sigma, sigma_next)
                .ok_or_else(|| ImageError::Leaf(format!("no step of the leaf's schedule goes from sigma {sigma} to {sigma_next}")))?;
            check_target(latents)?;
            let size = latents.size;
            let up = packed(latents);
            if self.started == Some((condition, size)) {
                self.leaf.set_latents(&up).map_err(image_error)?;
            } else {
                self.started = None;
                let (height, width) = (size.height / LATENT_SCALE, size.width / LATENT_SCALE);
                self.leaf.start_image(prefix, height, width, &up).map_err(image_error)?;
                self.started = Some((condition, size));
            }
            self.leaf.denoise_step(prefix, step).map_err(image_error)?;
            *latents = channels(size, &self.leaf.latents().map_err(image_error)?);
            if let Some(out) = velocity {
                *out = channels(size, &self.leaf.velocity().map_err(image_error)?).values;
            }
            Ok(())
        }

        /// The leaf decodes its image slot: `latents` go there first, the
        /// target a denoising step started (one the job is making). Any other
        /// size is a server bug, not the request's.
        fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
            check_target(latents)?;
            if self.started.map(|(_, size)| size) != Some(latents.size) {
                return Err(ImageError::Leaf(format!("vae_decode of a {} target no denoising step started", latents.size)));
            }
            self.leaf.set_latents(&packed(latents)).map_err(image_error)?;
            let pixels = self.leaf.vae_decode().map_err(image_error)?;
            Ok(RgbaImage { size: latents.size, pixels })
        }

        fn release_condition(&mut self, condition: ConditionId) {
            // A failed leaf release is logged (`ignis.image.release_failed`), but the
            // entry is already gone, so the gauge counts those bytes as freed.
            if let Some(prefix) = self.conditions.remove(&condition) {
                // The leaf's image slot goes with a released prefix.
                self.started = None;
                if let Err(error) = self.leaf.release_condition(prefix) {
                    tracing::error!(name: "ignis.image.release_failed", %error, "a condition prefix was not freed");
                }
            }
        }

        /// Each prefix's own device bytes, as the leaf allocated them.
        fn condition_bytes(&self) -> u64 {
            self.conditions.values().map(Condition::bytes).sum()
        }
    }

    impl Drop for LeafRunner {
        /// Every prefix still held, freed before the leaf drops.
        fn drop(&mut self) {
            for (_, prefix) in std::mem::take(&mut self.conditions) {
                if let Err(error) = self.leaf.release_condition(prefix) {
                    tracing::error!(name: "ignis.image.release_failed", %error, "a condition prefix was not freed as the image model unloaded");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::image_leaf::MAX_TARGET_TOKENS;

    /// A 48 x 32 target (3 x 2 latent cells), each value its channel, row and
    /// column.
    fn target() -> TargetLatents {
        let size = ImageSize::new(48, 32);
        let mut latents = TargetLatents::zeros(size);
        for c in 0..LATENT_CHANNELS {
            for cell in 0..6 {
                latents.values[c * 6 + cell] = (1000 * c + cell) as f32;
            }
        }
        latents
    }

    /// The leaf's slot is `[y · w + x][c]`, the job's latents `[c][y][x]`:
    /// cell (y 1, x 2) of channel 5 is packed row 5, column 5; back again,
    /// nothing moved.
    #[test]
    fn latents_go_to_the_leaf_packed_and_come_back_by_channel() {
        let latents = target();
        let up = packed(&latents);
        assert_eq!(up.len(), 6 * LATENT_CHANNELS);
        // cell y 1, x 2 of a row of 3 is cell 5
        assert_eq!(up[5 * LATENT_CHANNELS + 5], 5005.0);
        assert_eq!(up[..3], [0.0, 1000.0, 2000.0], "cell 0's first three channels");
        assert_eq!(channels(latents.size, &up), latents);
    }

    /// The tiled transposes are the plain ones, value for value: over cell
    /// counts that end a tile exactly, mid-tile, and below one, and over the
    /// 2048x2048 grid (16,384 cells) whose power-of-two stride made the plain
    /// `channels` 50x slower.
    #[test]
    fn the_tiled_transposes_move_every_value_where_the_plain_ones_did() {
        for (width, height) in [(16, 16), (48, 32), (128 * 16, 16), (130 * 16, 16), (300 * 16, 16), (2048, 2048)] {
            let size = ImageSize::new(width, height);
            let cells = size.latent_cells();
            let mut latents = TargetLatents::zeros(size);
            for (i, v) in latents.values.iter_mut().enumerate() {
                *v = (i as f32).sin();
            }
            let mut plain = vec![0.0f32; LATENT_CHANNELS * cells];
            for c in 0..LATENT_CHANNELS {
                for cell in 0..cells {
                    plain[cell * LATENT_CHANNELS + c] = latents.values[c * cells + cell];
                }
            }
            let up = packed(&latents);
            assert_eq!(up, plain, "packed, {width}x{height}");
            assert_eq!(channels(size, &up), latents, "channels, {width}x{height}");
        }
    }

    /// A step is the sampler's pair (σ, σ_next) bit for bit, the last one
    /// to the terminal 0; a pair the schedule does not have is no step.
    #[test]
    fn a_step_is_the_sampler_step_between_the_two_sigmas() {
        let sigmas = [1.0, 0.75, 0.5, 0.0];
        assert_eq!(step_of(&sigmas, 1.0, 0.75), Some(0));
        assert_eq!(step_of(&sigmas, 0.75, 0.5), Some(1));
        assert_eq!(step_of(&sigmas, 0.5, 0.0), Some(2));
        assert_eq!(step_of(&sigmas, 1.0, 0.5), None, "a skipped sigma");
        assert_eq!(step_of(&sigmas, 0.75, 0.5 + f32::EPSILON), None, "bit for bit");
    }

    /// A refusal is the request's, a failure and a plan the server's.
    #[test]
    fn the_leafs_refusal_is_the_requests_and_its_failure_the_servers() {
        assert_eq!(image_error(ImageLeafError::Refused("a RoPE position past the table".into())), ImageError::Refused("a RoPE position past the table".into()));
        assert_eq!(image_error(ImageLeafError::Failed("the DiT's blocks: launch".into())), ImageError::Leaf("the DiT's blocks: launch".into()));
        let plan = ignis_core::image_leaf::ImagePlanError::BudgetAboveFree { budget_bytes: 2, free_bytes: 1 };
        assert_eq!(image_error(ImageLeafError::Plan(plan.clone())), ImageError::Leaf(plan.to_string()));
    }

    /// The arena is sized for the largest of `image.sizes` within
    /// `image.max_target_cells`: by default 2400x1792's 16,800 cells, the
    /// leaf's own default; fewer sizes, less. A size past the cap does not
    /// grow it (review-321d F2), and with none within it the arena is the cap.
    #[test]
    fn the_largest_target_is_the_largest_size_served_within_the_cap() {
        let defaults = crate::config::DEFAULT_IMAGE_SIZES;
        assert_eq!(max_target_cells(&defaults, MAX_TARGET_TOKENS), 16_800);
        assert_eq!(max_target_cells(&defaults, MAX_TARGET_TOKENS), MAX_TARGET_TOKENS as usize);
        assert_eq!(max_target_cells(&[ImageSize::new(1024, 1024), ImageSize::new(512, 768)], MAX_TARGET_TOKENS), 4096);
        let huge = [ImageSize::new(4096, 4096), ImageSize::new(1024, 1024)];
        assert_eq!(max_target_cells(&huge, MAX_TARGET_TOKENS), 4096, "65,536 cells, past the cap");
        assert_eq!(max_target_cells(&defaults, 16_384), 16_384, "2048x2048, the largest within 16,384");
        assert_eq!(max_target_cells(&[ImageSize::new(4096, 4096)], 8_000), 8_000, "none within: the cap");
    }

    /// The RoPE layout of a prefix: its text first, each image's block in
    /// latent cells (height, width), the one-cell target at its end. The
    /// edit oracle's prefix: 26 text tokens and one 1024² image.
    #[test]
    fn the_targets_rope_layout_is_the_text_then_the_images_then_one_cell() {
        assert_eq!(target_rope_layout(31, &[]), (32, vec![31], vec![(1, 1)]));
        let (tokens, starts, shapes) = target_rope_layout(4122, &[ImageSize::new(1024, 1024)]);
        assert_eq!((tokens, starts, shapes), (4123, vec![26, 4122], vec![(64, 64), (1, 1)]));
        let (_, starts, shapes) = target_rope_layout(9 + 2048 + 64, &[ImageSize::new(512, 1024), ImageSize::new(128, 128)]);
        assert_eq!((starts, shapes), (vec![9, 9 + 2048, 9 + 2048 + 64], vec![(64, 32), (8, 8), (1, 1)]));
    }

    /// Review-321d L3: the load's plan is one `ignis.image.plan` event, its
    /// bytes and tokens as fields, as the text models' plans are; its
    /// message is the plan's line.
    #[test]
    fn the_plan_is_one_named_event_with_its_bytes() {
        const GIB: u64 = 1 << 30;
        let plan = ignis_core::image_leaf::ImagePlan {
            mode: ignis_core::VramMode::Derived { headroom_bytes: 4 * GIB },
            free_at_start_bytes: 31 * GIB,
            budget_bytes: 27 * GIB,
            weights_bytes: 13 * GIB,
            arena_bytes: 3 * GIB,
            condition_budget_tokens: 45_184,
            condition_budget_bytes: 11 * GIB,
        };
        use tracing_subscriber::layer::SubscriberExt;
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, || log_plan(&plan, 16_800, 9.5));
        let records: Vec<serde_json::Value> = sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect();
        assert_eq!(records.len(), 1, "{records:?}");
        let event = &records[0];
        assert_eq!(event["event_name"], "ignis.image.plan", "{event}");
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned().unwrap_or_else(|| panic!("no `{name}` in {event}"));
        assert_eq!(field("budget_bytes"), 27 * GIB);
        assert_eq!(field("weights_bytes"), 13 * GIB);
        assert_eq!(field("arena_bytes"), 3 * GIB);
        assert_eq!(field("condition_budget_tokens"), 45_184);
        assert_eq!(field("condition_budget_bytes"), 11 * GIB);
        assert_eq!(field("free_at_start_bytes"), 31 * GIB);
        assert_eq!(field("max_target_cells"), 16_800);
        assert_eq!(field("mode"), plan.mode.as_str());
        assert!(event.to_string().contains(&plan.line()), "the plan's line: {event}");
    }

    /// A target is whole latent cells, a value a channel a cell.
    #[test]
    fn a_target_is_whole_latent_cells() {
        assert_eq!(check_target(&target()), Ok(()));
        assert!(matches!(check_target(&TargetLatents { size: ImageSize::new(40, 32), values: vec![0.0; 64 * 4] }), Err(ImageError::Leaf(_))));
        assert!(matches!(check_target(&TargetLatents { size: ImageSize::new(48, 32), values: vec![0.0; 5] }), Err(ImageError::Leaf(_))));
    }
}
