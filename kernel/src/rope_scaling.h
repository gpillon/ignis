#pragma once
// GitHub #227: the text RoPE frequency table a load runs on -- OURS, not
// vendored (ADR 0009: the Program layer owns what the vendored ops are
// called with). `ninfer/ops/rope.h` already takes an arbitrary per-pair
// table and a q-side `attention_factor`, and says so outright: "YaRN-shaped
// tables are constructed by the owning target". This is that construction.
//
// Pure host math over the checkpoint's rope constants -- no device call, no
// tensor, nothing to bind. The reference builds the same table in
// `src/targets/qwen3_6/impl/runtime/rope_scaling.h`; that file sits in the
// reference's target layer, which is exactly the layer ADR 0009 says is
// ours, so this is written rather than vendored and carries no port claim.

#include "ninfer/ops/rope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace ignis {

// The Qwen 3.8-27B text rotary constants (the same geometry
// `kernel/src/gqa_layer.cu` rotates at, named once so the table and the op
// cannot disagree).
inline constexpr float kTextRopeTheta = 1.0e7F;
inline constexpr std::int32_t kTextRotaryDim = 64;
// The checkpoint's trained position envelope: the range the linear table is
// correct over, and the denominator every YaRN ramp bound is measured in.
inline constexpr std::uint32_t kTextOriginalPositions = 262144;
// The widest scaling a load accepts. 64x the trained envelope is already
// past every attention envelope the engine can serve
// (`kGqaAttentionMaximumVisibleKeys`), so this is a typo guard, not a
// capability claim.
inline constexpr float kMaxYarnFactor = 64.0F;

// The scaling a load asks for: the operator's `--rope-scaling`, carried
// across the ABI as four scalars. A `factor` of 0 or 1 is no scaling and
// selects the linear table -- which matters numerically, not just
// stylistically: `ops/rope.h` gives `attention_factor == 1` the exact legacy
// FP32 angle route and anything else an FP64-reduced one, so a "YaRN at
// factor 1" would quietly move the default engine's outputs.
struct RopeScaling {
  float factor = 0.0F;
  float temperature = 0.1F;
  float beta_fast = 32.0F;
  float beta_slow = 1.0F;

  bool is_yarn() const { return factor > 1.0F; }
};

// Why the scaling is not usable, or an empty string when it is. The caller
// turns it into the load's own error message.
inline std::string rope_scaling_rejection(const RopeScaling &scaling) {
  if (!std::isfinite(scaling.factor) || scaling.factor < 0.0F ||
      (scaling.factor > 0.0F && scaling.factor < 1.0F) || scaling.factor > kMaxYarnFactor) {
    return "rope_scaling_factor " + std::to_string(scaling.factor) +
           " must be 0 or 1 (linear) or a YaRN factor in (1, " +
           std::to_string(static_cast<int>(kMaxYarnFactor)) + "]";
  }
  if (!scaling.is_yarn()) {
    // The other three are the YaRN ramp's; without a factor they are unread,
    // and a caller that zero-fills the struct must not be refused for them.
    return {};
  }
  if (!std::isfinite(scaling.temperature) || scaling.temperature <= 0.0F) {
    return "rope_scaling_temperature " + std::to_string(scaling.temperature) +
           " must be positive and finite";
  }
  if (!std::isfinite(scaling.beta_fast) || !std::isfinite(scaling.beta_slow) ||
      scaling.beta_slow <= 0.0F || scaling.beta_fast <= scaling.beta_slow) {
    return "the YaRN ramp requires beta_fast > beta_slow > 0, got beta_fast " +
           std::to_string(scaling.beta_fast) + " and beta_slow " +
           std::to_string(scaling.beta_slow);
  }
  return {};
}

/**
 * The YaRN table (HF `_compute_yarn_parameters` semantics). For pair i in
 * [0, rotary_dim/2) with `linear = theta^(-2i/rotary_dim)`:
 *
 *   i < low    extrapolate   linear
 *   i > high   interpolate   linear / factor
 *   otherwise  blend         linear * ((1 - e) + e / factor),
 *                            e = (i - low) / (high - low)
 *
 * The ramp bounds are the floor/ceil of the correction dimension
 * `rotary_dim * ln(original / (beta * 2pi)) / (2 ln theta)`, clamped to leave
 * a non-empty extrapolation and interpolation segment. At the checkpoint's
 * constants (theta 1e7, rotary_dim 64, original 262144, beta_fast 32,
 * beta_slow 1) that is low = 14, high = 22.
 *
 * `attention_factor = temperature * ln(factor) + 1` is the q-side
 * temperature: it travels squared on the q rows and leaves cached K
 * factor-free, so a sequence's keys stay valid whatever a later round does.
 *
 * The caller has already passed `rope_scaling_rejection`; a factor at or
 * below 1 here would divide the table by something meaningless, so it is the
 * caller's job to have taken the linear branch instead.
 */
inline ninfer::ops::RopeFrequencies rope_yarn_frequencies(float theta, int rotary_dim,
                                                          std::uint32_t original_positions,
                                                          const RopeScaling &scaling) {
  constexpr double kTwoPi = 6.28318530717958648;
  const double base = static_cast<double>(theta);
  const double original = static_cast<double>(original_positions);
  const double factor = static_cast<double>(scaling.factor);
  const auto correction = [&](double beta) {
    return static_cast<double>(rotary_dim) * std::log(original / (beta * kTwoPi)) /
           (2.0 * std::log(base));
  };
  const int half = rotary_dim / 2;
  const int low = std::clamp(static_cast<int>(std::floor(correction(scaling.beta_fast))), 0,
                             half - 2);
  const int high = std::clamp(static_cast<int>(std::ceil(correction(scaling.beta_slow))), low + 1,
                              half - 1);

  ninfer::ops::RopeFrequencies frequencies;
  frequencies.attention_factor =
      static_cast<float>(static_cast<double>(scaling.temperature) * std::log(factor) + 1.0);
  for (int i = 0; i < half; ++i) {
    const double linear = std::pow(base, -2.0 * i / rotary_dim);
    if (i < low) {
      frequencies.inv_frequency[i] = linear;
    } else if (i > high) {
      frequencies.inv_frequency[i] = linear / factor;
    } else {
      const double blend = static_cast<double>(i - low) / static_cast<double>(high - low);
      frequencies.inv_frequency[i] = linear * ((1.0 - blend) + blend / factor);
    }
  }
  return frequencies;
}

// The table the text layers rotate at under this scaling: the legacy linear
// one when the load asked for no scaling (bit-for-bit the engine before
// #227), the YaRN one otherwise.
inline ninfer::ops::RopeFrequencies text_rope_frequencies(const RopeScaling &scaling) {
  return scaling.is_yarn()
             ? rope_yarn_frequencies(kTextRopeTheta, kTextRotaryDim, kTextOriginalPositions,
                                     scaling)
             : ninfer::ops::rope_linear_frequencies(kTextRopeTheta, kTextRotaryDim);
}

} // namespace ignis
