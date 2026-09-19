// GitHub #227: the text RoPE frequency table -- OURS, not vendored.
//
// Two host-only halves, no device work anywhere:
//
// 1. The table itself (kernel/src/rope_scaling.h) against reference tables
//    computed independently from the documented HF `_compute_yarn_parameters`
//    formula at the checkpoint's constants (theta 1e7, rotary_dim 64,
//    original_positions 262144, beta_fast 32, beta_slow 1), plus the two
//    things a wrong table would not show in the numbers: that no scaling
//    still builds the *legacy linear* table with an attention factor of 1
//    (`ops/rope.h` routes that case through its exact FP32 angle path, so a
//    "YaRN at factor 1" would move every existing output), and that the
//    temperature knob touches nothing but the attention factor.
// 2. The ABI gate: `ignis_model_load` refuses an unusable scaling by name,
//    before it binds anything, and lets a usable one through to the binding.
//
// The reference tables carry full round-trip precision. `std::pow` is
// allowed one ulp of toolchain spread (the reference's own rope-table test
// makes the same allowance), which is still far tighter than any difference
// a wrong ramp bound or a wrong segment would produce.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here
// needs a GPU in the first place, like test_paged_kv_page_budget.cpp.

#include "rope_scaling.h"

#include "ignis_model.h"
#include "ignis_seq.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

bool within_one_ulp(double got, double want) {
  return got == want || std::nextafter(want, 0.0) == got || std::nextafter(want, 1e30) == got;
}

// YaRN factor 4: ramp low=14, high=22 -- pairs 0..13 extrapolate, 14..22
// blend, 23..31 interpolate at inv/4.
constexpr double kYarn4[32] = {
    1.00000000000000000e+00, 6.04296390238132863e-01, 3.65174127254837722e-01,
    2.20673406908458991e-01, 1.33352143216332403e-01, 8.05842187761481865e-02,
    4.86967525165863113e-02, 2.94272717620928173e-02, 1.77827941003892293e-02,
    1.07460782832131743e-02, 6.49381631576211298e-03, 3.92418975848453627e-03,
    2.37137370566165538e-03, 1.43301257023696268e-03, 8.65964323360065387e-04,
    4.74239822680104593e-04, 2.56935059888680839e-04, 1.37349745076000397e-04,
    7.21738740430911334e-05, 3.70722498206803983e-05, 1.84492220250004719e-05,
    8.75977007117900255e-06, 3.84981631514872963e-06, 2.32643010232424761e-06,
    1.40585331297587280e-06, 8.49552082235639817e-07, 5.13381256614286516e-07,
    3.10234440187929882e-07, 1.87473552333113962e-07, 1.13289590940020448e-07,
    6.84604908566090348e-08, 4.13704274985795338e-08,
};

// YaRN factor 2, same ramp.
constexpr double kYarn2[32] = {
    1.00000000000000000e+00, 6.04296390238132863e-01, 3.65174127254837722e-01,
    2.20673406908458991e-01, 1.33352143216332403e-01, 8.05842187761481865e-02,
    4.86967525165863113e-02, 2.94272717620928173e-02, 1.77827941003892293e-02,
    1.07460782832131743e-02, 6.49381631576211298e-03, 3.92418975848453627e-03,
    2.37137370566165538e-03, 1.43301257023696268e-03, 8.65964323360065387e-04,
    4.90592920013901306e-04, 2.76699295264733170e-04, 1.55264929216348299e-04,
    8.66086488517093628e-05, 4.79758527091158151e-05, 2.63560314642863898e-05,
    1.43341692073838242e-05, 7.69963263029745927e-06, 4.65286020464849521e-06,
    2.81170662595174560e-06, 1.69910416447127963e-06, 1.02676251322857303e-06,
    6.20468880375859763e-07, 3.74947104666227924e-07, 2.26579181880040896e-07,
    1.36920981713218070e-07, 8.27408549971590677e-08,
};

// The legacy linear table, float-rounded theta^(-2i/64): what every GQA
// layer rotated at before the table became a load option.
constexpr float kLinearText[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

ignis::RopeScaling yarn(float factor) {
  ignis::RopeScaling scaling;
  scaling.factor = factor;
  return scaling;
}

void check_table(const char *label, const double (&expected)[32], float factor) {
  const ninfer::ops::RopeFrequencies table = ignis::text_rope_frequencies(yarn(factor));
  for (int i = 0; i < 32; ++i) {
    check(within_one_ulp(table.inv_frequency[i], expected[i]),
          std::string(label) + ": pair " + std::to_string(i) + " is " +
              std::to_string(table.inv_frequency[i]) + ", reference " +
              std::to_string(expected[i]));
  }
  const double want_factor = 0.1 * std::log(static_cast<double>(factor)) + 1.0;
  check(std::abs(static_cast<double>(table.attention_factor) - want_factor) < 1e-7 * want_factor,
        std::string(label) + ": attention factor is " + std::to_string(table.attention_factor) +
            ", want " + std::to_string(want_factor));
}

bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

ignis_topology no_layer_topology() {
  ignis_topology topology{};
  topology.num_layers = 0;
  topology.layer_kinds = nullptr;
  topology.hidden = 5120;
  topology.vocab = 248320;
  topology.ffn_intermediate = 17408;
  topology.gdn_num_layers = 1;
  return topology;
}

// Loads nothing under `options` and returns the leaf's last-error message.
// Every arm here must fail: a usable scaling reaches the binding and fails
// there on the text tensors this call hands over none of, which is exactly
// how the gate's two sides are told apart.
std::string load_error(const ignis_model_load_options &options) {
  const ignis_topology topology = no_layer_topology();
  ignis_bound_tensor none{};
  ignis_model *model = nullptr;
  const int32_t rc =
      ignis_model_load(&none, 0, &topology, /*prefill_chunk_tokens=*/128,
                       /*max_context_tokens=*/128, IGNIS_KV_FORMAT_BF16, &options, &model);
  if (rc == 0 || model != nullptr) {
    std::fprintf(stderr, "FATAL: this load must fail before any device work\n");
    if (model != nullptr) {
      ignis_model_free(model);
    }
    std::exit(EXIT_FAILURE);
  }
  return ignis_model_last_error();
}

ignis_model_load_options scaled(float factor, float temperature, float beta_fast,
                                float beta_slow) {
  ignis_model_load_options options{};
  options.size = sizeof(options);
  options.speculative_backend = IGNIS_SPECULATIVE_NONE;
  options.rope_scaling_factor = factor;
  options.rope_scaling_temperature = temperature;
  options.rope_scaling_beta_fast = beta_fast;
  options.rope_scaling_beta_slow = beta_slow;
  return options;
}

} // namespace

int main() {
  // --- 1. the YaRN tables -----------------------------------------------------
  check_table("yarn:4", kYarn4, 4.0F);
  check_table("yarn:2", kYarn2, 2.0F);

  // The ramp bounds, read off the table rather than recomputed: below `low`
  // a pair is the untouched linear frequency, at and past `high` it is the
  // linear one divided by the factor, and in between it is neither.
  {
    const ninfer::ops::RopeFrequencies linear = ignis::text_rope_frequencies({});
    const ninfer::ops::RopeFrequencies table = ignis::text_rope_frequencies(yarn(4.0F));
    check(table.inv_frequency[13] == linear.inv_frequency[13],
          "pair 13 is below the ramp and extrapolates");
    check(table.inv_frequency[14] == linear.inv_frequency[14],
          "pair 14 opens the ramp at blend 0, still the linear frequency");
    check(table.inv_frequency[15] != linear.inv_frequency[15], "pair 15 is inside the ramp");
    check(within_one_ulp(table.inv_frequency[22], linear.inv_frequency[22] / 4.0),
          "pair 22 closes the ramp at blend 1, fully interpolated");
    check(within_one_ulp(table.inv_frequency[31], linear.inv_frequency[31] / 4.0),
          "pair 31 is past the ramp and interpolates");
  }

  // --- 2. no scaling is the legacy table --------------------------------------
  {
    const ninfer::ops::RopeFrequencies table = ignis::text_rope_frequencies({});
    for (int i = 0; i < 32; ++i) {
      check(static_cast<float>(table.inv_frequency[i]) == kLinearText[i],
            "linear table: pair " + std::to_string(i) + " is not the legacy frequency");
    }
    check(table.attention_factor == 1.0F,
          "linear table: the attention factor must stay exactly 1 (ops/rope.h's legacy "
          "FP32 angle route hangs on it)");
    // A factor of exactly 1 is "no scaling", not "YaRN at 1": same table,
    // same untouched attention factor.
    const ninfer::ops::RopeFrequencies one = ignis::text_rope_frequencies(yarn(1.0F));
    check(one.attention_factor == 1.0F, "factor 1 is the linear table, not a YaRN table");
    for (int i = 0; i < 32; ++i) {
      check(one.inv_frequency[i] == table.inv_frequency[i],
            "factor 1 changed pair " + std::to_string(i));
    }
  }

  // --- 3. the temperature moves only the attention factor ---------------------
  {
    ignis::RopeScaling warm = yarn(2.0F);
    warm.temperature = 0.25F;
    const ninfer::ops::RopeFrequencies table = ignis::text_rope_frequencies(warm);
    const double want = 0.25 * std::log(2.0) + 1.0;
    check(std::abs(static_cast<double>(table.attention_factor) - want) < 1e-7 * want,
          "temperature 0.25 at factor 2 gives 0.25*ln(2)+1");
    const ninfer::ops::RopeFrequencies reference = ignis::text_rope_frequencies(yarn(2.0F));
    for (int i = 0; i < 32; ++i) {
      check(table.inv_frequency[i] == reference.inv_frequency[i],
            "temperature moved frequency pair " + std::to_string(i));
    }
  }

  // --- 4. what the builder refuses --------------------------------------------
  {
    check(ignis::rope_scaling_rejection({}).empty(), "the default (no scaling) is usable");
    check(ignis::rope_scaling_rejection(yarn(1.0F)).empty(), "factor 1 is usable");
    check(ignis::rope_scaling_rejection(yarn(ignis::kMaxYarnFactor)).empty(),
          "the widest factor is usable");
    check(!ignis::rope_scaling_rejection(yarn(0.5F)).empty(), "a factor below 1 is refused");
    check(!ignis::rope_scaling_rejection(yarn(ignis::kMaxYarnFactor + 1.0F)).empty(),
          "a factor past the limit is refused");
    check(!ignis::rope_scaling_rejection(yarn(std::nanf(""))).empty(), "NaN is refused");
    // A caller that zero-fills the struct asks for no scaling, and must not
    // be refused for the ramp fields it left at zero.
    ignis::RopeScaling zeroed;
    zeroed.temperature = 0.0F;
    zeroed.beta_fast = 0.0F;
    zeroed.beta_slow = 0.0F;
    check(ignis::rope_scaling_rejection(zeroed).empty(),
          "a zero-filled scaling is no scaling, not a broken ramp");
    zeroed.factor = 4.0F;
    check(!ignis::rope_scaling_rejection(zeroed).empty(),
          "the same zeroed ramp with a factor is refused");
    ignis::RopeScaling inverted = yarn(4.0F);
    inverted.beta_fast = 1.0F;
    inverted.beta_slow = 32.0F;
    check(!ignis::rope_scaling_rejection(inverted).empty(),
          "beta_fast below beta_slow is refused");
  }

  // --- 5. the ABI gate --------------------------------------------------------
  {
    const std::string bad = load_error(scaled(0.5F, 0.1F, 32.0F, 1.0F));
    check(contains(bad, "rope_scaling_factor"), "a bad factor is refused by name: " + bad);
    const std::string cold = load_error(scaled(4.0F, 0.0F, 32.0F, 1.0F));
    check(contains(cold, "rope_scaling_temperature"),
          "a non-positive temperature is refused by name: " + cold);
    const std::string ramp = load_error(scaled(4.0F, 0.1F, 1.0F, 32.0F));
    check(contains(ramp, "beta_fast"), "an inverted ramp is refused by name: " + ramp);
    // A usable scaling is not the load's business to refuse: it reaches the
    // binding, which fails on the text tensors this call hands over none of.
    for (const float factor : {0.0F, 1.0F, 4.0F}) {
      const std::string m = load_error(scaled(factor, 0.1F, 32.0F, 1.0F));
      check(contains(m, "text/token_embedding"),
            "factor " + std::to_string(factor) + " reaches the binding: " + m);
    }
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "rope scaling: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("rope scaling: all checks passed\n");
  return EXIT_SUCCESS;
}
