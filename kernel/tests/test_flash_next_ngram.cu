// Flash-Next's n-gram embedding, device side (fn_ngram_add, kernel/src/flash_next/ngram.cu) at the
// real geometry -- OURS (spec flash-next/04, slice S4 of GitHub #302).
//
// The oracle is transformers' Qwen4ExpTextPLELayer.forward (BF16, CPU), recorded by
// fixtures/flash_next_ngram/record.py on 16 tokens of gathered INT4 rows from an empty conv state:
// its output `ple_out`. The weights and the hidden input are regenerated here from the counter hash
// with record.py's constants (fp8_test_common.h's make_fp8 for the FP8 projections).
//
//   host model   the device's arithmetic restated on the host, projections in fp64: matches the
//                recorded reference within the FP8-vs-BF16-weight tolerance (the reference
//                multiplies BF16-rounded weights; the FP8 linear scales an exact sum).
//   one shot     16 tokens in one call: matches the host model.
//   chunked      the same tokens in calls of 5, 1, 1, 1, 2, 6 (prefill chunks, then decode steps,
//                one call shorter than the conv's 9 past columns): matches the one-shot call.
//   lanes        three sequences in three slots, decoded one token per call, all lanes in each
//                call: each matches its own one-shot call.
//   positions    one call whose lanes sit at positions 5, 0 and 12: each matches its own run.
//   graph        the decode rounds captured once in a CUDA graph and replayed with the inputs,
//                slots and positions refreshed in place: bit for bit the eager rounds.
//   windows      300 tokens in one call (two internal windows): matches 150 + 150.
//   fresh        a slot left dirty by an earlier sequence: a call at position 0 ignores it.
//
// Every call's scratch arena is exactly fn_ngram_add_scratch_bytes(): the declaration is held.
//
// `--host-only` runs the host-model check alone (no CUDA call): the tolerance calibration.
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "fp8_test_common.h"
#include "ignis_fp8_linear.h"
#include "flash_next/flash_next_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifndef IGNIS_FLASH_NEXT_NGRAM_FIXTURE_DIR
#error "IGNIS_FLASH_NEXT_NGRAM_FIXTURE_DIR must name fixtures/flash_next_ngram"
#endif

using namespace moe_test;
namespace fn = ignis::flash_next;

namespace {

// ---- record.py's geometry and generation constants ----------------------------------------
constexpr int kHidden = 2560, kStreams = 4, kWidth = kHidden * kStreams;
constexpr int kEmbed = 2560, kHeads = 16, kHeadDim = 160, kRowBytes = 90;
constexpr int kKernel = 4, kDilation = 3, kState = (kKernel - 1) * kDilation;
constexpr int kTokens = 16;
constexpr float kEps = 1e-6F;
constexpr uint32_t S_KEY = 4101, S_VALUE = 4201;
constexpr uint32_t S_NORM_KEY = 4301, S_NORM_QUERY = 4302, S_NORM_CONV = 4303, S_CONV = 4304;
constexpr uint32_t S_HIDDEN = 4401, S_ROW_CODES = 4501, S_ROW_SCALES = 4502;
constexpr float KEY_SCALE_MAG = 0.0004F, VALUE_SCALE_MAG = 0.0008F;
constexpr float NORM_AMP = 0.5F, CONV_AMP = 0.5F, HIDDEN_AMP = 0.05F, ROW_SCALE_MAG = 0.02F;

#define NG_CUDA MOE_CUDA
#define NG_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, fn::fn_last_error());       \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

float bf(float x) { return bf16_to_f32(f32_to_bf16(x)); }

std::vector<uint16_t> bf16_vector(uint32_t stream, std::size_t n, float amplitude) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = f32_to_bf16(hash_uniform(stream, i, amplitude));
  return v;
}

// record.py's gathered_rows: hashed code bytes, then five fp16 scales per row.
std::vector<uint8_t> gathered_rows(uint32_t codes_stream, uint32_t scales_stream, int tokens) {
  const std::size_t n = static_cast<std::size_t>(tokens) * kHeads;
  std::vector<uint8_t> raw(n * kRowBytes);
  for (std::size_t row = 0; row < n; ++row) {
    uint8_t *out = &raw[row * kRowBytes];
    for (int b = 0; b < 80; ++b) out[b] = static_cast<uint8_t>(hash_u32(codes_stream, row * 80 + b) & 0xFF);
    for (int g = 0; g < 5; ++g) {
      const float s = ROW_SCALE_MAG * (0.5F + std::fabs(hash_uniform(scales_stream, row * 5 + g, 1.0F)));
      const uint16_t bits = f32_to_f16(s);
      std::memcpy(out + 80 + 2 * g, &bits, 2);
    }
  }
  return raw;
}

struct Weights {
  Fp8Matrix key, value;
  std::vector<uint16_t> norm_key, norm_query, norm_conv, conv;
};

Weights make_weights() {
  Weights w;
  w.key = make_fp8(S_KEY, kWidth, kEmbed, KEY_SCALE_MAG);
  w.value = make_fp8(S_VALUE, kHidden, kEmbed, VALUE_SCALE_MAG);
  w.norm_key = bf16_vector(S_NORM_KEY, kWidth, NORM_AMP);
  w.norm_query = bf16_vector(S_NORM_QUERY, kWidth, NORM_AMP);
  w.norm_conv = bf16_vector(S_NORM_CONV, kWidth, NORM_AMP);
  w.conv = bf16_vector(S_CONV, static_cast<std::size_t>(kWidth) * kKernel, CONV_AMP);
  return w;
}

// ---- the host model ---------------------------------------------------------------------------

// One sequence through the device's arithmetic: `hidden` ([tokens][kWidth] BF16 values) gains the
// layer's output; `contribution` (if given) receives that output before the residual add;
// `columns` ([kState][kWidth], oldest first) is the conv state, carried.
void host_ngram(const Weights &w, const uint8_t *rows, int tokens, std::vector<float> &hidden,
                std::vector<float> &columns, std::vector<float> *contribution) {
  std::vector<double> e(kEmbed);
  std::vector<float> k(kWidth), v(kHidden);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      const uint8_t *row = rows + (static_cast<std::size_t>(t) * kHeads + h) * kRowBytes;
      for (int i = 0; i < kHeadDim; ++i) {
        const uint8_t byte = row[i / 2];
        const int nibble = (i & 1) ? (byte >> 4) : (byte & 0xF);
        uint16_t bits;
        std::memcpy(&bits, row + 80 + 2 * (i / 32), 2);
        e[h * kHeadDim + i] = bf(static_cast<float>(nibble - 8) * f16_to_f32(bits));
      }
    }
    double unused;
    for (int o = 0; o < kWidth; ++o) {
      double y;
      fp8_row_f64(w.key, o, e.data(), &y, &unused);
      k[o] = bf(static_cast<float>(y));
    }
    for (int o = 0; o < kHidden; ++o) {
      double y;
      fp8_row_f64(w.value, o, e.data(), &y, &unused);
      v[o] = bf(static_cast<float>(y));
    }
    float *hid = &hidden[static_cast<std::size_t>(t) * kWidth];
    float gates[kStreams], rms[kStreams];
    for (int s = 0; s < kStreams; ++s) {
      float kk = 0.0F, qq = 0.0F;
      for (int i = 0; i < kHidden; ++i) {
        kk += k[s * kHidden + i] * k[s * kHidden + i];
        qq += hid[s * kHidden + i] * hid[s * kHidden + i];
      }
      const float rk = 1.0F / std::sqrt(kk / kHidden + kEps);
      const float rq = 1.0F / std::sqrt(qq / kHidden + kEps);
      float dot = 0.0F;
      for (int i = 0; i < kHidden; ++i) {
        const int c = s * kHidden + i;
        const float kn = bf(k[c] * rk * (1.0F + bf16_to_f32(w.norm_key[c])));
        const float qn = bf(hid[c] * rq * (1.0F + bf16_to_f32(w.norm_query[c])));
        dot += bf(kn * qn);
      }
      const float g = bf(bf(dot) / std::sqrt(static_cast<float>(kHidden)));
      const float root = bf(std::sqrt(bf(std::fmax(std::fabs(g), 1e-6F))));
      const float signed_root = g > 0.0F ? root : (g < 0.0F ? -root : 0.0F);
      gates[s] = bf(1.0F / (1.0F + std::exp(-signed_root)));
      float gg = 0.0F;
      for (int i = 0; i < kHidden; ++i) {
        const float gv = bf(gates[s] * v[i]);
        gg += gv * gv;
      }
      rms[s] = 1.0F / std::sqrt(gg / kHidden + kEps);
    }
    for (int c = 0; c < kWidth; ++c) {
      const int s = c / kHidden, i = c % kHidden;
      const float gv = bf(gates[s] * v[i]);
      const float gvn = bf(gv * rms[s] * (1.0F + bf16_to_f32(w.norm_conv[c])));
      float acc = 0.0F;
      for (int tap = 0; tap < kKernel - 1; ++tap) {
        acc += bf16_to_f32(w.conv[static_cast<std::size_t>(c) * kKernel + tap]) *
               columns[static_cast<std::size_t>(tap * kDilation) * kWidth + c];
      }
      acc += bf16_to_f32(w.conv[static_cast<std::size_t>(c) * kKernel + kKernel - 1]) * gvn;
      const float conv_out = bf(acc);
      const float add = bf(gv + bf(conv_out / (1.0F + std::exp(-conv_out))));
      if (contribution != nullptr) (*contribution)[static_cast<std::size_t>(t) * kWidth + c] = add;
      hid[c] = bf(hid[c] + add);
      for (int m = 0; m < kState - 1; ++m) {
        columns[static_cast<std::size_t>(m) * kWidth + c] = columns[static_cast<std::size_t>(m + 1) * kWidth + c];
      }
      columns[static_cast<std::size_t>(kState - 1) * kWidth + c] = gvn;
    }
  }
}

struct Diff {
  double max_abs = 0.0, sum_abs = 0.0, max_ref = 0.0;
  std::size_t n = 0, worst = 0;
  double mean_abs() const { return n == 0 ? 0.0 : sum_abs / static_cast<double>(n); }
};

Diff diff(const std::vector<float> &got, const std::vector<float> &want) {
  Diff d;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double a = std::fabs(static_cast<double>(got[i]) - want[i]);
    if (a > d.max_abs) {
      d.max_abs = a;
      d.worst = i;
    }
    d.sum_abs += a;
    d.max_ref = std::max(d.max_ref, std::fabs(static_cast<double>(want[i])));
  }
  d.n = want.size();
  return d;
}

void report(const char *what, const Diff &d) {
  std::printf("  %-34s max |d| %.6f (at %zu)  mean |d| %.7f  max |ref| %.4f\n", what, d.max_abs, d.worst,
              d.mean_abs(), d.max_ref);
}

// The FP8-vs-BF16-weight tolerance of the host model against the recorded reference: the
// reference rounds each weight to BF16 (2^-9 relative) before an fp32 product; every later step
// rounds as the device does. Measured with --host-only on the recorded fixture: max |d| 0.03125
// (one BF16 ulp at the largest outputs, 2^-7.4 of max|ref| 5.125), mean |d| 6.4e-4 (2^-13.0 of
// max|ref|); held with a 2x margin: max |d| <= 2^-6 max|ref|, mean |d| <= 2^-12 max|ref|. (The
// conv's taps in reverse order measure max |d| 4.88, mean 0.128.)
bool within_reference(const Diff &d) {
  return d.max_abs <= std::ldexp(d.max_ref, -6) && d.mean_abs() <= std::ldexp(d.max_ref, -12);
}

// Device against the host model (or against itself across call shapes): both run the same
// arithmetic; the FP8 linear's fp32 sums against the host's fp64 ones (and the GEMV route's order
// against the tensor cores') can only flip a BF16 rounding here and there, about one BF16 ulp of
// the value it touches. max |d| <= 2^-7 max|ref| (two ulps at the largest outputs), mean |d| <=
// 2^-14 max|ref| (a quarter of the reference tolerance's mean).
bool within_device(const Diff &d) {
  return d.max_abs <= std::ldexp(d.max_ref, -7) && d.mean_abs() <= std::ldexp(d.max_ref, -14);
}

std::vector<float> to_float(const std::vector<uint16_t> &bits) {
  std::vector<float> v(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) v[i] = bf16_to_f32(bits[i]);
  return v;
}

// ---- the device harness -----------------------------------------------------------------------

constexpr int kSlots = 4;

fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = kHidden;
  g.streams = kStreams;
  g.ngram_size = kDilation;
  g.ngram_heads = kHeads;
  g.ngram_embed_dim = kEmbed;
  g.ngram_conv_kernel = kKernel;
  g.ngram_layer = 1;
  g.rms_norm_eps = kEps;
  return g;
}

struct Device {
  DeviceBytes key, value, norm_key, norm_query, norm_conv, conv, state;
  fn::NgramWeights w;
  fn::Context ctx;

  explicit Device(const Weights &h)
      : key(h.key.payload.size()), value(h.value.payload.size()), norm_key(kWidth * 2),
        norm_query(kWidth * 2), norm_conv(kWidth * 2), conv(static_cast<std::size_t>(kWidth) * kKernel * 2),
        state(static_cast<std::size_t>(kSlots) * kState * kWidth * 2) {
    upload(key, h.key.payload);
    upload(value, h.value.payload);
    upload(norm_key, h.norm_key);
    upload(norm_query, h.norm_query);
    upload(norm_conv, h.norm_conv);
    upload(conv, h.conv);
    // Garbage in every slot: a sequence's first call must not read it.
    std::vector<uint16_t> dirty(static_cast<std::size_t>(kSlots) * kState * kWidth);
    for (std::size_t i = 0; i < dirty.size(); ++i) dirty[i] = f32_to_bf16(hash_uniform(9001, i, 3.0F));
    upload(state, dirty);
    w.key_proj = fn::Linear{key.p, kWidth, kEmbed, fn::WeightFormat::Fp8RowScale};
    w.value_proj = fn::Linear{value.p, kHidden, kEmbed, fn::WeightFormat::Fp8RowScale};
    w.norm_key = norm_key.p;
    w.norm_query = norm_query.p;
    w.norm_conv = norm_conv.p;
    w.conv = conv.p;
    ctx.g = geometry();
    ctx.ngram.conv_columns = state.p;
  }

  // One call: `lanes` sequences of `tokens` tokens each (lane-major rows and hidden), lane l in
  // slot slots[l] at position positions[l].
  void call(const std::vector<uint8_t> &rows, std::vector<uint16_t> &hidden, int lanes, int tokens,
            const std::vector<int32_t> &slots, const std::vector<int32_t> &positions) {
    const int n = lanes * tokens;
    DeviceBytes d_rows(rows.size()), d_hidden(hidden.size() * 2), d_slots(lanes * 4), d_positions(lanes * 4);
    upload(d_rows, rows);
    upload(d_hidden, hidden);
    upload(d_slots, slots);
    upload(d_positions, positions);
    // Exactly the declared scratch: a call that needs more fails here.
    ninfer::DeviceArena scratch(fn::fn_ngram_add_scratch_bytes(ctx.g, n));
    fn::Batch batch;
    batch.lanes = lanes;
    batch.tokens = tokens;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    batch.max_visible = positions[0] + tokens;
    NG_RC(fn::fn_ngram_add(ctx, w, batch, d_rows.p, d_hidden.p, scratch, nullptr));
    NG_CUDA(cudaDeviceSynchronize());
    hidden = download<uint16_t>(d_hidden.p, hidden.size());
  }
};

// Rows [first, first + count) of a sequence's [tokens][kHeads][kRowBytes] rows.
std::vector<uint8_t> row_slice(const std::vector<uint8_t> &rows, int first, int count) {
  const std::size_t per = static_cast<std::size_t>(kHeads) * kRowBytes;
  return std::vector<uint8_t>(rows.begin() + first * per, rows.begin() + (first + count) * per);
}

std::vector<uint16_t> hidden_slice(const std::vector<uint16_t> &hidden, int first, int count) {
  return std::vector<uint16_t>(hidden.begin() + static_cast<std::size_t>(first) * kWidth,
                               hidden.begin() + static_cast<std::size_t>(first + count) * kWidth);
}

// A whole sequence through calls of the given token counts, in one slot.
std::vector<uint16_t> run_chunks(Device &d, const std::vector<uint8_t> &rows, const std::vector<uint16_t> &hidden,
                                 const std::vector<int> &chunks, int32_t slot) {
  std::vector<uint16_t> out;
  int at = 0;
  for (const int count : chunks) {
    std::vector<uint16_t> h = hidden_slice(hidden, at, count);
    d.call(row_slice(rows, at, count), h, 1, count, {slot}, {at});
    out.insert(out.end(), h.begin(), h.end());
    at += count;
  }
  return out;
}

bool host_reference_check(const Weights &w, const std::vector<uint8_t> &rows, const std::vector<uint16_t> &hidden_in,
                          const std::vector<float> &ple_out) {
  std::vector<float> hidden = to_float(hidden_in);
  std::vector<float> columns(static_cast<std::size_t>(kState) * kWidth, 0.0F);
  std::vector<float> contribution(static_cast<std::size_t>(kTokens) * kWidth);
  host_ngram(w, rows.data(), kTokens, hidden, columns, &contribution);
  const Diff d = diff(contribution, ple_out);
  report("host model vs recorded reference", d);
  check(within_reference(d), "the host model is within the FP8 tolerance of the recorded reference");
  return within_reference(d);
}

}  // namespace

int main(int argc, char **argv) {
  const bool host_only = argc > 1 && std::string(argv[1]) == "--host-only";
  const auto fx = read_fixture(std::string(IGNIS_FLASH_NEXT_NGRAM_FIXTURE_DIR) + "/ngram_ref.bin");
  const FixtureTensor &rows_t = need(fx, "rows");
  const FixtureTensor &ple_t = need(fx, "ple_out");
  const std::vector<uint8_t> rows = rows_t.bytes;
  std::vector<uint16_t> ple_bits(ple_t.bytes.size() / 2);
  std::memcpy(ple_bits.data(), ple_t.bytes.data(), ple_t.bytes.size());
  const std::vector<float> ple_out = to_float(ple_bits);
  const std::vector<uint16_t> hidden_in = bf16_vector(S_HIDDEN, static_cast<std::size_t>(kTokens) * kWidth, HIDDEN_AMP);
  // The fixture's rows are the hash's: a drifted generator would test nothing.
  check(rows == gathered_rows(S_ROW_CODES, S_ROW_SCALES, kTokens), "the fixture's rows are regenerated");

  const Weights w = make_weights();
  std::printf("[host model]\n");
  host_reference_check(w, rows, hidden_in, ple_out);
  if (host_only) {
    std::printf("%s\n", g_failed == 0 ? "PASS (host only)" : "FAIL (host only)");
    return g_failed == 0 ? 0 : 1;
  }

  NG_CUDA(cudaSetDevice(0));
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return 1;
  }
  Device device(w);

  // Host model output (with the residual add), the device's target.
  std::vector<float> host_hidden = to_float(hidden_in);
  {
    std::vector<float> columns(static_cast<std::size_t>(kState) * kWidth, 0.0F);
    host_ngram(w, rows.data(), kTokens, host_hidden, columns, nullptr);
  }

  std::printf("[device]\n");
  const std::vector<uint16_t> one_shot = run_chunks(device, rows, hidden_in, {kTokens}, 2);
  {
    const Diff d = diff(to_float(one_shot), host_hidden);
    report("one shot vs host model", d);
    check(within_device(d), "one shot matches the host model");
  }
  {
    const std::vector<uint16_t> chunked = run_chunks(device, rows, hidden_in, {5, 1, 1, 1, 2, 6}, 1);
    const Diff d = diff(to_float(chunked), to_float(one_shot));
    report("5+1+1+1+2+6 vs one shot", d);
    check(within_device(d), "chunked prefill and decode match the one-shot call");
  }
  {
    // Three sequences decoded together, one token per call, slots 3, 0, 2.
    const int lanes = 3;
    const std::vector<int32_t> slots = {3, 0, 2};
    std::vector<std::vector<uint8_t>> lane_rows;
    std::vector<std::vector<uint16_t>> lane_hidden, lane_alone, lane_out(lanes);
    for (int l = 0; l < lanes; ++l) {
      lane_rows.push_back(l == 0 ? rows : gathered_rows(S_ROW_CODES + 10 * l, S_ROW_SCALES + 10 * l, kTokens));
      lane_hidden.push_back(l == 0 ? hidden_in
                                   : bf16_vector(S_HIDDEN + 10 * l, static_cast<std::size_t>(kTokens) * kWidth, HIDDEN_AMP));
      lane_alone.push_back(run_chunks(device, lane_rows[l], lane_hidden[l], {kTokens}, 1));
    }
    for (int t = 0; t < kTokens; ++t) {
      std::vector<uint8_t> step_rows;
      std::vector<uint16_t> step_hidden;
      std::vector<int32_t> positions;
      for (int l = 0; l < lanes; ++l) {
        const auto r = row_slice(lane_rows[l], t, 1);
        const auto h = hidden_slice(lane_hidden[l], t, 1);
        step_rows.insert(step_rows.end(), r.begin(), r.end());
        step_hidden.insert(step_hidden.end(), h.begin(), h.end());
        positions.push_back(t);
      }
      device.call(step_rows, step_hidden, lanes, 1, slots, positions);
      for (int l = 0; l < lanes; ++l) {
        lane_out[l].insert(lane_out[l].end(), step_hidden.begin() + static_cast<std::size_t>(l) * kWidth,
                           step_hidden.begin() + static_cast<std::size_t>(l + 1) * kWidth);
      }
    }
    for (int l = 0; l < lanes; ++l) {
      const Diff d = diff(to_float(lane_out[l]), to_float(lane_alone[l]));
      report(("lane " + std::to_string(l) + " decoded vs alone").c_str(), d);
      check(within_device(d), "lane " + std::to_string(l) + " decoded beside two others matches its own run");
    }
  }
  {
    // One call whose lanes sit at different positions: 5 tokens in, fresh, 12 tokens in.
    const std::vector<int> prefix = {5, 0, 12};
    const std::vector<int32_t> slots = {0, 3, 2};
    std::vector<uint8_t> step_rows;
    std::vector<uint16_t> step_hidden;
    std::vector<std::vector<uint16_t>> alone;
    std::vector<int32_t> positions;
    for (int l = 0; l < 3; ++l) {
      const int length = prefix[l] + 1;
      const auto lane_rows = gathered_rows(S_ROW_CODES + 200 + l, S_ROW_SCALES + 200 + l, length);
      const auto lane_hidden = bf16_vector(S_HIDDEN + 200 + l, static_cast<std::size_t>(length) * kWidth, HIDDEN_AMP);
      const auto whole = run_chunks(device, lane_rows, lane_hidden, {length}, 1);
      alone.push_back(hidden_slice(whole, prefix[l], 1));
      if (prefix[l] > 0) run_chunks(device, lane_rows, lane_hidden, {prefix[l]}, slots[l]);
      const auto r = row_slice(lane_rows, prefix[l], 1);
      const auto h = hidden_slice(lane_hidden, prefix[l], 1);
      step_rows.insert(step_rows.end(), r.begin(), r.end());
      step_hidden.insert(step_hidden.end(), h.begin(), h.end());
      positions.push_back(prefix[l]);
    }
    device.call(step_rows, step_hidden, 3, 1, slots, positions);
    for (int l = 0; l < 3; ++l) {
      const Diff d = diff(to_float(hidden_slice(step_hidden, l, 1)), to_float(alone[l]));
      report(("lane at position " + std::to_string(prefix[l]) + " vs alone").c_str(), d);
      check(within_device(d), "a lane at position " + std::to_string(prefix[l]) + " matches its own run");
    }
  }
  {
    // The decode rounds as a CUDA graph: captured once, replayed with fixed buffers whose
    // contents (rows, hidden, positions) are refreshed before each launch.
    const int lanes = 3;
    const std::vector<int32_t> slots = {1, 3, 0};
    std::vector<std::vector<uint8_t>> lane_rows;
    std::vector<std::vector<uint16_t>> lane_hidden;
    for (int l = 0; l < lanes; ++l) {
      lane_rows.push_back(gathered_rows(S_ROW_CODES + 300 + l, S_ROW_SCALES + 300 + l, kTokens));
      lane_hidden.push_back(bf16_vector(S_HIDDEN + 300 + l, static_cast<std::size_t>(kTokens) * kWidth, HIDDEN_AMP));
    }
    auto round_inputs = [&](int t, std::vector<uint8_t> &r, std::vector<uint16_t> &h) {
      r.clear();
      h.clear();
      for (int l = 0; l < lanes; ++l) {
        const auto rr = row_slice(lane_rows[l], t, 1);
        const auto hh = hidden_slice(lane_hidden[l], t, 1);
        r.insert(r.end(), rr.begin(), rr.end());
        h.insert(h.end(), hh.begin(), hh.end());
      }
    };
    // Eager rounds.
    std::vector<uint16_t> eager;
    for (int t = 0; t < kTokens; ++t) {
      std::vector<uint8_t> r;
      std::vector<uint16_t> h;
      round_inputs(t, r, h);
      device.call(r, h, lanes, 1, slots, std::vector<int32_t>(lanes, t));
      eager.insert(eager.end(), h.begin(), h.end());
    }
    // The same rounds replayed from one captured graph.
    const std::size_t row_bytes = static_cast<std::size_t>(lanes) * kHeads * kRowBytes;
    const std::size_t hidden_count = static_cast<std::size_t>(lanes) * kWidth;
    DeviceBytes d_rows(row_bytes), d_hidden(hidden_count * 2), d_slots(lanes * 4), d_positions(lanes * 4);
    upload(d_slots, slots);
    ninfer::DeviceArena scratch(fn::fn_ngram_add_scratch_bytes(device.ctx.g, lanes));
    fn::Batch batch;
    batch.lanes = lanes;
    batch.tokens = 1;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    batch.max_visible = kTokens;
    cudaStream_t stream;
    NG_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    NG_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    NG_RC(fn::fn_ngram_add(device.ctx, device.w, batch, d_rows.p, d_hidden.p, scratch, stream));
    NG_CUDA(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    NG_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<uint16_t> replayed;
    for (int t = 0; t < kTokens; ++t) {
      std::vector<uint8_t> r;
      std::vector<uint16_t> h;
      round_inputs(t, r, h);
      upload(d_rows, r);
      upload(d_hidden, h);
      upload(d_positions, std::vector<int32_t>(lanes, t));
      NG_CUDA(cudaGraphLaunch(exec, stream));
      NG_CUDA(cudaStreamSynchronize(stream));
      const auto out = download<uint16_t>(d_hidden.p, hidden_count);
      replayed.insert(replayed.end(), out.begin(), out.end());
    }
    NG_CUDA(cudaGraphExecDestroy(exec));
    NG_CUDA(cudaGraphDestroy(graph));
    NG_CUDA(cudaStreamDestroy(stream));
    const Diff d = diff(to_float(replayed), to_float(eager));
    report("graph replay vs eager rounds", d);
    check(d.max_abs == 0.0, "the decode rounds replayed from a captured graph equal the eager rounds");
  }
  {
    // 300 tokens: one call spans two internal windows.
    const int tokens = 300;
    const auto long_rows = gathered_rows(S_ROW_CODES + 100, S_ROW_SCALES + 100, tokens);
    const auto long_hidden = bf16_vector(S_HIDDEN + 100, static_cast<std::size_t>(tokens) * kWidth, HIDDEN_AMP);
    const auto whole = run_chunks(device, long_rows, long_hidden, {tokens}, 0);
    const auto halves = run_chunks(device, long_rows, long_hidden, {150, 150}, 0);
    const Diff d = diff(to_float(whole), to_float(halves));
    report("300 in one call vs 150 + 150", d);
    check(within_device(d), "a call spanning two windows matches two calls");
  }
  {
    // Slot 1 was left holding the chunked sequence's state; a fresh sequence there starts at zero.
    const std::vector<uint16_t> again = run_chunks(device, rows, hidden_in, {kTokens}, 1);
    const Diff d = diff(to_float(again), to_float(one_shot));
    report("fresh sequence in a used slot", d);
    check(d.max_abs == 0.0, "a sequence's first call ignores the slot's previous state");
  }

  std::printf("%s\n", g_failed == 0 ? "PASS" : "FAIL");
  return g_failed == 0 ? 0 : 1;
}
