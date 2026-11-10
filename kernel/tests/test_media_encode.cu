// GitHub #557 (ADR 0087): the media encode at both text models' merger widths -- OURS, not vendored.
//
// Flash-Next's vision companion holds the 27B's tower objects with a merger that writes Flash-Next's
// 2,560 rows. Its canary's first image failed in the encode, on the merger's fc2 [2560, 4608], which
// the W8 dispatch had no route for: no test ran an encode at any width but the 27B's. Since GitHub
// #556 every matrix of Flash-Next's companion is W8; the 27B's keeps its grouped formats (Q6 patch
// embedding, Q4 qkv and fc1, Q5 output and fc2, W8 merger). Here, each family in its own formats:
//
//   routes   every linear an encode runs (kernel/src/vision_encode.cu's `encode`), in the family's
//            stored formats, has a route at every width it runs at, for every item up to 32,768
//            merged tokens (the ceiling the tower's Q arms and the W8 K-4608 arm share): the tower's
//            linears at 4 x tokens patches, the merger's fc1 at the tokens, its fc2 once per
//            embedding pool page at the page's width (1 .. IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS), at
//            2,560 (Flash-Next, every matrix W8) and 5,120 (the 27B, grouped). Asked of
//            ops::linear_workspace_capacity_bytes, which asks the dispatch the op takes. Negative
//            control: a 3,072-wide merger, which the dispatch does not register, has no route.
//            Host-only: no device call.
//   encode   ignis_media_encode end to end, through ignis_model_load: the 27B program with no text
//            layers at hidden 2,560 and 5,120 (bind_vision binds the merger's fc2 at the model's
//            hidden, the pool's pages and the encode take it from there, as a Flash-Next load
//            does), under each vision residency. The tower is the family's companion's: its 333
//            tensors in their stored formats, each a 4 KiB-aligned span of one allocation (VRAM, or
//            mapped pinned host memory staged per encode), the 27 blocks and the patch embedding all
//            zero (each block adds exactly 0 to the residual), the merger random. Items of 1, 5,
//            16, 17 and 133 merged tokens put each route of the fc2 under a page (T 1, 5, 16, 17,
//            then 128 and 5 over two pages). Each embedding against fp64 that rounds to BF16
//            where the device stores BF16 (every column of the small items, ten across the two
//            pages of the large one; negative control: the large item's columns against their
//            neighbours'), and the host residency's bytes equal the device's.
//
//   ignis_kernel_media_encode_test [--cases routes,encode] [--bound B]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails the encode case.

#include "ignis_model.h"
#include "ignis_step.h"
#include "model_internal.h"
#include "moe_fixture.h"

#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr int32_t kFlashNextHidden = 2560;
constexpr int32_t k27bHidden = 5120;

// A family's companion: its merger's width and its tower's stored formats (crates/artifact's
// `flash_next::vision::entries` and `inventory::vision_scope_27b`); both mergers are W8.
struct Family {
  const char *name;
  int32_t hidden;
  ninfer::QType patch, qkv, output, fc1, fc2;
};
constexpr Family kFlashNext{"Flash-Next", kFlashNextHidden, ninfer::QType::W8G32_F16S, ninfer::QType::W8G32_F16S,
                            ninfer::QType::W8G32_F16S, ninfer::QType::W8G32_F16S, ninfer::QType::W8G32_F16S};
constexpr Family k27b{"27B", k27bHidden, ninfer::QType::Q6G64_F16S, ninfer::QType::Q4G64_F16S,
                      ninfer::QType::Q5G64_F16S, ninfer::QType::Q4G64_F16S, ninfer::QType::Q5G64_F16S};
constexpr std::array<Family, 2> kFamilies{kFlashNext, k27b};
// The most merged tokens an item may hold: past it the tower's Q arms (131,072 patches) and the
// W8 K-4608 arm (the merger's fc1) have no route.
constexpr int32_t kMaxItemTokens = 32768;

struct Problem {
  const char *name;
  ninfer::QType qtype;
  int32_t n, k;
};

const char *format_name(ninfer::QType qtype) {
  switch (qtype) {
    case ninfer::QType::W8G32_F16S: return "W8";
    case ninfer::QType::Q4G64_F16S: return "Q4";
    case ninfer::QType::Q5G64_F16S: return "Q5";
    case ninfer::QType::Q6G64_F16S: return "Q6";
    default: return "?";
  }
}

std::string shape(const Problem &p) {
  return std::string(p.name) + " (" + format_name(p.qtype) + ") [" + std::to_string(p.n) + ", " + std::to_string(p.k) + "]";
}

// The first T of first, first + step, .. last at which `p` has no route, or 0.
int32_t first_unrouted(const Problem &p, int32_t first, int32_t last, int32_t step) {
  for (int32_t t = first; t <= last; t += step) {
    try {
      (void)ninfer::ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, ninfer::ops::LinearPolicy::A16Only, t, t);
    } catch (const std::exception &) {
      return t;
    }
  }
  return 0;
}

void expect_routed(const Problem &p, int32_t first, int32_t last, int32_t step) {
  const int32_t t = first_unrouted(p, first, last, step);
  std::printf("  %-40s T %6d .. %6d: %s\n", shape(p).c_str(), first, last,
              t == 0 ? "routed" : ("no route at T " + std::to_string(t)).c_str());
  check(t == 0, "routes: " + shape(p) + " has no route at T " + std::to_string(t));
}

void case_routes() {
  std::printf("routes: every linear of an encode at every width it runs, items of 1 .. %d merged tokens\n",
              kMaxItemTokens);
  using ninfer::QType;
  for (const Family &f : kFamilies) {
    std::printf(" %s, the merger %d wide:\n", f.name, f.hidden);
    const Problem tower[] = {
        {"patch embedding", f.patch, kVisionHidden, kVisionPatchDim},
        {"block qkv", f.qkv, 3 * kVisionHidden, kVisionHidden},
        {"block output", f.output, kVisionHidden, kVisionHidden},
        {"block fc1", f.fc1, kVisionIntermediate, kVisionHidden},
        {"block fc2", f.fc2, kVisionHidden, kVisionIntermediate},
    };
    for (const Problem &p : tower) {
      expect_routed(p, kVisionMergeUnit, kMaxItemTokens * kVisionMergeUnit, kVisionMergeUnit);
    }
    expect_routed({"merger fc1", QType::W8G32_F16S, kVisionMergerHidden, kVisionMergerHidden}, 1, kMaxItemTokens, 1);
    expect_routed({"merger fc2", QType::W8G32_F16S, f.hidden, kVisionMergerHidden}, 1,
                  IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS, 1);
  }
  const Problem unregistered{"merger fc2", QType::W8G32_F16S, 3072, kVisionMergerHidden};
  check(first_unrouted(unregistered, 1, 1, 1) == 1,
        "routes: the negative control (" + shape(unregistered) + ", unregistered) has no route");
}

// ---- the encode -------------------------------------------------------------------------------

constexpr int32_t kVocab = 14336;  // the no-layer load's text tables (test_model_load_decode_lanes.cpp)
constexpr uint32_t kEnvelope = 512;
constexpr uint32_t kMaxContext = 1024;
constexpr uint32_t kPrefillChunk = 128;
constexpr int32_t kSide = 48;     // the position table's side
constexpr double kEps = 1e-6;     // the merger's LayerNorm
constexpr std::size_t kSpan = 4096;  // the binder's span alignment in a mapped container

double bf(double v) { return bf16_to_f32(f32_to_bf16(static_cast<float>(v))); }

std::size_t align_up(std::size_t bytes, std::size_t alignment) { return (bytes + alignment - 1) / alignment * alignment; }

int32_t padded(int32_t k) { return (k + 127) / 128 * 128; }

// One W8G32 matrix as the artifact stores it (row-split-k128-v1): codes int8 [rows][K padded to
// 128], zero in the padding, then FP16 scales [rows][padded / 32]. Random codes with scales of about
// 1 / sqrt(K var(code)) keep a unit-scale input at unit scale.
struct W8 {
  int32_t rows, cols, pad;
  std::vector<int8_t> codes;
  std::vector<uint16_t> scales;
  W8(int32_t r, int32_t c, uint32_t stream)
      : rows(r), cols(c), pad(padded(c)), codes(static_cast<std::size_t>(r) * pad, 0),
        scales(static_cast<std::size_t>(r) * (pad / 32)) {
    const float base = 1.0f / std::sqrt(5376.0f * static_cast<float>(c));
    for (int32_t row = 0; row < r; ++row) {
      for (int32_t col = 0; col < c; ++col) {
        const std::size_t i = static_cast<std::size_t>(row) * pad + col;
        codes[i] = static_cast<int8_t>(static_cast<int>(hash_u32(stream, i) % 255u) - 127);
      }
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
      scales[i] = f32_to_f16(base * (0.5f + std::fabs(hash_uniform(stream ^ 0x5CA1Eu, i, 1.0f))));
    }
  }
  // Row `row` dotted with x[0, cols), in fp64, one scale per group of 32.
  double dot(int32_t row, const double *x) const {
    const int8_t *code = &codes[static_cast<std::size_t>(row) * pad];
    const uint16_t *scale = &scales[static_cast<std::size_t>(row) * (pad / 32)];
    double d = 0.0;
    for (int32_t g = 0; g * 32 < cols; ++g) {
      double s = 0.0;
      for (int32_t k = g * 32; k < std::min(cols, g * 32 + 32); ++k) s += code[k] * x[k];
      d += s * f16_to_f32(scale[g]);
    }
    return d;
  }
};

// A BF16 vector: center + uniform(-amplitude, amplitude).
std::vector<uint16_t> bf16_vector(std::size_t n, uint32_t stream, float center, float amplitude) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = f32_to_bf16(center + hash_uniform(stream, i, amplitude));
  return v;
}

// What is not zero of the tower: the patch embedding's bias, the position table and the merger,
// whose fc2 is the 27B's 5,120 rows wide; a narrower model's merger is its first `hidden` rows.
struct Tower {
  std::vector<uint16_t> patch_bias = bf16_vector(kVisionHidden, 101, 0.0f, 0.1f);
  std::vector<uint16_t> table = bf16_vector(static_cast<std::size_t>(kVisionPositionEmbeddings) * kVisionHidden, 102, 0.0f, 0.1f);
  std::vector<uint16_t> norm_w = bf16_vector(kVisionHidden, 103, 1.0f, 0.25f);
  std::vector<uint16_t> norm_b = bf16_vector(kVisionHidden, 104, 0.0f, 0.1f);
  W8 fc1{kVisionMergerHidden, kVisionMergerHidden, 105};
  std::vector<uint16_t> fc1_bias = bf16_vector(kVisionMergerHidden, 106, 0.0f, 0.1f);
  W8 fc2{k27bHidden, kVisionMergerHidden, 107};
  std::vector<uint16_t> fc2_bias = bf16_vector(k27bHidden, 108, 0.0f, 0.1f);
};

// One tensor of the tower image: its span's offset and bytes, its planes' offsets from it.
struct Entry {
  std::string name;
  std::vector<int32_t> shape;
  int32_t qtype = IGNIS_QTYPE_BF16_CTRL;
  std::size_t offset = 0, bytes = 0, high = 0, scales = 0;
  bool has_high = false;
};

// The tower as the binder maps a container: each tensor a span of its own, 4 KiB-aligned at both
// ends, in one allocation -- which the host residency's staging runs are built on.
struct Image {
  std::vector<uint8_t> bytes;
  std::vector<Entry> entries;
  // A zeroed span for `name`; its bytes until the next add.
  uint8_t *add(const std::string &name, std::vector<int32_t> shape, int32_t qtype) {
    Entry e{name, shape, qtype};
    std::size_t elements = 1;
    for (const int32_t d : shape) elements *= static_cast<std::size_t>(d);
    if (qtype == IGNIS_QTYPE_BF16_CTRL) {
      e.bytes = elements * 2;
    } else {
      // Row-split planes: the 4 low bits (Q4/Q5/Q6) or the code (W8), the Q5/Q6 high bits, then
      // the FP16 scales (one per 64, W8 one per 32), each row K-padded to 128.
      const std::size_t rows = static_cast<std::size_t>(shape[0]);
      const std::size_t k = static_cast<std::size_t>(padded(shape[1]));
      const std::size_t low = qtype == IGNIS_QTYPE_W8G32_F16S ? k : k / 2;
      const std::size_t high = qtype == IGNIS_QTYPE_Q5G64_F16S ? k / 8 : qtype == IGNIS_QTYPE_Q6G64_F16S ? k / 4 : 0;
      const std::size_t scale = (qtype == IGNIS_QTYPE_W8G32_F16S ? k / 32 : k / 64) * 2;
      e.high = rows * low;
      e.has_high = high > 0;
      e.scales = rows * (low + high);
      e.bytes = rows * (low + high + scale);
    }
    e.offset = align_up(bytes.size(), kSpan);
    bytes.resize(align_up(e.offset + e.bytes, kSpan), 0);
    entries.push_back(e);
    return bytes.data() + e.offset;
  }
  void add_bf16(const std::string &name, const std::vector<uint16_t> &v, std::size_t n) {
    std::memcpy(add(name, {static_cast<int32_t>(n)}, IGNIS_QTYPE_BF16_CTRL), v.data(), n * 2);
  }
  // The first `rows` rows of `m`.
  void add_w8(const std::string &name, const W8 &m, int32_t rows) {
    uint8_t *at = add(name, {rows, m.cols}, IGNIS_QTYPE_W8G32_F16S);
    const std::size_t codes = static_cast<std::size_t>(rows) * m.pad;
    std::memcpy(at, m.codes.data(), codes);
    std::memcpy(at + codes, m.scales.data(), static_cast<std::size_t>(rows) * (m.pad / 32) * 2);
  }
};

// Family `f`'s companion's 333 tensors (crates/artifact's vision inventory, bind_vision's shapes),
// in its stored formats, for a model `f.hidden` wide.
Image tower_image(const Tower &t, const Family &f) {
  const int32_t h = kVisionHidden, inter = kVisionIntermediate, merged = kVisionMergerHidden, hidden = f.hidden;
  const auto code = [](ninfer::QType qtype) { return static_cast<int32_t>(qtype); };
  Image im;
  im.add("vision/patch_embedding", {h, kVisionPatchDim}, code(f.patch));
  im.add_bf16("vision/patch_embedding_bias", t.patch_bias, h);
  std::memcpy(im.add("vision/position_embedding", {kVisionPositionEmbeddings, h}, IGNIS_QTYPE_BF16_CTRL),
              t.table.data(), t.table.size() * 2);
  for (int32_t b = 0; b < kVisionLayers; ++b) {
    const std::string p = "vision/layers/" + std::to_string(b) + "/";
    im.add(p + "attention/qkv", {3 * h, h}, code(f.qkv));
    im.add(p + "attention/qkv_bias", {3 * h}, IGNIS_QTYPE_BF16_CTRL);
    im.add(p + "attention/output", {h, h}, code(f.output));
    im.add(p + "attention/output_bias", {h}, IGNIS_QTYPE_BF16_CTRL);
    im.add(p + "mlp/fc1", {inter, h}, code(f.fc1));
    im.add(p + "mlp/fc1_bias", {inter}, IGNIS_QTYPE_BF16_CTRL);
    im.add(p + "mlp/fc2", {h, inter}, code(f.fc2));
    im.add(p + "mlp/fc2_bias", {h}, IGNIS_QTYPE_BF16_CTRL);
    for (const char *norm : {"norm1/weight", "norm1/bias", "norm2/weight", "norm2/bias"}) {
      im.add(p + norm, {h}, IGNIS_QTYPE_BF16_CTRL);
    }
  }
  im.add_w8("vision/merger/fc1", t.fc1, merged);
  im.add_bf16("vision/merger/fc1_bias", t.fc1_bias, merged);
  im.add_w8("vision/merger/fc2", t.fc2, hidden);
  im.add_bf16("vision/merger/fc2_bias", t.fc2_bias, hidden);
  im.add_bf16("vision/merger/norm/weight", t.norm_w, h);
  im.add_bf16("vision/merger/norm/bias", t.norm_b, h);
  return im;
}

// The image in VRAM, or in mapped pinned host memory.
struct Placed {
  uint8_t *base = nullptr;
  bool host = false;
  Placed(const Image &im, bool in_host) : host(in_host) {
    void *p = nullptr;
    if (host) {
      MOE_CUDA(cudaHostAlloc(&p, im.bytes.size(), cudaHostAllocMapped | cudaHostAllocPortable));
      std::memcpy(p, im.bytes.data(), im.bytes.size());
    } else {
      MOE_CUDA(cudaMalloc(&p, im.bytes.size()));
      MOE_CUDA(cudaMemcpy(p, im.bytes.data(), im.bytes.size(), cudaMemcpyHostToDevice));
    }
    base = static_cast<uint8_t *>(p);
  }
  ~Placed() { host ? cudaFreeHost(base) : cudaFree(base); }
  Placed(const Placed &) = delete;
  Placed &operator=(const Placed &) = delete;
};

ignis_bound_tensor bound_tensor(const char *name, int32_t qtype, const std::vector<int32_t> &shape, const void *qdata,
                         const void *qhigh, const void *scales, uint64_t bytes) {
  ignis_bound_tensor t{};
  t.name = name;
  t.qtype = qtype;
  t.layout = qtype == IGNIS_QTYPE_BF16_CTRL ? IGNIS_LAYOUT_CONTIGUOUS : IGNIS_LAYOUT_ROW_SPLIT;
  t.qdata = qdata;
  t.qhigh = qhigh;
  t.scales = scales;
  t.bytes = bytes;
  t.ndim = static_cast<uint32_t>(shape.size());
  for (std::size_t d = 0; d < 4; ++d) {
    t.shape[d] = d < shape.size() ? shape[d] : 1;
    t.padded_shape[d] = t.shape[d];
  }
  return t;
}

// One image of one frame (crates/core's vision_item_control): the patches in 2x2 merge-block
// order, the bilinear corners at fl32(i * 47 / (side - 1)), the position ids as two planes (every
// row, then every column).
struct Call {
  int32_t h = 0, w = 0, patches = 0;
  std::vector<uint16_t> rows;  // BF16 [patches][1536]
  std::vector<int32_t> pos_rows, pos_cols, cu, idx;
  std::vector<float> wts;
  std::vector<int32_t> position_ids() const {
    std::vector<int32_t> p(pos_rows);
    p.insert(p.end(), pos_cols.begin(), pos_cols.end());
    return p;
  }
  int32_t tokens() const { return patches / kVisionMergeUnit; }
};

float table_coordinate(int32_t i, int32_t size) {
  return size <= 1 ? 0.0f : static_cast<float>(i) * static_cast<float>(kSide - 1) / static_cast<float>(size - 1);
}

Call make_call(int32_t h, int32_t w, uint32_t stream) {
  Call c;
  c.h = h;
  c.w = w;
  for (int32_t by = 0; by < h / 2; ++by) {
    for (int32_t bx = 0; bx < w / 2; ++bx) {
      for (int32_t iy = 0; iy < 2; ++iy) {
        for (int32_t ix = 0; ix < 2; ++ix) {
          const int32_t y = by * 2 + iy, x = bx * 2 + ix;
          c.pos_rows.push_back(y);
          c.pos_cols.push_back(x);
          const float yf = table_coordinate(y, h), xf = table_coordinate(x, w);
          const int32_t y0 = static_cast<int32_t>(yf), x0 = static_cast<int32_t>(xf);
          const int32_t y1 = std::min(y0 + 1, kSide - 1), x1 = std::min(x0 + 1, kSide - 1);
          const float wy = yf - static_cast<float>(y0), wx = xf - static_cast<float>(x0);
          c.idx.insert(c.idx.end(), {y0 * kSide + x0, y0 * kSide + x1, y1 * kSide + x0, y1 * kSide + x1});
          c.wts.insert(c.wts.end(), {(1.0f - wy) * (1.0f - wx), (1.0f - wy) * wx, wy * (1.0f - wx), wy * wx});
        }
      }
    }
  }
  c.patches = h * w;
  c.cu = {0, c.patches};
  c.rows.resize(static_cast<std::size_t>(c.patches) * kVisionPatchDim);
  for (std::size_t i = 0; i < c.rows.size(); ++i) c.rows[i] = f32_to_bf16(hash_uniform(stream, i, 1.0f));
  return c;
}

// The merger's output at some of an item's columns, in fp64 rounded to BF16 where the device stores
// BF16, 5,120 rows a column (a narrower model's are the first rows). The blocks and the patch
// embedding are zero, so the residual the merger reads is bf(bias + the bilinear table); its
// LayerNorm runs over each patch's 1,152 before the 2x2 view, then fc1 + bias, GELU-exact, fc2 +
// bias.
struct Reference {
  std::vector<int32_t> columns;
  std::vector<double> rows;  // [columns][5120]
};

Reference reference(const Tower &t, const Call &c, const std::vector<int32_t> &columns) {
  Reference r{columns, {}};
  std::vector<double> x(kVisionMergerHidden), y(kVisionMergerHidden), hidden(kVisionMergerHidden);
  for (const int32_t column : columns) {
    for (int32_t j = 0; j < kVisionMergeUnit; ++j) {
      const int32_t p = column * kVisionMergeUnit + j;
      double *xp = &x[static_cast<std::size_t>(j) * kVisionHidden];
      for (int32_t n = 0; n < kVisionHidden; ++n) {
        double s = 0.0;
        for (int32_t q = 0; q < 4; ++q) {
          s += c.wts[p * 4 + q] * bf16_to_f32(t.table[static_cast<std::size_t>(c.idx[p * 4 + q]) * kVisionHidden + n]);
        }
        xp[n] = bf(bf16_to_f32(t.patch_bias[n]) + s);
      }
      double mean = 0.0, var = 0.0;
      for (int32_t n = 0; n < kVisionHidden; ++n) mean += xp[n];
      mean /= kVisionHidden;
      for (int32_t n = 0; n < kVisionHidden; ++n) var += (xp[n] - mean) * (xp[n] - mean);
      var /= kVisionHidden;
      const double rstd = 1.0 / std::sqrt(var + kEps);
      for (int32_t n = 0; n < kVisionHidden; ++n) {
        y[static_cast<std::size_t>(j) * kVisionHidden + n] =
            bf((xp[n] - mean) * rstd * bf16_to_f32(t.norm_w[n]) + bf16_to_f32(t.norm_b[n]));
      }
    }
    for (int32_t n = 0; n < kVisionMergerHidden; ++n) {
      const double v = bf(bf(t.fc1.dot(n, y.data())) + bf16_to_f32(t.fc1_bias[n]));
      hidden[n] = bf(0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))));
    }
    for (int32_t n = 0; n < k27bHidden; ++n) {
      r.rows.push_back(bf(bf(t.fc2.dot(n, hidden.data())) + bf16_to_f32(t.fc2_bias[n])));
    }
  }
  return r;
}

// The relative L2 distance of an embedding's `columns` ([hidden, tokens] BF16) from the reference's
// columns `shift` places on (0: their own).
double rel_l2(const std::vector<uint16_t> &got, int32_t hidden, const Reference &r, std::size_t shift) {
  double e = 0.0, n = 0.0;
  for (std::size_t i = 0; i < r.columns.size(); ++i) {
    const std::size_t want = (i + shift) % r.columns.size();
    for (int32_t row = 0; row < hidden; ++row) {
      const double g = bf16_to_f32(got[static_cast<std::size_t>(r.columns[i]) * hidden + row]);
      const double w = r.rows[want * k27bHidden + row];
      e += (g - w) * (g - w);
      n += w * w;
    }
  }
  return std::sqrt(e / n);
}

ignis_topology no_layer_topology(int32_t hidden) {
  ignis_topology topology{};
  topology.num_layers = 0;
  topology.layer_kinds = nullptr;
  topology.hidden = static_cast<uint64_t>(hidden);
  topology.vocab = kVocab;
  topology.ffn_intermediate = 512;
  topology.gdn_num_layers = 0;
  topology.rms_norm_eps = 1e-6F;
  return topology;
}

// One item's grid in patches: 1, 5, 16, 17 and 133 merged tokens, the last over two pool pages.
struct Item {
  int32_t h, w;
  uint32_t stream;
};
constexpr Item kItems[] = {{2, 2, 31}, {2, 10, 32}, {8, 8, 33}, {2, 34, 34}, {14, 38, 35}};

// The columns checked against fp64: all of a small item's, ten across the large one's two pages.
std::vector<int32_t> checked_columns(int32_t tokens) {
  if (tokens <= 32) {
    std::vector<int32_t> all(static_cast<std::size_t>(tokens));
    for (int32_t i = 0; i < tokens; ++i) all[static_cast<std::size_t>(i)] = i;
    return all;
  }
  const int32_t page = IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS;
  return {0, 1, page / 2 - 1, page / 2, page - 2, page - 1, page, page + 1, tokens - 2, tokens - 1};
}

// Each item's embedding from one load: `hidden` wide, the tower where `host` says.
std::vector<std::vector<uint16_t>> encode_items(const Image &im, int32_t hidden, bool host, const std::vector<Call> &calls) {
  const std::string label = std::to_string(hidden) + " wide, " + (host ? "host" : "device") + " residency";
  std::vector<std::vector<uint16_t>> out(calls.size());
  const std::size_t text_bytes = static_cast<std::size_t>(kVocab) * hidden * 2;
  DeviceBytes text(text_bytes);
  MOE_CUDA(cudaMemset(text.p, 0, text_bytes));
  const Placed tower(im, host);
  std::vector<ignis_bound_tensor> tensors{
      bound_tensor("text/token_embedding", IGNIS_QTYPE_BF16_CTRL, {kVocab, hidden}, text.p, nullptr, nullptr, text_bytes),
      bound_tensor("text/final_norm", IGNIS_QTYPE_BF16_CTRL, {hidden}, text.p, nullptr, nullptr, static_cast<uint64_t>(hidden) * 2),
      bound_tensor("text/output_head", IGNIS_QTYPE_BF16_CTRL, {kVocab, hidden}, text.p, nullptr, nullptr, text_bytes)};
  for (const Entry &e : im.entries) {
    const uint8_t *at = tower.base + e.offset;
    tensors.push_back(bound_tensor(e.name.c_str(), e.qtype, e.shape, at, e.has_high ? at + e.high : nullptr,
                            e.qtype == IGNIS_QTYPE_BF16_CTRL ? nullptr : at + e.scales, e.bytes));
  }
  const ignis_topology topology = no_layer_topology(hidden);
  ignis_model_load_options options{};
  options.size = sizeof(options);
  options.vision_max_tokens = kEnvelope;
  options.vision_item_max_tokens = kEnvelope;
  options.vision_residency = host ? IGNIS_VISION_RESIDENCY_HOST : IGNIS_VISION_RESIDENCY_DEVICE;
  ignis_model *model = nullptr;
  if (ignis_model_load(tensors.data(), tensors.size(), &topology, kPrefillChunk, kMaxContext, IGNIS_KV_FORMAT_BF16,
                       &options, &model) != 0 ||
      model == nullptr) {
    check(false, "encode: " + label + ": the load: " + ignis_model_last_error());
    return out;
  }
  for (std::size_t i = 0; i < calls.size(); ++i) {
    const Call &c = calls[i];
    const std::vector<int32_t> pos = c.position_ids();
    ignis_media_encode_input in{};
    in.size = sizeof(in);
    in.grid_t = 1;
    in.grid_h = static_cast<uint32_t>(c.h);
    in.grid_w = static_cast<uint32_t>(c.w);
    in.patches = c.rows.data();
    in.position_ids = pos.data();
    in.cu_seqlens = c.cu.data();
    in.position_table_indices = c.idx.data();
    in.position_table_weights = c.wts.data();
    const std::string item = label + ", " + std::to_string(c.tokens()) + " merged tokens";
    ignis_media_embedding *embedding = nullptr;
    const int32_t rc = ignis_media_encode(model, &in, &embedding);
    if (rc != 0 || embedding == nullptr) {
      check(false, "encode: " + item + ": " + std::to_string(rc) + ": " + ignis_media_last_error());
      continue;
    }
    check(ignis_media_embedding_columns(embedding) == static_cast<uint32_t>(c.tokens()), "encode: " + item + ": its columns");
    std::vector<uint16_t> columns(static_cast<std::size_t>(c.tokens()) * hidden);
    if (ignis_media_embedding_read(embedding, columns.data(), columns.size() * 2) != 0) {
      check(false, "encode: " + item + ": read: " + ignis_media_last_error());
    } else {
      out[i] = std::move(columns);
    }
    ignis_media_embedding_release(embedding);
  }
  ignis_model_free(model);
  return out;
}

void case_encode(double bound) {
  std::printf("encode: ignis_media_encode on a no-layer load, the merger %d and %d wide, both residencies\n",
              kFlashNextHidden, k27bHidden);
  const Tower t;
  std::vector<Call> calls;
  std::vector<Reference> refs;
  for (const Item &item : kItems) {
    calls.push_back(make_call(item.h, item.w, item.stream));
    refs.push_back(reference(t, calls.back(), checked_columns(calls.back().tokens())));
  }
  for (const Family &f : kFamilies) {
    const int32_t hidden = f.hidden;
    const Image im = tower_image(t, f);
    const auto device = encode_items(im, hidden, false, calls);
    const auto host = encode_items(im, hidden, true, calls);
    for (std::size_t i = 0; i < calls.size(); ++i) {
      const std::string item = std::string(f.name) + ", " + std::to_string(hidden) + " wide, " +
                               std::to_string(calls[i].tokens()) + " merged tokens";
      if (device[i].empty() || host[i].empty()) {
        continue;
      }
      const double e = rel_l2(device[i], hidden, refs[i], 0);
      const bool large = calls[i].tokens() > IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS;
      const double e_neg = large ? rel_l2(device[i], hidden, refs[i], 1) : 0.0;
      std::printf("  %-34s rel L2 %.3e against fp64 (bound %.1e)", item.c_str(), e, bound);
      if (large) std::printf("; against the neighbours' columns %.3e", e_neg);
      std::printf("\n");
      check(e <= bound, "encode: " + item + ": the embedding against fp64");
      if (large) {
        check(e_neg > 10.0 * bound, "encode: " + item + ": the negative control (the neighbours' columns) fails");
      }
      check(device[i] == host[i], "encode: " + item + ": the host residency's embedding is the device's, bit for bit");
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string cases = "routes,encode";
  double bound = 1e-2;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--cases") cases = argv[++i];
    else if (std::string(argv[i]) == "--bound") bound = std::atof(argv[++i]);
  }
  const auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  if (want("routes")) case_routes();
  if (want("encode")) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::fprintf(stderr, "FATAL: no CUDA device (ADR 0006: a GPU test fails without one)\n");
      return EXIT_FAILURE;
    }
    case_encode(bound);
  }
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("all checks passed\n");
  return EXIT_SUCCESS;
}
