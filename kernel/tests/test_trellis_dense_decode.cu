// The dense trellis tile decode held bit for bit to exllamav3's `reconstruct` at every K the 27B's
// matrices take -- OURS (spec 27b-16gb/02 Acceptance 1, GitHub #337).
//
// No tolerance: the decoded weight is defined to equal `reconstruct`'s fp16 output exactly
// (ADR 0060), so any differing bit is a failure.
//
//   fixtures     one 256 x 128 projection per K in {2, 2.5, 3, 4, 5, 6}, encoded by exllamav3's own
//                quantizer, against its recorded `reconstruct` output; the host restatement of the
//                format is held to the same output first.
//   geometry     every shape the 27B serves (spec 02's table, fused and split, the head included)
//                at every K, over counter-hash trellis words (any bit pattern is a valid encoding),
//                against the checksum of `reconstruct` recorded for the same words.
//   view         a fused plane's column view (tile_stride) decodes to the plane's own columns.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU or fixture fails.

#include "trellis_test_common.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace trellis_test;

namespace {

const char *kName[] = {"k2", "k2p5", "k3", "k4", "k5", "k6"};
const uint32_t kK2[] = {4, 5, 6, 8, 10, 12};

std::string fixture(const std::string &name) { return std::string(IGNIS_TRELLIS_DENSE_FIXTURE_DIR) + "/" + name; }

ignis_trellis_weight weight_of(const void *trellis, const void *suh, const void *svh, uint32_t k2, uint32_t in,
                               uint32_t out, uint32_t stride = 0) {
  ignis_trellis_weight w{};
  w.trellis = trellis;
  w.suh = suh;
  w.svh = svh;
  w.k2 = k2;
  w.in = in;
  w.out = out;
  w.tile_stride = stride;
  return w;
}

std::size_t mismatches(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, std::size_t *first) {
  std::size_t n = 0;
  *first = a.size();
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (a[i] != b[i]) {
      if (n == 0) *first = i;
      ++n;
    }
  }
  return n + (a.size() != b.size() ? 1 : 0);
}

void fixtures_arm() {
  for (int k = 0; k < 6; ++k) {
    const auto fx = moe_test::read_fixture(fixture(std::string("trellis_") + kName[k] + ".bin"));
    const auto &tr = moe_test::need(fx, "trellis");
    const auto &rc = moe_test::need(fx, "reconstruct");
    check(moe_test::need(fx, "K_times_2").as<int32_t>()[0] == static_cast<int32_t>(kK2[k]),
          std::string(kName[k]) + ": K in fixture");
    const int in = static_cast<int>(rc.dims[0]);
    const int out = static_cast<int>(rc.dims[1]);
    const auto words = tr.as<uint16_t>();
    const auto oracle = rc.as<uint16_t>();
    std::size_t first = 0;
    const std::size_t host_bad = mismatches(moe_test::host_trellis_decode(words.data(), kK2[k], in, out), oracle, &first);
    check(host_bad == 0, std::string(kName[k]) + ": host restatement differs from reconstruct in " +
                             std::to_string(host_bad) + " weights (first at " + std::to_string(first) + ")");
    DeviceBytes d_words(words.size() * 2), d_scales(4096), d_w(static_cast<std::size_t>(in) * out * 2);
    upload(d_words, words);
    const ignis_trellis_weight w = weight_of(d_words.p, d_scales.p, d_scales.p, kK2[k], in, out);
    TL_RC(ignis_trellis_reconstruct(&w, d_w.p, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    const std::size_t dev_bad = mismatches(download<uint16_t>(d_w.p, static_cast<std::size_t>(in) * out), oracle, &first);
    check(dev_bad == 0, std::string(kName[k]) + ": kernel decode differs from reconstruct in " + std::to_string(dev_bad) +
                            " weights (first at " + std::to_string(first) + ")");
    std::printf("  fixture %-5s %dx%d: host %zu, kernel %zu mismatching weights\n", kName[k], in, out, host_bad, dev_bad);
  }
}

void geometry_arm() {
  const auto fx = moe_test::read_fixture(fixture("trellis_checksums.bin"));
  const char *shapes[] = {"gate_up", "gate", "down", "qkvz", "qkv", "z", "gdn_out", "qkgv", "q", "kv", "head"};
  DeviceBytes d_scales(4096);
  for (int k = 0; k < 6; ++k) {
    for (const char *shape : shapes) {
      const std::string key = std::string(shape) + "_" + kName[k];
      const auto rec = moe_test::need(fx, key).as<uint64_t>();  // stream, checksum, in, out
      const uint32_t in = static_cast<uint32_t>(rec[2]), out = static_cast<uint32_t>(rec[3]);
      const uint64_t n_words = static_cast<uint64_t>(in / 16) * (out / 16) * tile_words16(kK2[k]);
      DeviceBytes d_words(n_words * 2), d_w(static_cast<std::size_t>(in) * out * 2);
      device_hash_words(d_words.p, n_words, static_cast<uint32_t>(rec[0]));
      const ignis_trellis_weight w = weight_of(d_words.p, d_scales.p, d_scales.p, kK2[k], in, out);
      TL_RC(ignis_trellis_reconstruct(&w, d_w.p, nullptr));
      const uint64_t sum = device_checksum(d_w.p, static_cast<uint64_t>(in) * out);
      check(sum == rec[1], key + ": kernel decode checksum differs from reconstruct's");
      std::printf("  geometry %-13s %6u x %-6u checksum %016llx (reconstruct %016llx)\n", key.c_str(), in, out,
                  static_cast<unsigned long long>(sum), static_cast<unsigned long long>(rec[1]));
    }
  }
}

void view_arm() {
  // Columns [256, 640) of a 5120 x 1024 plane at K = 5, as a view, against the full decode.
  const uint32_t in = 5120, out = 1024, k2 = 10, c0 = 256, cols = 384;
  const uint64_t n_words = static_cast<uint64_t>(in / 16) * (out / 16) * tile_words16(k2);
  DeviceBytes d_words(n_words * 2), d_full(static_cast<std::size_t>(in) * out * 2), d_view(static_cast<std::size_t>(in) * cols * 2),
      d_scales(4096);
  device_hash_words(d_words.p, n_words, 4242);
  const ignis_trellis_weight full = weight_of(d_words.p, d_scales.p, d_scales.p, k2, in, out);
  TL_RC(ignis_trellis_reconstruct(&full, d_full.p, nullptr));
  const char *view_words = static_cast<const char *>(d_words.p) + static_cast<std::size_t>(c0 / 16) * tile_words16(k2) * 2;
  const ignis_trellis_weight view = weight_of(view_words, d_scales.p, d_scales.p, k2, in, cols, out / 16);
  TL_RC(ignis_trellis_reconstruct(&view, d_view.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto f = download<uint16_t>(d_full.p, static_cast<std::size_t>(in) * out);
  const auto v = download<uint16_t>(d_view.p, static_cast<std::size_t>(in) * cols);
  std::size_t bad = 0;
  for (uint32_t r = 0; r < in; ++r)
    for (uint32_t c = 0; c < cols; ++c) bad += f[static_cast<std::size_t>(r) * out + c0 + c] != v[static_cast<std::size_t>(r) * cols + c];
  check(bad == 0, "a tile_stride view decodes to its plane's columns (" + std::to_string(bad) + " differ)");
  std::printf("  view    columns [%u, %u) of %u x %u: %zu differing weights\n", c0, c0 + cols, in, out, bad);
}

void rejects_arm() {
  DeviceBytes d(4096);
  ignis_trellis_weight w = weight_of(d.p, d.p, d.p, 7, 128, 128);
  check(ignis_trellis_reconstruct(&w, d.p, nullptr) != 0, "k2 = 7 is refused");
  w = weight_of(d.p, d.p, d.p, 6, 128, 64);
  check(ignis_trellis_reconstruct(&w, d.p, nullptr) != 0, "out = 64 is refused (multiples of 128)");
  w = weight_of(static_cast<char *>(d.p) + 4, d.p, d.p, 6, 128, 128);
  check(ignis_trellis_reconstruct(&w, d.p, nullptr) != 0, "an unaligned trellis is refused");
  w = weight_of(d.p, d.p, d.p, 6, 128, 256, 8);
  check(ignis_trellis_reconstruct(&w, d.p, nullptr) != 0, "a tile_stride narrower than out / 16 is refused");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("dense trellis decode vs exllamav3 reconstruct\n");
  fixtures_arm();
  geometry_arm();
  view_arm();
  rejects_arm();
  if (moe_test::g_failed != 0) {
    std::fprintf(stderr, "test_trellis_dense_decode: %d failure(s)\n", moe_test::g_failed);
    return 1;
  }
  std::printf("test_trellis_dense_decode: OK\n");
  return 0;
}
