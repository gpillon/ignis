// The trellis decode every Flash-Next expert op runs, held bit for bit to exllamav3's
// `reconstruct` -- OURS (spec flash-next/02 Acceptance 2, GitHub #300).
//
// There is no tolerance here: the decoded weight is defined to equal `reconstruct`'s fp16
// output exactly (ADR 0044), so any differing bit is a failure.
//
//   fixtures     one 256 x 128 projection per K in {2, 2.5, 3, 4}, encoded by exllamav3's own
//                quantizer, against its recorded `reconstruct` output. The host restatement
//                of the format (moe_fixture.h) is held to the same output first, since the
//                other arms and every other MoE test lean on it.
//   geometry     the two real expert shapes (fused gate/up 2560 x 1280, down 640 x 2560) at
//                every K, over counter-hash trellis words (any bit pattern is a valid
//                encoding), against the checksum of `reconstruct` recorded for the same words,
//                and element by element against the host restatement.
//   records      the expert-projection record sizes against layout.md's class table.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU or fixture fails.

#include "ignis_moe.h"
#include "moe_fixture.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

const char *kName[] = {"k2", "k2p5", "k3", "k4"};
const uint32_t kK2[] = {4, 5, 6, 8};

std::string fixture(const std::string &name) {
  return std::string(IGNIS_FLASH_NEXT_FIXTURE_DIR) + "/" + name;
}

std::vector<uint16_t> device_decode(const std::vector<uint16_t> &words, uint32_t k2, int in, int out) {
  DeviceBytes d_words(words.size() * 2);
  DeviceBytes d_w(static_cast<std::size_t>(in) * out * 2);
  upload(d_words, words);
  MOE_RC(ignis_moe_trellis_reconstruct(d_words.p, k2, in, out, d_w.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  return download<uint16_t>(d_w.p, static_cast<std::size_t>(in) * out);
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
  for (int k = 0; k < 4; ++k) {
    const auto fx = read_fixture(fixture(std::string("trellis_") + kName[k] + ".bin"));
    const auto &tr = need(fx, "trellis");
    const auto &rc = need(fx, "reconstruct");
    check(need(fx, "K_times_2").as<int32_t>()[0] == static_cast<int32_t>(kK2[k]), std::string(kName[k]) + ": K in fixture");
    const int in = static_cast<int>(rc.dims[0]);
    const int out = static_cast<int>(rc.dims[1]);
    const auto words = tr.as<uint16_t>();
    const auto oracle = rc.as<uint16_t>();
    std::size_t first = 0;
    const std::size_t host_bad = mismatches(host_trellis_decode(words.data(), kK2[k], in, out), oracle, &first);
    check(host_bad == 0, std::string(kName[k]) + ": host restatement differs from reconstruct in " +
                             std::to_string(host_bad) + " weights (first at " + std::to_string(first) + ")");
    const std::size_t dev_bad = mismatches(device_decode(words, kK2[k], in, out), oracle, &first);
    check(dev_bad == 0, std::string(kName[k]) + ": kernel decode differs from reconstruct in " +
                            std::to_string(dev_bad) + " weights (first at " + std::to_string(first) + ")");
    std::printf("  fixture %-5s %dx%d: host %zu, kernel %zu mismatching weights\n", kName[k], in, out, host_bad, dev_bad);
  }
}

void geometry_arm() {
  const auto fx = read_fixture(fixture("trellis_checksums.bin"));
  struct Shape {
    const char *name;
    int in, out;
  } shapes[] = {{"gate_up", 2560, 1280}, {"down", 640, 2560}};
  for (int k = 0; k < 4; ++k) {
    for (const Shape &s : shapes) {
      const std::string key = std::string(s.name) + "_" + kName[k];
      const auto rec = need(fx, key).as<uint64_t>();
      const auto words = hash_trellis_words(static_cast<uint32_t>(rec[0]), s.in, s.out, kK2[k]);
      const auto dev = device_decode(words, kK2[k], s.in, s.out);
      const uint64_t sum = checksum_u16(dev);
      check(sum == rec[1], key + ": kernel decode checksum differs from reconstruct's");
      std::size_t first = 0;
      const std::size_t bad = mismatches(dev, host_trellis_decode(words.data(), kK2[k], s.in, s.out), &first);
      check(bad == 0, key + ": kernel decode differs from the host restatement in " + std::to_string(bad) +
                          " weights (first at " + std::to_string(first) + ")");
      std::printf("  geometry %-12s checksum %016llx (reconstruct %016llx), %zu host mismatches\n", key.c_str(),
                  static_cast<unsigned long long>(sum), static_cast<unsigned long long>(rec[1]), bad);
    }
  }
}

void records_arm() {
  // layout.md §3's class table, record bytes with padding.
  const uint64_t gu[] = {827392, 1032192, 1236992, 1646592};
  const uint64_t dn[] = {417792, 520192, 622592, 827392};
  for (int k = 0; k < 4; ++k) {
    uint64_t b = 0;
    MOE_RC(ignis_moe_record_bytes(IGNIS_MOE_PROJ_GATE_UP, kK2[k], &b));
    check(b == gu[k], std::string("gate/up record bytes at ") + kName[k]);
    MOE_RC(ignis_moe_record_bytes(IGNIS_MOE_PROJ_DOWN, kK2[k], &b));
    check(b == dn[k], std::string("down record bytes at ") + kName[k]);
  }
  uint64_t b = 0;
  check(ignis_moe_record_bytes(IGNIS_MOE_PROJ_DOWN, 7, &b) != 0, "k2 = 7 is refused");
  check(ignis_moe_record_bytes(2, 4, &b) != 0, "an unknown projection is refused");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("trellis decode vs exllamav3 reconstruct\n");
  fixtures_arm();
  geometry_arm();
  records_arm();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_trellis_decode: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_trellis_decode: OK\n");
  return 0;
}
