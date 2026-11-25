// Which NVML handle answers for a CUDA device on a MIG-enabled card (kernel/src/nvml_mig.h) -- OURS.
//
// NVML's spelling of a UUID from CUDA's 16 bytes, then the choice among the MIG instance slots:
// the open slot whose UUID is the device's; else the only open slot, whatever UUID the runtime
// reported (a container is given one instance); else none, so the queries read the physical card
// as on a card without MIG.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "nvml_mig.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

using ignis::nvml_mig::Slot;

int g_failed = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failed;
}

void check_format() {
  const unsigned char bytes[16] = {0x6c, 0x5d, 0x0a, 0x1f, 0x2b, 0x3c, 0x4d, 0x5e,
                                   0x00, 0xff, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab};
  char out[96];
  ignis::nvml_mig::format_uuid("MIG-", bytes, out, sizeof(out));
  check(std::strcmp(out, "MIG-6c5d0a1f-2b3c-4d5e-00ff-0123456789ab") == 0, std::string("MIG uuid: ") + out);
  ignis::nvml_mig::format_uuid("GPU-", bytes, out, sizeof(out));
  check(std::strcmp(out, "GPU-6c5d0a1f-2b3c-4d5e-00ff-0123456789ab") == 0, std::string("GPU uuid: ") + out);
  char tiny[8];
  ignis::nvml_mig::format_uuid("MIG-", bytes, tiny, sizeof(tiny));
  check(std::strlen(tiny) == sizeof(tiny) - 1 && std::strncmp(tiny, "MIG-6c5", 7) == 0, "a short buffer is truncated and terminated");
}

void check_choice() {
  using ignis::nvml_mig::choose;
  const char *mine = "MIG-11111111-2222-3333-4444-555555555555";
  const char *other = "MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
  {
    // A privileged process sees every instance: the UUID decides.
    const Slot slots[] = {{true, other}, {false, nullptr}, {true, mine}, {true, other}};
    check(choose(mine, slots, 4) == 2, "the open slot with the device's UUID");
  }
  {
    // A container sees one instance; the runtime reported the physical card's UUID.
    const Slot slots[] = {{false, nullptr}, {true, other}, {false, nullptr}};
    check(choose(mine, slots, 3) == 1, "the only open slot when no UUID matches");
    check(choose(nullptr, slots, 3) == 1, "the only open slot when the runtime gave no UUID");
    check(choose("", slots, 3) == 1, "the only open slot when the UUID is empty");
  }
  {
    // An open slot whose UUID cannot be read still counts as the only one.
    const Slot slots[] = {{true, nullptr}};
    check(choose(mine, slots, 1) == 0, "the only open slot with an unreadable UUID");
  }
  {
    const Slot slots[] = {{true, other}, {true, other}};
    check(choose(mine, slots, 2) == -1, "none among two open slots without the device's UUID");
    check(choose(nullptr, slots, 2) == -1, "none among two open slots without a UUID to match");
  }
  {
    const Slot slots[] = {{false, nullptr}, {false, nullptr}};
    check(choose(mine, slots, 2) == -1, "none when no slot is open");
    check(choose(mine, slots, 0) == -1, "none among no slots");
  }
}

}  // namespace

int main() {
  check_format();
  check_choice();
  if (g_failed != 0) {
    std::printf("%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
