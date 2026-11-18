// GitHub #676 (spec #672): how a 27B verify round returns -- OURS. The two
// host functions the round calls (kernel/src/verify_return.h): whether it
// stages through host-mapped memory and waits on its ticket, and whether it
// then returns with its fold in flight. MTP loads, teacher-forced rounds, a
// lane at extent 0 and a penalty count that comes back to the host keep
// earlier releases' synchronous return.
//
// Pure host logic, built without any CUDA library (kernel/tests/CMakeLists.txt
// links nothing), so this test can create no CUDA context: it runs on a card
// another process holds.

#include "verify_return.h"

#include <cstdio>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label);
    ++failures;
  }
}

} // namespace

int main() {
  constexpr bool kOn = true;       // IGNIS_DEFER_VERIFY unset, or `1`
  constexpr bool kDflash2 = true;  // the load's drafter
  constexpr bool kForced = true;   // a teacher-forced round

  // Which rounds stage through host-mapped memory and wait on a ticket.
  expect(ignis_verify_round_maps(kOn, kDflash2, !kForced), "a DFlash2 load's live round maps");
  expect(!ignis_verify_round_maps(!kOn, kDflash2, !kForced), "IGNIS_DEFER_VERIFY=0 keeps the synchronous round");
  expect(!ignis_verify_round_maps(kOn, !kDflash2, !kForced),
         "an MTP load's round (its drafter is not DFlash2) keeps the synchronous round");
  expect(!ignis_verify_round_maps(kOn, kDflash2, kForced), "a teacher-forced round keeps the synchronous round");

  // Which of them return with the fold in flight.
  constexpr bool kExtent0 = true;  // a lane at extent 0
  constexpr bool kCounts = true;   // a penalty count comes back to the host
  expect(ignis_verify_fold_stays_in_flight(true, !kExtent0, !kCounts),
         "a mapped round with nothing read after its fold leaves it in flight");
  expect(!ignis_verify_fold_stays_in_flight(true, kExtent0, !kCounts),
         "a lane at extent 0 synchronizes the fold");
  expect(!ignis_verify_fold_stays_in_flight(true, !kExtent0, kCounts),
         "a penalty count read back synchronizes the fold");
  expect(!ignis_verify_fold_stays_in_flight(true, kExtent0, kCounts), "both synchronize the fold");
  expect(!ignis_verify_fold_stays_in_flight(false, !kExtent0, !kCounts),
         "a round that does not map (MTP, teacher-forced, the switch at 0) synchronizes the fold");

  if (failures != 0) {
    std::fprintf(stderr, "ignis_kernel_verify_return_test: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("ignis_kernel_verify_return_test: OK\n");
  return 0;
}
