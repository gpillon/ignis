#pragma once
// GitHub #676 (spec #672): how a 27B verify round returns -- OURS.
//
// Earlier releases wait on the host twice in a DFlash2 verify round: for its
// accept, through a pageable copy and a stream synchronize, and after its
// commit's device work (the ring invalidation, the fold, the accepted hidden,
// the drafter's append), through a second synchronize. Now such a round
// stages its inputs from host-mapped memory with one kernel, takes its accept
// back through host-mapped memory behind a ticket the host waits on, and
// returns with its commit's device work still on the model's stream: every
// lane's state is committed from the host's own numbers, and every later leaf
// call reaches the device behind that work (kernel/src/deferred_verify.cu).
// IGNIS_DEFER_VERIFY=0, read at each capture of the decode graphs, gives every
// round earlier releases' path, which a load's rounds before its first
// capture take too.
//
// Host-only, no CUDA type: kernel/tests/test_verify_return.cpp builds it with
// no CUDA library, so it runs on a card another process holds.

// Whether a verify round stages through host-mapped memory and waits on a
// ticket for its accept. Only a DFlash2 load's live rounds do: an MTP load's
// rounds keep earlier releases' path until they are measured there (spec
// #672), a verify-only load's take their drafts from the caller, and a
// teacher-forced round (test-only) replaces the accept's verdict on the host.
inline bool ignis_verify_round_maps(bool switch_on, bool dflash2, bool teacher_forced) {
  return switch_on && dflash2 && !teacher_forced;
}

// Whether a round that maps returns with its fold in flight: only when
// nothing after the fold reads the device on the host. A lane at extent 0
// carries its anchor column's taps back to the host (GitHub #157), and a
// sampled lane whose run was cut or redirected takes its penalty counts back
// (`adjust_penalty_counts`): either synchronizes the fold, as earlier
// releases did.
inline bool ignis_verify_fold_stays_in_flight(bool maps, bool lane_at_extent_0, bool penalty_counts_back) {
  return maps && !lane_at_extent_0 && !penalty_counts_back;
}
