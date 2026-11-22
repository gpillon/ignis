#pragma once
// ignis kernel leaf - GitHub #676 (spec #672): a DFlash2 verify round's
// staging through host-mapped memory, the hand-back of its accept and the
// drafter's append as a graph -- OURS. Leaf-internal, never across the C ABI.
//
// kernel/src/verify_return.h says which rounds take them; the verify round in
// kernel/src/step.cu calls them on the load's `VerifyStaging`
// (kernel/src/model_internal.h). On the RTX 5090 a pageable copy queued after
// a graph starts 120-180 us after the graph ends, and the next kernel another
// 110-170 us after the copy (spec #672): a kernel that reads or writes
// host-mapped memory keeps the stream on one engine, and a ticket the host
// reads in place answers sooner than a copy and a stream synchronize.

#include "model_internal.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ignis::deferred_verify {

// IGNIS_DEFER_VERIFY: true unset, empty or `1`, false `0`. Throws
// std::invalid_argument, naming the variable, on any other value.
bool read_switch();

// The round's slot: the other one than the round before took, once the
// round that used it last has run past its last kernel reading it.
cudaError_t begin(VerifyStaging &staging);

// `bytes` of `host` into the slot, for `dst` (device) at the next `flush`.
// False, with nothing staged, when the slot or its copy list is full: the
// caller copies them itself.
bool stage(VerifyStaging &staging, void *dst, const void *host, std::size_t bytes);

// Every copy staged since the last flush, as one kernel on `stream`.
cudaError_t flush(VerifyStaging &staging, cudaStream_t stream);

// One device region the host reads back.
struct Readback {
  void *host;
  const void *device;
  std::size_t bytes;
};

// `reads` into the slot by one kernel queued after everything on `stream`,
// which then writes a new ticket; waits on the host until it reads that
// ticket, then copies each read to its host destination. False, with nothing
// enqueued, when they do not fit the slot or its copy list; else true, with
// `*err` what the wait met: a fault on the stream, or a stream that finished
// without writing the ticket.
bool hand_back(VerifyStaging &staging, std::span<const Readback> reads, cudaStream_t stream, cudaError_t *err);

// The slot is the next round's to wait for: an event after everything the
// round enqueued on `stream`, whether it committed or failed.
cudaError_t end(VerifyStaging &staging, cudaStream_t stream);

// The DFlash2 drafter's append after a verify round's commit at `width` lanes
// (`ignis_dflash2_append_round`), captured as one graph: its inputs are device
// buffers and its scratch the drafter's arena from its base, so one capture
// serves every round at the width. GitHub #686: at shape `verify`, whose
// buffers it reads; the caller keeps the graph with that shape's. 0 with
// `*out_exec` set, or -1 with `*error` and nothing held. GitHub #712: and
// `*out_nodes` the graph's node count (`ignis_program_stats`), 0 where the
// capture failed or the runtime cannot say. GitHub #713: after a round laid
// out as `layout`, a uniform round of its width or a mixed round's mix.
int32_t capture_append(ignis_model *model, ignis_seq_pool *pool, const IgnisVerifyRound &verify,
                       const ignis::round_layout::Layout &layout, cudaGraphExec_t *out_exec, std::uint32_t *out_nodes,
                       std::string *error);

// The append after a round at shape `verify` laid out as `layout`
// (GitHub #712): that shape's graph at the round's width where the latest
// capture made one and the round is uniform, GitHub #713 the mix's graph where
// the round is mixed and the capture made one, eagerly otherwise. Throws on a
// failed launch, as the eager append does.
void append_round(ignis_model *model, ignis_seq_pool *pool, const IgnisVerifyRound &verify,
                  const ignis::round_layout::Layout &layout);

} // namespace ignis::deferred_verify
