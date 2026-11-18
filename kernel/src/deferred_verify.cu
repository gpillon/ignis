// ignis kernel leaf - GitHub #676 (spec #672): a DFlash2 verify round's
// staging through host-mapped memory, the hand-back of its accept and the
// drafter's append as a graph -- OURS. See kernel/src/deferred_verify.h.

#include "deferred_verify.h"

#include "dflash2_drafter.h"
#include "process_environment.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

VerifyStaging::VerifyStaging() {
  std::memset(region.host, 0, region.bytes);
  for (cudaEvent_t &event : used_by) {
    const cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      event = nullptr;
      for (cudaEvent_t made : used_by) {
        if (made != nullptr) {
          cudaEventDestroy(made);
        }
      }
      throw std::runtime_error(std::string("cudaEventCreateWithFlags(verify staging) failed: ") +
                               cudaGetErrorString(err));
    }
  }
}

VerifyStaging::~VerifyStaging() {
  for (cudaEvent_t event : used_by) {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
}

namespace ignis::deferred_verify {
namespace {

// One block: every copy, then -- with a ticket -- `value` into it. Every
// thread fences its own writes at system scope before the barrier, so a host
// that reads the value reads every byte the copies wrote.
__global__ void copy_then_ticket(VerifyStaging::Copies copies, unsigned long long *ticket,
                                 unsigned long long value) {
  for (std::uint32_t i = 0; i < copies.count; ++i) {
    const auto *src = static_cast<const unsigned char *>(copies.at[i].src);
    auto *dst = static_cast<unsigned char *>(copies.at[i].dst);
    for (std::uint32_t b = threadIdx.x; b < copies.at[i].bytes; b += blockDim.x) {
      dst[b] = src[b];
    }
  }
  if (ticket != nullptr) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      *reinterpret_cast<volatile unsigned long long *>(ticket) = value;
    }
  }
}

cudaError_t launch(const VerifyStaging::Copies &copies, unsigned long long *ticket, unsigned long long value,
                   cudaStream_t stream) {
  copy_then_ticket<<<1, 256, 0, stream>>>(copies, ticket, value);
  return cudaGetLastError();
}

// `bytes` of the slot from where the round's last reservation ended, 16-byte
// aligned: the host's view and the device's. False when they do not fit.
bool reserve(VerifyStaging &staging, std::size_t bytes, unsigned char **host, unsigned char **device) {
  const std::size_t at = (staging.used + 15) & ~static_cast<std::size_t>(15);
  if (at + bytes > VerifyStaging::kSlotBytes) {
    return false;
  }
  const std::size_t offset = static_cast<std::size_t>(staging.slot) * VerifyStaging::kSlotBytes + at;
  staging.used = at + bytes;
  *host = static_cast<unsigned char *>(staging.region.host) + offset;
  *device = static_cast<unsigned char *>(staging.region.device) + offset;
  return true;
}

// The ticket, past the two slots.
constexpr std::size_t kTicketOffset = 2 * VerifyStaging::kSlotBytes;

void spin_pause() {
#if defined(_M_X64) || defined(__x86_64__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

} // namespace

bool read_switch() {
  return ignis::named_switch("IGNIS_DEFER_VERIFY", true, {{"0", false}, {"1", true}});
}

cudaError_t begin(VerifyStaging &staging) {
  staging.slot ^= 1;
  staging.used = 0;
  staging.pending.count = 0;
  return cudaEventSynchronize(staging.used_by[static_cast<std::size_t>(staging.slot)]);
}

bool stage(VerifyStaging &staging, void *dst, const void *host, std::size_t bytes) {
  unsigned char *in_slot = nullptr;
  unsigned char *device = nullptr;
  if (staging.pending.count == VerifyStaging::kMaxCopies || !reserve(staging, bytes, &in_slot, &device)) {
    return false;
  }
  std::memcpy(in_slot, host, bytes);
  staging.pending.at[staging.pending.count++] =
      VerifyStaging::Copy{dst, device, static_cast<std::uint32_t>(bytes)};
  return true;
}

cudaError_t flush(VerifyStaging &staging, cudaStream_t stream) {
  if (staging.pending.count == 0) {
    return cudaSuccess;
  }
  const cudaError_t err = launch(staging.pending, nullptr, 0, stream);
  staging.pending.count = 0;
  return err;
}

bool hand_back(VerifyStaging &staging, std::span<const Readback> reads, cudaStream_t stream, cudaError_t *err) {
  if (reads.size() > VerifyStaging::kMaxCopies) {
    return false;
  }
  VerifyStaging::Copies back;
  unsigned char *in_slot[VerifyStaging::kMaxCopies] = {};
  const std::size_t used = staging.used;
  for (const Readback &read : reads) {
    unsigned char *device = nullptr;
    if (!reserve(staging, read.bytes, &in_slot[back.count], &device)) {
      staging.used = used;
      return false;
    }
    back.at[back.count++] = VerifyStaging::Copy{device, read.device, static_cast<std::uint32_t>(read.bytes)};
  }
  auto *const ticket_host =
      reinterpret_cast<volatile unsigned long long *>(static_cast<unsigned char *>(staging.region.host) + kTicketOffset);
  auto *const ticket_device =
      reinterpret_cast<unsigned long long *>(static_cast<unsigned char *>(staging.region.device) + kTicketOffset);
  const unsigned long long ticket = ++staging.ticket;
  *err = launch(back, ticket_device, ticket, stream);
  if (*err != cudaSuccess) {
    return true;
  }
  // The queued work goes to the device now, not at the driver's next
  // submission (Windows batches a stream's launches).
  if (cudaStreamQuery(stream) == cudaErrorNotReady) {
    (void)cudaGetLastError();
  }
  // The stream is asked about a fault about once a millisecond, not on every
  // spin; a stream that finished without the ticket is a fault of its own.
  auto asked = std::chrono::steady_clock::now();
  for (std::uint64_t spins = 1; *ticket_host != ticket; ++spins) {
    spin_pause();
    if (spins % 256 != 0) {
      continue;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - asked < std::chrono::milliseconds(1)) {
      continue;
    }
    asked = now;
    const cudaError_t state = cudaStreamQuery(stream);
    if (state == cudaErrorNotReady) {
      (void)cudaGetLastError();
      continue;
    }
    if (state != cudaSuccess) {
      *err = state;
      return true;
    }
    if (*ticket_host != ticket) {
      *err = cudaErrorUnknown;
      return true;
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  for (std::size_t i = 0; i < reads.size(); ++i) {
    std::memcpy(reads[i].host, in_slot[i], reads[i].bytes);
  }
  *err = cudaSuccess;
  return true;
}

cudaError_t end(VerifyStaging &staging, cudaStream_t stream) {
  return cudaEventRecord(staging.used_by[static_cast<std::size_t>(staging.slot)], stream);
}

int32_t capture_append(ignis_model *model, ignis_seq_pool *pool, std::uint32_t width, cudaGraphExec_t *out_exec,
                       std::string *error) {
  *out_exec = nullptr;
  cudaError_t err = cudaStreamBeginCapture(model->stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    *error = std::string("cudaStreamBeginCapture failed: ") + cudaGetErrorString(err);
    return -1;
  }
  std::string failed;
  try {
    ignis_dflash2_append_round(model, pool, width);
  } catch (const std::exception &e) {
    failed = e.what();
  }
  cudaGraph_t graph = nullptr;
  err = cudaStreamEndCapture(model->stream, &graph);
  if (err != cudaSuccess || !failed.empty()) {
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    *error = !failed.empty() ? failed : std::string("cudaStreamEndCapture failed: ") + cudaGetErrorString(err);
    (void)cudaGetLastError();
    return -1;
  }
  cudaGraphExec_t exec = nullptr;
  err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  if (err != cudaSuccess) {
    *error = std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(err);
    (void)cudaGetLastError();
    return -1;
  }
  *out_exec = exec;
  return 0;
}

void append_round(ignis_model *model, ignis_seq_pool *pool, std::uint32_t width) {
  const cudaGraphExec_t exec = model->verify->append_exec[width - 1];
  if (exec == nullptr) {
    ignis_dflash2_append_round(model, pool, width);
    return;
  }
  const cudaError_t err = cudaGraphLaunch(exec, model->stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGraphLaunch(the drafter's append) failed: ") + cudaGetErrorString(err));
  }
}

} // namespace ignis::deferred_verify
