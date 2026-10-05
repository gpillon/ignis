// The Flash-Next QSA sparse attention (spec flash-next/04, GitHub #302, slice S3) at real
// geometry over BF16 KV -- OURS (ADR 0043): 24 query heads over 2 KV heads of 256, every row
// attending to the tokens its selection lists (up to 2051), against fp64 attention masked to
// those tokens (flash_next_sparse_test_common.h states the tolerance).
//
//   prefill  one lane, 300 rows at positions 5000..5299, each with its own 512 blocks + tail
//            (one split per row), and a call straddling the dense threshold (rows <= 2050 list
//            every visible token);
//   decode   three lanes of one token (a long sparse row, a short sparse row, a dense row): the
//            lists spread over 65 splits and merged; also captured in a CUDA graph and replayed;
//   index    the same decode rows read from a [rows][width][kv_heads][256] scratch by list index
//            (the hq-e8-2b decode route's source) must equal the paged run bit for bit;
//   refusals a dense selection, a GQA group other than 12.
//
// K, V and q are counter-hash BF16 values in [-1, 1) (q in [-3, 3) for peaked softmaxes); pages
// are permuted so the gather goes through the block table.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next_sparse_test_common.h"

using namespace sparse_test;

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  const fn::Geometry g = geometry();
  std::printf("Flash-Next QSA sparse attention (BF16 KV): 24 q / 2 KV heads of 256, lists of up to 2051 tokens\n");
  const Pages pg = make_pages();
  const RowFn exact = exact_rows(pg);
  std::vector<uint16_t> q(static_cast<size_t>(kMaxRows) * kQHeads * kHd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = f32_to_bf16(hash_uniform(kQ, i, 3.0F));

  DeviceBytes dk(pg.k.size() * 2), dv(pg.v.size() * 2), dtables(pg.tables.size() * 4);
  upload(dk, pg.k);
  upload(dv, pg.v);
  upload(dtables, pg.tables);
  Device d(std::max<size_t>(sp::partial_bytes(g, 3), 256));
  upload(d.q, q);
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  sp::KvSource paged;
  paged.k = dk.as<__nv_bfloat16>();
  paged.v = dv.as<__nv_bfloat16>();
  paged.block_tables = dtables.as<int32_t>();
  paged.logical_pages = kLogicalPages;
  paged.kv_heads = kKvHeads;
  auto attend_on = [&](const sp::KvSource &kv) -> Enqueue {
    return [&, kv](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
      SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
    };
  };

  check(sp::splits_for(g, 300) == 1, "a 300-row call must not split");
  check(sp::splits_for(g, 3) == 65, "a 3-row call must spread each list over 65 splits");

  // Prefill-shaped: one lane, its own list per row.
  for (int first : {5000, 1900}) {
    Call c;
    c.slots = {0};
    c.positions = {first};
    c.tokens = kMaxRows;
    for (int r = 0; r < kMaxRows; ++r) c.lists.push_back(make_list(first + r, 0));
    const auto got = run(d, c, attend_on(paged), stream);
    check_call("prefill rows " + std::to_string(first) + "..", exact, q, c, got, 7);
  }

  // Decode-shaped: three lanes, one token each, split over 65 CTAs per (row, KV head).
  Call dec;
  dec.slots = {0, 1, 2};
  dec.positions = {5999, 2100, 1000};
  dec.tokens = 1;
  for (int l = 0; l < 3; ++l) dec.lists.push_back(make_list(dec.positions[l], static_cast<uint32_t>(l + 1)));
  const auto eager = run(d, dec, attend_on(paged), stream);
  check_call("decode 3 lanes, 65 splits", exact, q, dec, eager, 1);
  check(run(d, dec, attend_on(paged), stream, true) == eager, "decode: the graph replay differs from eager");

  // The by-index source: the same rows' listed K/V copied into [rows][width][kv_heads][256].
  {
    const size_t n = static_cast<size_t>(3) * kWidth * kKvHeads * kHd;
    std::vector<uint16_t> sk(n, 0), sv(n, 0);
    for (int r = 0; r < 3; ++r) {
      for (size_t i = 0; i < dec.lists[r].size(); ++i) {
        for (int h = 0; h < kKvHeads; ++h) {
          const size_t src = pg.row_at(dec.slots[r], dec.lists[r][i], h);
          const size_t dst = ((static_cast<size_t>(r) * kWidth + i) * kKvHeads + h) * kHd;
          std::copy(&pg.k[src], &pg.k[src] + kHd, &sk[dst]);
          std::copy(&pg.v[src], &pg.v[src] + kHd, &sv[dst]);
        }
      }
    }
    DeviceBytes scratch_k(n * 2), scratch_v(n * 2);
    upload(scratch_k, sk);
    upload(scratch_v, sv);
    sp::KvSource by_index;
    by_index.k = scratch_k.as<__nv_bfloat16>();
    by_index.v = scratch_v.as<__nv_bfloat16>();
    by_index.kv_heads = kKvHeads;
    by_index.mode = sp::KvSource::Mode::ByIndex;
    check(run(d, dec, attend_on(by_index), stream) == eager, "by-index source differs from the paged run");
  }

  // Refusals.
  {
    fn::Selection dense;
    dense.dense = true;
    fn::Batch b;
    b.lanes = 1;
    b.tokens = 1;
    check(sp::attend(g, paged, b, d.q.as<__nv_bfloat16>(), dense, d.out.as<__nv_bfloat16>(), nullptr, stream) != nullptr,
          "a dense selection must be refused");
    fn::Geometry wrong = g;
    wrong.q_heads = 16;
    check(sp::check_geometry(wrong) != nullptr, "a GQA group other than 12 must be refused");
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "test_flash_next_sparse_attention: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_flash_next_sparse_attention: OK\n");
  return 0;
}
