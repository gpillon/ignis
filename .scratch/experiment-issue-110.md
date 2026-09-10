# #110 probe: skip prefill-chunk sync/output on non-final chunks

Experimental patch tried in a throwaway worktree (`worktree-experiment-104-chunk-sync`),
**not merged, not for merge**. See #110 for the measurement result (ITL p95
ratio 1.130 → 1.110 — a small, real improvement, not the dominant cause of
the gap) and the retry-safety caveat this drops.

```diff
diff --git a/crates/core/src/concrete.rs b/crates/core/src/concrete.rs
index 3dbfe2b..1df20c4 100644
--- a/crates/core/src/concrete.rs
+++ b/crates/core/src/concrete.rs
@@ -1118,6 +1118,7 @@ impl Scheduler for ConcreteScheduler {
                         .min(u32::MAX as u64)) as u32,
                     start_position: start,
                     params,
+                    is_final_chunk: take == remaining,
                 }
             })
             .collect();
diff --git a/crates/core/src/scheduler.rs b/crates/core/src/scheduler.rs
index 4c60118..dfad93a 100644
--- a/crates/core/src/scheduler.rs
+++ b/crates/core/src/scheduler.rs
@@ -39,6 +39,12 @@ pub struct PrefillJob {
     /// The request's generation parameters (carried so the backend can set
     /// up the decode state; prefill only warms the KV).
     pub params: DecodeParams,
+    /// EXPERIMENTAL (#110 probe, not for merge): true when `tokens` reaches
+    /// the request's actual prompt end, i.e. `start_position + tokens.len()
+    /// == ` the full prompt length -- not merely this job's own last token.
+    /// Lets the leaf skip the output head and its sync on a chunk the
+    /// scheduler already knows is not the request's last.
+    pub is_final_chunk: bool,
 }
 
 /// One decode job: a single lane step for a running request.
diff --git a/crates/core/src/step.rs b/crates/core/src/step.rs
index c2dafca..251ece4 100644
--- a/crates/core/src/step.rs
+++ b/crates/core/src/step.rs
@@ -48,6 +48,8 @@ mod ffi {
         pub size: u32,
         pub route: i32,
         pub compute_policy: i32,
+        /// EXPERIMENTAL (#110 probe, not for merge): see the C header.
+        pub caller_confirms_final: i32,
     }
 
     /// `enum ignis_prefill_route`.
@@ -395,6 +397,7 @@ impl PrefillRoute {
             size: std::mem::size_of::<ffi::IgnisPrefillOptions>() as u32,
             route,
             compute_policy,
+            caller_confirms_final: 1,
         }
     }
 }
@@ -484,6 +487,48 @@ pub fn prefill_program_sampled(
     Ok(())
 }
 
+/// EXPERIMENTAL (#110 probe, not for merge): [`prefill_program_sampled`],
+/// but the caller states whether `token_ids` is the request's *actual*
+/// final chunk (not just this call's own last chunk, which a single-chunk
+/// serving call always trivially is). `is_final = false` tells the leaf to
+/// skip the output head and its confirming sync entirely for this call --
+/// see the C header's `caller_confirms_final` for the retry-safety caveat
+/// this drops.
+pub fn prefill_program_sampled_experimental_chunk(
+    model: &Model,
+    pool: &SeqPool,
+    sequence: &mut Seq<'_>,
+    token_ids: &[i32],
+    start_position: u64,
+    sampling: SamplingParams,
+    is_final: bool,
+) -> Result<(), String> {
+    let params = sampling.to_ffi();
+    let options = ffi::IgnisPrefillOptions {
+        size: std::mem::size_of::<ffi::IgnisPrefillOptions>() as u32,
+        route: ffi::IGNIS_PREFILL_ROUTE_CHUNKED,
+        compute_policy: ffi::IGNIS_PREFILL_COMPUTE_POLICY_ENGINE_DEFAULT,
+        caller_confirms_final: is_final as i32,
+    };
+    let rc = unsafe {
+        ffi::ignis_program_prefill(
+            model.handle(),
+            pool.handle(),
+            sequence.handle(),
+            token_ids.as_ptr(),
+            token_ids.len() as u64,
+            start_position,
+            &params,
+            &options,
+            std::ptr::null_mut(),
+        )
+    };
+    if rc != 0 {
+        return Err(last_error());
+    }
+    Ok(())
+}
+
 /// [`prefill_program`], with the route selectable (ADR 0016, P2-02, GitHub
 /// #84). Test-only entry point: [`PrefillRoute::PerToken`] exists so the
 /// chunked route has a self-oracle (the same prompt prefilled both ways
diff --git a/crates/runtime/src/cuda_leaf.rs b/crates/runtime/src/cuda_leaf.rs
index 39bf3fe..2be82ac 100644
--- a/crates/runtime/src/cuda_leaf.rs
+++ b/crates/runtime/src/cuda_leaf.rs
@@ -292,16 +292,20 @@ impl StepLeaf for CudaLeaf {
         tokens: &[TokenId],
         start_position: u32,
         params: DecodeParams,
+        is_final_chunk: bool,
     ) -> Result<(), i32> {
         let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
-        step::prefill_program_sampled(
+        // EXPERIMENTAL (#110 probe, not for merge): the serving path routes
+        // through the chunk-aware entry point so an intermediate chunk can
+        // skip the leaf's output head and its sync.
+        step::prefill_program_sampled_experimental_chunk(
             &model.model,
             &model.pool,
             sequence,
             &token_ids,
             u64::from(start_position),
             sampling_params(params),
-            None,
+            is_final_chunk,
         )
         .map_err(|e| leaf_error("prefill", e))
     }
diff --git a/crates/runtime/src/lib.rs b/crates/runtime/src/lib.rs
index 4b71c3e..688903a 100644
--- a/crates/runtime/src/lib.rs
+++ b/crates/runtime/src/lib.rs
@@ -126,7 +126,10 @@ pub trait StepLeaf: Send + Sync + 'static {
     ) -> Result<Self::Sequence, i32>;
     /// Release a sequence allocation.
     fn release_sequence(&self, model: &Self::Model, sequence: Self::Sequence);
-    /// Warm one sequence with a prefill span.
+    /// Warm one sequence with a prefill span. `is_final_chunk` is
+    /// EXPERIMENTAL (#110 probe, not for merge): true when `tokens` reaches
+    /// the request's actual prompt end, letting the leaf skip output-head
+    /// work and its sync on a chunk known not to be the request's last.
     fn prefill(
         &self,
         model: &Self::Model,
@@ -134,6 +137,7 @@ pub trait StepLeaf: Send + Sync + 'static {
         tokens: &[TokenId],
         start_position: u32,
         params: DecodeParams,
+        is_final_chunk: bool,
     ) -> Result<(), i32>;
     /// Decode one round over a batch of warmed sequences with one parameter
     /// set per sequence. `params` is parallel to `sequences`. On an error,
@@ -261,6 +265,7 @@ impl<L: StepLeaf> Compute for RuntimeCompute<L> {
                 &job.tokens,
                 job.start_position,
                 job.params,
+                job.is_final_chunk,
             ) {
                 // Core retries a failed *batch*, not only this job. Return
                 // every batch sequence to zero state so that retry does not
diff --git a/crates/runtime/tests/cuda_leaf_gpu.rs b/crates/runtime/tests/cuda_leaf_gpu.rs
index 4dee450..423cfcf 100644
--- a/crates/runtime/tests/cuda_leaf_gpu.rs
+++ b/crates/runtime/tests/cuda_leaf_gpu.rs
@@ -73,6 +73,7 @@ fn the_cuda_leaf_prefills_and_decodes_a_real_prompt_through_the_compute_trait()
                 max_tokens: Some(MAX_GENERATED as u32),
                 ..DecodeParams::default()
             },
+            is_final_chunk: true,
         }])
         .unwrap_or_else(|e| panic!("prefill_step: {e}"));
 
diff --git a/crates/runtime/tests/runtime_compute.rs b/crates/runtime/tests/runtime_compute.rs
index c1a96d0..4babdb5 100644
--- a/crates/runtime/tests/runtime_compute.rs
+++ b/crates/runtime/tests/runtime_compute.rs
@@ -137,6 +137,7 @@ impl StepLeaf for StubLeaf {
         _tokens: &[u32],
         start_position: u32,
         params: DecodeParams,
+        _is_final_chunk: bool,
     ) -> Result<(), i32> {
         let call = {
             let mut calls = self.calls.lock().unwrap();
@@ -205,6 +206,7 @@ fn runtime_threads_each_requests_sampling_params_to_the_leaf_batch() {
                 context_tokens: 9,
                 start_position: 0,
                 params: left,
+                is_final_chunk: true,
             },
             PrefillJob {
                 request: 2,
@@ -212,6 +214,7 @@ fn runtime_threads_each_requests_sampling_params_to_the_leaf_batch() {
                 context_tokens: 9,
                 start_position: 0,
                 params: right,
+                is_final_chunk: true,
             },
         ])
         .unwrap();
@@ -263,6 +266,7 @@ fn prefill(request: u64, max_tokens: Option<u32>) -> PrefillJob {
             max_tokens,
             ..DecodeParams::default()
         },
+        is_final_chunk: true,
     }
 }
 
diff --git a/crates/server/src/runtime.rs b/crates/server/src/runtime.rs
index f9ada97..1185bd0 100644
--- a/crates/server/src/runtime.rs
+++ b/crates/server/src/runtime.rs
@@ -182,6 +182,7 @@ mod tests {
             _tokens: &[TokenId],
             _start_position: u32,
             _params: DecodeParams,
+            _is_final_chunk: bool,
         ) -> Result<(), i32> {
             Ok(())
         }
diff --git a/kernel/include/ignis_step.h b/kernel/include/ignis_step.h
index 3362dc9..de7a8f5 100644
--- a/kernel/include/ignis_step.h
+++ b/kernel/include/ignis_step.h
@@ -125,6 +125,16 @@ struct ignis_prefill_options {
   uint32_t size;           /* sizeof(struct ignis_prefill_options) */
   int32_t route;           /* enum ignis_prefill_route */
   int32_t compute_policy;  /* enum ignis_prefill_compute_policy */
+  /* EXPERIMENTAL (#110 probe, not for merge): nonzero means the caller's
+   * `num_tokens` span really is the request's last chunk, so the leaf's own
+   * "last chunk of this call" check should be trusted; zero means the
+   * caller (the serving scheduler, mid-prefill) already knows this call is
+   * NOT the request's final chunk regardless of what a single-call view
+   * would conclude, so the output head and its confirming sync are skipped
+   * outright. Omitted from a NULL options pointer's defaults only in the
+   * sense that NULL is handled before this field is ever read (treated as
+   * 1, unchanged production behavior). */
+  int32_t caller_confirms_final;
 };
 
 /* Prefill a token span for one sequence starting at `start_position`
diff --git a/kernel/src/step.cu b/kernel/src/step.cu
index 0ca6a94..0f3256d 100644
--- a/kernel/src/step.cu
+++ b/kernel/src/step.cu
@@ -512,13 +512,26 @@ int32_t run_program_chunk(ignis_model *model, ignis_seq_pool *pool, ignis_seq *s
     // layer's body above only enqueues work, and (when present) so does the
     // output head, so this confirms the entire chunk -- not one layer --
     // completed before the caller advances `seq`'s position state.
-    err = cudaStreamSynchronize(model->stream);
-    if (err != cudaSuccess) {
-      set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) +
-                " (" + std::to_string(num_tokens) + ") tokens for sequence slot " +
-                std::to_string(seq->slot) +
-                " failed: cudaStreamSynchronize: " + cudaGetErrorString(err));
-      return -1;
+    //
+    // EXPERIMENTAL (#110 probe, NOT FOR MERGE): gated on `compute_output`
+    // so an intermediate serving chunk (caller already knows it isn't the
+    // request's final chunk) enqueues and returns without waiting, same as
+    // ninfer's non-finalized prefill chunks. This deliberately drops the
+    // P2-02/#84 retry-path guarantee for those chunks: a kernel failure
+    // here is NOT detected until the next mandatory sync (the next final
+    // chunk or the next decode round), by which point the host may already
+    // have advanced position past work that never completed. Fine for a
+    // throwaway timing probe on a healthy GPU; not fine to ship -- a real
+    // fix needs a cheaper failure-detection path, not just this.
+    if (compute_output) {
+      err = cudaStreamSynchronize(model->stream);
+      if (err != cudaSuccess) {
+        set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) +
+                  " (" + std::to_string(num_tokens) + ") tokens for sequence slot " +
+                  std::to_string(seq->slot) +
+                  " failed: cudaStreamSynchronize: " + cudaGetErrorString(err));
+        return -1;
+      }
     }
 
     if (compute_output && out_logits != nullptr) {
@@ -559,12 +572,17 @@ int32_t run_program_chunk(ignis_model *model, ignis_seq_pool *pool, ignis_seq *s
 int32_t run_program_prefill_chunked(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                     const int32_t *token_ids, uint64_t num_tokens,
                                     const ignis_sampling_params &sampling, float *out_logits,
-                                    LinearPolicyMode mode) {
+                                    LinearPolicyMode mode, bool caller_confirms_final) {
   const uint64_t chunk_width = model->prefill_chunk_tokens;
   uint64_t offset = 0;
   while (offset < num_tokens) {
     const uint64_t chunk_len = std::min<uint64_t>(chunk_width, num_tokens - offset);
-    const bool is_last_chunk = (offset + chunk_len == num_tokens);
+    // EXPERIMENTAL (#110 probe): a single-chunk serving call always makes
+    // the within-call check true; `caller_confirms_final` is the scheduler
+    // telling the leaf whether that's *also* true of the whole request, so
+    // an intermediate serving chunk can skip the output head and its sync
+    // (ninfer-style) instead of paying both on every chunk.
+    const bool is_last_chunk = (offset + chunk_len == num_tokens) && caller_confirms_final;
     int32_t successor = -1;
     float *slot_logits = is_last_chunk ? out_logits : nullptr;
     if (run_program_chunk(model, pool, seq, token_ids + offset, chunk_len, offset, is_last_chunk,
@@ -657,6 +675,10 @@ extern "C" int32_t ignis_program_prefill(struct ignis_model *model,
   // override, for tests that compare the routes on identical inputs).
   int32_t route = IGNIS_PREFILL_ROUTE_CHUNKED;
   LinearPolicyMode mode = LinearPolicyMode::kEngineDefault;
+  // EXPERIMENTAL (#110 probe): NULL options (every existing caller except
+  // the new serving-only entry point) keeps meaning "trust this call's own
+  // last-chunk math," unchanged.
+  bool caller_confirms_final = true;
   if (options != nullptr) {
     if (options->size != sizeof(struct ignis_prefill_options)) {
       set_error("ignis_program_prefill: unrecognized ignis_prefill_options size " +
@@ -679,6 +701,7 @@ extern "C" int32_t ignis_program_prefill(struct ignis_model *model,
     mode = options->compute_policy == IGNIS_PREFILL_COMPUTE_POLICY_A16_ONLY
                ? LinearPolicyMode::kA16Only
                : LinearPolicyMode::kEngineDefault;
+    caller_confirms_final = options->caller_confirms_final != 0;
   }
 
   const auto began = std::chrono::steady_clock::now();
@@ -703,7 +726,7 @@ extern "C" int32_t ignis_program_prefill(struct ignis_model *model,
     }
   } else {
     rc = run_program_prefill_chunked(model, pool, seq, token_ids, num_tokens, *sampling,
-                                     out_logits, mode);
+                                     out_logits, mode, caller_confirms_final);
   }
   if (rc != 0) {
     return rc;
```
