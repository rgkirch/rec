# M1 Max Worker-Count Experiment Journal

This is the working lab notebook for Recursive Flint resident-worker experiments on the M1 Max. It records hypotheses, procedures, observations, and conclusions; it is not a benchmark summary. Entries are chronological. Preserve failed and inconclusive results, and distinguish observations from interpretations. Correct the record with a dated follow-up rather than silently rewriting prior results.

For each run, record the date, Rec commit and worktree state, exact `workers` / `mlp_workers` / `head_workers` profile, model and runtime identity, test procedure and input, repeat count, correctness/synchronization results, and phase-specific performance or memory results when measured. Record unknowns explicitly. Keep reusable test code in the existing test suite; do not create scattered run scripts or raw-log files.

## 2026-10-27 — Seed record: existing evidence and 12-worker question

This entry consolidates the experiment state available when this journal was created. It is a retrospective record; historical run timestamps and exact source revisions are included only where they were retained.

### Current 8-worker baseline

**Configuration:** M1 Max, `workers=8`, `mlp_workers=8`, `head_workers=8`; optimized resident decode; Flint INT4 checkpoint.

**Steps and observations:**

- Repeated the isolated 8-worker synchronization stress test five times. Each run exercised 256 global barriers and 512 KV-team barriers; all workers completed, with no diagnostics or payload mismatches.
- Compared normal Engine greedy generation with native MLX. Both produced `[3157, 5586]` for the tested short generation.
- Ran four warmups followed by 64 sequential decode-forward steps using a repeated input token. The run completed without a synchronization failure at 30.435 decode-forward tok/s. This is a diagnostic workload result, not a controlled benchmark or a coding-quality evaluation.

**Interpretation:** The zero-increment atomic RMW polling change is validated for these 8-worker tests. The exact Metal/compiler cause is still unproven. This does not validate longer contexts, full 256K prefill, measured peak memory, or GPU saturation.

**Code record:** The polling change and focused regression assertions are in Rec commit `155e4c5` (`Poll resident barriers with atomic read-modify-write`). The worktree was clean at journal creation.

### Historical higher-worker probes

These probes predate commit `155e4c5`; the higher counts have not been retested with the RMW polling change.

- A temporary `16/16/16` override differed from native in 1,984–2,015 of 2,048 layer-0 values across runs. Stage probes found varying O-projection mismatches. Restricting O projection to eight groups made earlier stages exact in that probe, but activation still differed in 768–2,048 of 6,144 values across runs. The exact fault was not identified.
- A temporary `32/32/32` override differed from native in 2,003–2,035 of 2,048 layer-0 values across five runs, with run-to-run variation. A stage probe with single-group O projection matched through attention and gate/up, then differed at activation and downstream. This localizes a symptom; it does not prove the underlying cause.
- The checked-in expert-team transform supports MLP widths 8, 32, and 64, not 16. The M1 profile remains `8/8/8`; the 16/32 overrides are not checked-in M1 configurations.
- Counts 9, 10, and 12 have no runtime correctness or performance results.

Historical probe commands, exact timestamps, and source revisions for these higher-worker runs were not retained in the available record.

### Next hypothesis to test

**Hypothesis:** A resident `workers=12` schedule with `mlp_workers=8` and `head_workers=8` may be viable because the KV-team transform computes `groups = workers // 4 = 3`, mapping 12 groups into four teams of three. This arithmetic is only a structural rationale; it is not evidence of correct execution or useful speed.

**Planned falsification:** Use the project-owned validation harness to compare native and Rec layer traces, logits, and KV state at 12 resident workers; repeat for determinism, then run sustained sequential decode. Measure throughput only after correctness and reliability pass. Keep the default M1 profile at `8/8/8` unless the evidence supports a separate profile change.

### 2026-10-27 — Persistent checkpoint setup

**Question:** Can the existing model mirror supply the checkpoint without using `/tmp` or creating a separate artifact directory?

**Steps:** Searched persistent home locations for the Flint INT4 weights. The existing `RecursiveSI/flint-int4` model mirror contained Git LFS pointer files; each weight pointer was only 135 bytes, and its local LFS object store was 132 KB. The five shard pointers specify about 22.6 GB of weights. The volume had 102 GB free. Initialized Git LFS locally in the existing model mirror and ran `git lfs pull` there.

**Results:** The five safetensors shards and tokenizer were materialized in `/Users/me/mirror/huggingface.co/RecursiveSI/flint-int4`; the model directory is about 22 GB and the model repository worktree remains clean. No checkpoint was copied to or read from `/tmp`, and nothing was pushed.

**Interpretation:** The persistent checkpoint prerequisite is resolved. The project’s `engine/.gitignore` already ignores `engine/.venv/`, so a project-local Python environment can be used without adding environment files to Git. Environment setup and 12-worker validation remain to be done.

### 2026-10-27 — Focused worker-parity harness

**Question:** Can a worker-count trial check native parity over sustained decode without changing the default M1 profile or leaving loose run files?

**Steps:** Extended the existing `engine/tests/validate_resident.py` harness with a test-only `--workers` override, `--sustained-steps`, and `--repeats`. The override changes only the resident `workers` field; MLP/head widths remain at the device profile values. The focused mode uses a one-token prompt and a fixed non-EOS token, comparing every Rec decode step’s full layer trace and logits with native MLX. The full validator remains the default mode, and result-file output is optional.

**Checks:** Python compilation, CLI help, the existing M1 profile routing test, the BF16 expert-reduction test, and `git diff --check` passed.

### 2026-10-27 — 12/8/8 two-step smoke test

**Question:** Does the 12-resident-worker, 8-MLP-worker, 8-head-worker profile pass even a short native parity check after the RMW barrier change?

**Steps:** Ran `engine/tests/validate_resident.py` against the persistent mirrored Flint INT4 checkpoint with `--workers 12 --sustained-steps 2 --repeats 1`. The prompt was `[1]`; the decode input was fixed token `150006`. The runtime printed the intended `12/8/8` profile and compiled/launched the kernel.

**Result:** Exact native layer-trace parity failed at repeat 1, step 1. The initial harness stopped at that assertion before checking logits and did not print per-layer differences. No sustained result or throughput measurement was obtained.

**Interpretation:** This is a real post-RMW correctness failure for the tested 12/8/8 path, not evidence that every 12-worker schedule fails. The mismatching trace row and stage are not yet known. The next harness revision will report per-row differences and logits before stopping.

### 2026-10-27 — 12/8/8 first-divergence diagnosis

**Steps:** Repeated the same 2-step smoke test with the diagnostic harness at Rec commit `9e18f51`.

**Results:** At step 1, the embedding trace row matched exactly. The first divergent row was row 1 (layer 0): 2,044 of 2,048 values differed, with maximum absolute difference 0.18579. The remaining layer rows also differed; final logits differed in 148,942 values, maximum absolute difference 19.8828. The run stopped at step 1, so no sustained-decode result exists.

**Interpretation:** The error begins within the first model layer, not in token embedding or sampling. The probe does not identify which layer-0 operation is responsible. Before attributing this solely to the 12-worker kernel, run the same harness at the validated 8/8/8 profile as a control.

### 2026-10-27 — 8/8/8 harness control

**Steps:** Ran the same two-step focused validator with `--workers 8 --sustained-steps 2 --repeats 1`.

**Results:** Both decode steps passed exact native parity for all layer traces and logits. The output reported two native layer checks and no diagnostics.

**Interpretation:** The focused harness and fixed-token/native-cache comparison pass at the known-good 8/8/8 profile. The 12/8/8 failure is therefore specific to the changed resident-worker configuration under this test; its first divergent kernel stage remains unknown.

### 2026-10-27 — Layer-0 stage diagnostic added

**Question:** Which layer-0 operation first diverges at 12/8/8?

**Steps:** Added a diagnostic mode to the existing project validator. It builds an instrumented copy of the captured decode shader, captures layer-0 input normalization, attention output/residual, post-attention normalization, dense gate/up/activation/down, and final layer output, then compares those arrays with native MLX intermediates. The instrumentation is test-only and does not alter the checked-in runtime kernel or default profile.

**Checks:** Python compilation, CLI help, the M1 profile routing test, the BF16 expert-reduction test, and `git diff --check` passed. The instrumented shader has not yet been compiled or run.

### 2026-10-27 — Stage-probe harness correction

**Observation:** The first `--stage-diagnostics` invocation stopped before compiling the instrumented kernel with `KeyError: 'output_shapes'`. In this kernel API, output names are part of the kernel specification, while output shapes and dtypes are supplied at each invocation.

**Correction:** Keep the additional output name in the instrumented kernel specification and append its shape/dtype in the invocation wrapper. No inference output was produced by the failed attempt.

### 2026-10-27 — 12-worker first-stage localization

**Procedure:** Ran the layer-0 stage diagnostic at resident `workers=12`, `mlp_workers=8`, `head_workers=8`, comparing one repeated-token decode step with native MLX.

**Results:** The input normalization matched exactly. The first mismatch was the attention output projection result: 2,043 of 2,048 values differed, with max absolute difference 0.291015625. Attention residual, post-attention normalization, dense gate/up/activation/down, and final layer output also differed, as expected downstream of that point.

**Interpretation:** The first error occurs in the attention subsystem, before layer-0 MLP. This test does not yet distinguish Q/K/V or attention merge from the output projection.

**Follow-up probe:** Extended the diagnostic to capture the merged attention vector before O projection and to compute the corresponding native Q/K/V, cache update, and attention output using the model's `Attention` implementation. Static compilation passes; this finer probe is pending.

### 2026-10-27 — 8-worker stage control and 12-worker attention-core mismatch

**Procedure:** Ran the extended stage diagnostic once at 12/8/8 and once at 8/8/8, with one repeated-token decode step in each run.

**Results:** At 8/8/8, all captured layer-0 stages—including the merged attention vector, attention output, and dense MLP intermediates—matched native MLX exactly. At 12/8/8, input normalization matched, but the merged attention vector first diverged (2,046 of 4,096 values; max absolute difference 2.703125). The attention output then differed in 2,047 of 2,048 values (max absolute difference 0.291015625).

**Interpretation:** The 8-worker control validates the stage capture and native attention reference for this path. The 12-worker fault occurs before O projection, in Q/K/V production or attention computation/merge. The precise operation remains unknown.

**Next probe:** Added captures for normalized Q, K, and V (after the global post-attention barrier) to distinguish projection/cache publication from attention partial/merge. The source compiles as Python; the expanded Metal probe is pending.

### 2026-10-27 — Q/K/V projection mismatch at 12 workers

**Procedure:** Ran the Q/K/V stage probe at 12/8/8 and then at 8/8/8 as a control.

**Results:** At 12/8/8, input normalization matched exactly, while normalized Q had 1,344/4,096 unequal values (max absolute difference 6.640625), K had 160/512 (5.65625), and V had 76/512 (1.8515625). The merged attention vector also differed (2,046/4,096; 2.703125). At 8/8/8, Q, K, V, attention core/output, residual, and every dense MLP stage matched native exactly.

**Interpretation:** The 12-worker error is present in Q/K/V production before attention computation, not just in attention merge or O projection. The 8-worker control validates the new Q/K/V captures and native reference. The shared projection's three-group KV-team mapping is the leading area to inspect; the exact failure mechanism is not established.

**Next probe:** Report Q/K/V mismatches by head to determine whether corruption follows a specific KV team or is distributed across all teams. The diagnostic now emits per-head mismatch counts; its run is pending.

### 2026-10-27 — Per-head Q/K/V mismatch pattern

**Procedure:** Repeated the 12/8/8 stage diagnostic with per-head mismatch reporting. The kernel instrumentation was unchanged from the previous 12-worker Q/K/V run; only the Python summary changed.

**Results:** This run had 1,536 unequal Q values (previous run: 1,344), 192 unequal K values (previous: 160), and 80 unequal V values (previous: 76). In this run, Q heads 0–17 matched and heads 18–31 differed; K heads 0–1 matched, head 2 differed completely, and head 3 differed partially; V heads 0–1 matched, head 2 differed partially, and head 3 differed completely.

**Interpretation:** Q/K/V corruption is run-variable, and this sample's errors cluster in later head indices. That pattern is not stable enough to identify a single KV team as the cause. The changing totals are consistent with a race or incomplete work distribution, but do not prove which mechanism.

**Next probe:** Repeat 12/8/8 again with the same instrumented shader and compare per-head patterns before changing the shared projection implementation.

### 2026-10-27 — Second 12-worker Q/K/V repeat

**Results:** A further 12/8/8 run again produced Q=1,344, K=160, and V=44 unequal values. Q heads 0–17 matched; head 18 and heads 21, 23–31 differed, with head 26 partially different. K heads 0–1 matched, head 2 differed completely, and head 3 partially; V heads 0–1 matched, with partial mismatches in head 2 and head 3. Attention core remained incorrect (2,046/4,096 values).

**Interpretation:** Across the two per-head runs, corruption remains concentrated in later Q/K/V heads but individual values/head counts vary. This is evidence of nondeterministic corruption in the shared projection path, not a stable head-index mapping error. It still does not distinguish raw matrix projection from Q/K normalization or KV publication.

**Next probe:** Capture the Q/K/V projection outputs before head normalization, then compare raw and normalized values separately. The test-only shader instrumentation and native raw-projection references are implemented; Python compilation passes, and the M1 run is pending.

### 2026-10-27 — Raw-projection capture consistency failure

**Procedure:** Ran the raw Q/K/V stage probe at 12/8/8.

**Results:** The recorded raw outputs differed from native by 32/4,096 Q values (max 0.4921875), 76/512 K values (0.7890625), and 40/512 V values (1.390625). The later post-QKV outputs differed by 1,408/4,096 Q, 128/512 K, and zero V values. The recorded raw V also differed from the later V buffer capture in 40 entries, although V is not normalized or otherwise transformed between those captures and the later V capture exactly matched native.

**Interpretation:** The raw capture is not trustworthy yet. The Q/K copy runs before `head_norm`, which executes in SIMD group 0, but the diagnostic had no threadgroup barrier between copying across SIMD groups and beginning normalization. This is a diagnostic race. It does not explain the raw-V inconsistency, which remains unresolved.

**Correction:** Added a threadgroup device barrier after the raw Q/K/V copy and a same-run check that raw V equals the later V-buffer capture. Do not interpret raw-stage mismatch counts until this consistency check passes.

**Next probe:** Run the corrected instrumentation at 8/8/8 and 12/8/8; first verify raw-V capture consistency and the 8-worker native control.

### 2026-10-27 — Corrected raw-capture controls and unresolved 12-worker inconsistency

**Procedure:** Added the pre-normalization threadgroup barrier, then ran stage diagnostics at 8/8/8 and 12/8/8.

**Results:** At 8/8/8, raw V equaled the later V-buffer capture, and all raw/normalized Q/K/V and downstream stages matched native exactly. At 12/8/8, the same-run raw-V check still failed (44 unequal entries), even though the later V-buffer stage exactly matched native. Raw Q/K/V capture values also differed; the previous missing-barrier issue is fixed, but the 12-worker raw-V contradiction remains.

**Interpretation:** The 12-worker production-path failure is still established by the post-QKV Q/K stages and downstream mismatch, but the raw projection capture is not yet validated at 12. Do not use it to distinguish raw projection from normalization. The capture tool's indexing or visibility under cross-group writes may be faulty; that is not yet understood.

**Next probe:** Add per-head and sample-level details to the raw-V-versus-final-V invariant to determine whether the capture is displaced, partially written, or otherwise inconsistent before using raw-stage results.

### 2026-10-27 — Raw-V capture points to missing diagnostic writes

**Results:** With the same 12/8/8 instrumented kernel, raw V differed from the later V buffer in 96 entries: 32 in KV head 2 and 64 in head 3. The first missing capture values were zeros while the later V buffer held nonzero values. The later V buffer still matched native exactly. The 8/8/8 raw-V invariant remained exact.

**Interpretation:** The per-head raw capture itself fails to write some values in the 12-worker case; it cannot be treated as evidence that Rec's V projection is wrong. The stage data makes a concurrency/indexing flaw in the diagnostic visible.

**Correction:** Replaced concurrent per-head stores with a group-0 snapshot of Q/K/V bracketed by global barriers after QKV projection and before normalization. This is instrumentation only and adds barriers to the diagnostic kernel; it does not modify the runtime kernel. The raw-V invariant remains in the report.

**Next probe:** Run the corrected single-group snapshot at 8/8/8 and 12/8/8. Confirm raw V equals its later capture and the full 8-worker stage control remains exact before interpreting 12-worker raw Q/K.

### 2026-10-27 — Single-group raw snapshot still inconsistent at 12 workers

**Procedure:** Replaced per-head raw writes with group-0 copies bracketed by two added global barriers. Ran the instrumented diagnostic at 8/8/8 and 12/8/8.

**Results:** At 8/8/8, the raw-V consistency check and all Q/K/V and downstream native comparisons passed exactly. At 12/8/8, raw V differed from the later V capture in 168 entries (40 in head 2, all 128 in head 3); raw Q/K/V and the final normalized Q/K/V captures differed from native. The final normalized captures had Q=1,408/4,096, K=128/512, and V=4/512 mismatches. The first normalized-stage mismatch was Q.

**Interpretation:** Even a single-group raw snapshot is unreliable in this 12-worker diagnostic, so its raw Q/K/V figures remain unusable. The inserted global barriers also change scheduling. The native mismatch in normalized Q/K remains, but this instrumentation does not identify whether the fault is in Q4 task distribution, head normalization, or KV publication.

**Next probe:** Build an independent one-threadgroup-per-head Q/K/V projection check using the packed layer-0 weights and native input normalization. This bypasses the 12-worker shared projection queue and head partitioning, and can determine whether the packed matrix math/head normalization are sound outside that path.

**Implementation:** Added a test-only kernel that assigns one threadgroup to each of the 40 Q/K/V heads, uses `q4_local_block` plus the corresponding Q/K normalization, and compares the packed-weight result to native MLX. Python compilation passes; the Metal kernel has not yet been compiled or run.

### 2026-10-27 — Independent-projection harness input correction

**Observation:** The first run stopped before the independent Metal kernel was invoked with `ValueError: Invalid type list received in array initialization`. The helper passed `engine.metadata`, which is the Python metadata list, rather than the per-session MLX `uint32` metadata array.

**Correction:** Pass `native_session._state["metadata"]` to the kernel. No independent-projection result was produced by the failed attempt.

### 2026-10-27 — Independent per-head Q/K/V check

**Procedure:** Ran the independent one-threadgroup-per-head projection kernel at 8/8/8 and 12/8/8, using the packed layer-0 matrices and native input normalization.

**Results:** All 40 independent Q/K/V heads matched native exactly at both worker settings. At 8/8/8, the full instrumented layer-0 stages also matched native exactly. At 12/8/8, the full resident path still had normalized Q=1,408/4,096 and K=128/512 mismatches; V differed in 4/512, and downstream attention/MLP stages diverged. The raw snapshot invariant remained broken, so raw-stage counts are not used.

**Interpretation:** The packed weights, per-head Q4 projection arithmetic, and per-head normalization can match native outside the resident shared-projection schedule. The 12-worker failure is in that schedule's coordination/partitioning or publication path, not a general inability to compute those heads. This does not identify the exact shared-path defect.

**Next probe:** Run a test-only alternate QKV schedule in which each resident group owns complete heads using `q4_local`, followed by a global barrier. This will test whether avoiding the shared ticket queue restores 12/8/8 parity without changing production code.

**Implementation:** Added `--qkv-single-owner` to the stage diagnostic. It replaces only the test kernel's shared QKV queue with per-group full-head work and a global barrier; the checked-in runtime source/profile remain unchanged. Static Python compilation and CLI checks pass; Metal validation is pending.

### 2026-10-27 — Single-owner QKV diagnostic result

**Procedure:** Ran the test-only single-owner QKV schedule at 12/8/8, then at 8/8/8 as a control. The independent one-threadgroup-per-head check ran alongside both.

**Results:** The independent projection check matched all 40 Q/K/V heads exactly in both runs. The single-owner full path matched all stages at 8/8/8. At 12/8/8, Q and V matched exactly, but K had 192/512 unequal values (heads 0:32, 1:0, 2:64, 3:96); attention and later stages diverged.

**Interpretation:** Assigning whole heads to resident groups restores Q and V in this diagnostic but does not validate the 12-worker schedule; the K path remains incorrect. This variant adds a global QKV barrier and is test-only. No runtime schedule or default profile has changed.

**Instrumentation cleanup:** The raw pre-normalization capture is known to be inconsistent at 12 workers and adds global barriers to the standard diagnostic. Remove that capture and those extra barriers; retain only the post-QKV stage captures and independent per-head check. Then rerun the standard 12-worker stage diagnostic against the unmodified shared-QKV schedule.

### 2026-10-27 — Single-owner probe source-rendering correction

**Observation:** The first `--qkv-single-owner` run failed Metal compilation before inference. The test replacement inserted `__LOGBASE__` and `__EPS__` after the normal source-rendering pass, leaving those placeholders undeclared.

**Correction:** Render those two constants from the loaded model config when constructing the test-only replacement. No parity or stage result was produced by the failed attempt.
