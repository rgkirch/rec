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

**Instrumentation cleanup:** Removed the inconsistent raw pre-normalization capture and its extra global barriers from the standard diagnostic. The stage buffer now retains only post-QKV captures and the independent per-head check. Python compilation, CLI help, and `git diff --check` pass.

### 2026-10-27 — Clean standard 12-worker stage diagnostic

**Procedure:** Re-ran the standard diagnostic after cleanup at 12/8/8 and at 8/8/8. The standard mode leaves the shared-QKV code and barrier schedule unchanged; it only copies post-QKV stage arrays after the existing global barrier. The independent per-head kernel ran in both cases.

**Results:** At 8/8/8, the independent Q/K/V check and every captured full-path stage matched native exactly. At 12/8/8, the independent check again matched all 40 heads exactly. In the standard shared-QKV path, input normalization matched, but normalized Q differed in 1,408/4,096 values (heads 18, 21, and 23–31 fully), K differed in 128/512 values (all 128 in KV head 2), and V differed in 20/512 values (KV head 2). The merged attention output differed in 2,046/4,096 values; all later stages diverged.

**Interpretation:** The first incorrect resident stage is the normalized Q output. The independent packed-weight projection and normalization are exact, so the per-head math/weights are valid outside the 12-worker shared schedule. The failure is in that schedule's multi-group Q/K/V task distribution, normalization, or publication; the exact cause is unresolved. The separate single-owner diagnostic at 12 fixes Q/V but still fails K, and is not a validated alternative.

**Decision:** Keep the production M1 profile at 8/8/8. Do not report a 12-worker throughput result or treat the test-only schedule as a candidate until full parity and repeatability pass.

### 2026-10-27 — Single-owner probe source-rendering correction

**Observation:** The first `--qkv-single-owner` run failed Metal compilation before inference. The test replacement inserted `__LOGBASE__` and `__EPS__` after the normal source-rendering pass, leaving those placeholders undeclared.

**Correction:** Render those two constants from the loaded model config when constructing the test-only replacement. No parity or stage result was produced by the failed attempt.

### 2026-10-27 — Next single-owner projection variant

**Observation:** The test-only per-group `q4_local` schedule at 12/8/8 made normalized Q and V exact but left K incorrect; the independent one-head check uses `q4_local_block` and matched all heads. This leaves room for a per-group `q4_local_block` schedule to eliminate the K discrepancy.

**Next probe:** Add a second test-only schedule that assigns complete heads per group but computes each head with the independently validated `q4_local_block` path. Compare all stages at 12/8/8 before considering any runtime change.

**Implementation:** Added `--qkv-single-owner-block` alongside the earlier `q4_local` variant. This changes only the captured diagnostic shader, not `kv_teams.py` or the M1 profile. Python compilation and CLI validation are pending before its first M1 run.

### 2026-10-27 — Block-based single-owner QKV result

**Procedure:** Ran the `q4_local_block` single-owner variant at 8/8/8 and 12/8/8, with the independent one-head check.

**Results:** At 8/8/8, all stages matched native exactly. At 12/8/8, the independent per-head kernel still matched all 40 heads; the single-owner block path produced exact Q and V but K differed in 192/512 values (heads 0:96, 1:32, 2:32, 3:32). Attention and later stages diverged.

**Interpretation:** Changing the per-head inner projection from `q4_local` to `q4_local_block` does not resolve the K error. Both single-owner variants interleave projection and normalization/cache publication per head. The remaining issue may be in that sequencing or K publication rather than the projection arithmetic.

**Next probe:** Add a test-only two-phase QKV schedule: compute all assigned Q/K/V heads first, perform a global barrier, then normalize Q/K and publish K/V. This isolates projection completion from normalization/publication before changing the runtime transform.

**Implementation:** Added `--qkv-two-phase`, using per-group `q4_local_block` work followed by a global barrier before Q/K normalization and KV publication. It changes only the instrumented test kernel.

### 2026-10-27 — Two-phase QKV diagnostic result

**Procedure:** Ran the two-phase schedule at 8/8/8 and 12/8/8. It computes each assigned head with `q4_local_block`, globally synchronizes, then normalizes Q/K and publishes K/V.

**Results:** All captured stages matched native at 8/8/8. At 12/8/8, the independent per-head check remained exact, but the two-phase schedule had Q=1,024/4,096 mismatches, K=224/512, and exact V; attention and later stages diverged. This differs from the interleaved single-owner variant, which had exact Q/V but incorrect K.

**Interpretation:** A global barrier before normalization/publication does not fix 12 workers; it changes which outputs diverge. That makes guessing from post-normalization captures insufficient. The two-phase variant now provides a clean point to compare raw projections before normalization.

**Next probe:** After the projection global barrier, snapshot raw Q/K/V from group 0, synchronize again, then proceed with normalization. Compare those raw vectors directly to native projections; this separates projection work from head normalization/publication.

**Implementation:** Extended only the `--qkv-two-phase` test shader with a group-0 raw Q/K/V snapshot between global barriers and raw-stage comparisons against native projections. The buffer now allocates slots 13–15 for those raw vectors. Also extended the independent one-head kernel to return its pre-normalization projections, so the fused schedule can be compared directly against the same Q4 computation rather than only native outputs.

**Harness correction:** The first run after adding the second output assigned the returned raw and normalized arrays in reverse order. The raw stage snapshot itself matched native exactly at 8/8/8 and the full stages also passed, but the new independent-kernel comparisons from that invocation are invalid. Corrected the Python output ordering.

### 2026-10-27 — Two-phase raw projection comparison

**Procedure:** Re-ran the two-phase diagnostic with the corrected raw/normalized output ordering at 8/8/8 and 12/8/8. The independent one-head kernel now returns both pre-normalization and normalized Q/K/V; the fused diagnostic captures its raw Q/K/V after a global barrier and before normalization.

**Results:** At 8/8/8, independent raw and normalized outputs matched native exactly, the captured raw outputs matched the independent outputs exactly, and every full-path stage matched. At 12/8/8, both independent outputs still matched native exactly. The fused per-group raw result differed from the independent result in Q heads 22 and 23 (32/128 values each) and all four K heads (128/128 each); V was exact. The corresponding raw/native comparisons showed 64 Q, 512 K, and zero V unequal values. After normalization, Q had 1,024 mismatches, K 224, and V remained exact.

**Interpretation:** The mismatch exists before Q/K normalization and KV publication in this test schedule. The raw snapshot is validated by the exact 8-worker control and its explicit global barriers. Need still distinguish per-group input state from Q4 work execution.

**Next probe:** Capture the pre-normalized hidden input and normalized `n` tensor separately for each of the 12 threadgroups at layer 0. Compare every group against native to see whether groups that own the divergent heads already have different inputs.

**Implementation:** Added a test-only capture of layer-0 hidden input and normalized `n` for each resident group. It is independent of the QKV schedule variant and leaves runtime code unchanged.

**Tooling failure:** The first M1 run failed Metal compilation: adding a second kernel output exceeded MLX's Metal buffer binding limit (`buffer(31)`, valid range ends at 30). No inference result was produced. Packed per-worker slices into the existing stage-trace output, preserving a single added binding.

### 2026-10-27 — Per-worker input consistency

**Procedure:** Re-ran the two-phase raw-QKV diagnostic at 8/8/8 and 12/8/8 with per-group layer-0 hidden and normalized-input captures.

**Results:** Every worker group matched native hidden input and normalized `n` exactly at both worker counts. At 12/8/8, the independent raw/normalized per-head QKV remained exact, while the grouped raw projection mismatch remained: Q heads 22 and 23 had 32/128 differences each, all four K heads differed in all 128 values, and V was exact. At 8/8/8, all captures and model stages matched exactly.

**Interpretation:** The shared per-group projection input is not the cause. The mismatch occurs while assigning/executing Q/K projection work in the grouped schedule, before normalization; it is not yet clear whether work iteration/order or a kernel indexing/coordination detail is responsible.

**Next probe:** Assign exactly one head to each group per round, with a global barrier between rounds, and compare raw and normalized Q/K/V. This tests whether reusing a threadgroup across multiple head iterations causes the mismatch.

**Implementation:** Added the test-only `--qkv-rounds` mode. It uses the independent `q4_local_block` work for one head per group, globally synchronizes each round, then snapshots raw Q/K/V and normalizes/publishes them. No runtime source changed.

### 2026-10-27 — Synchronized-round QKV result

**Procedure:** Ran `--qkv-rounds` at 8/8/8 and 12/8/8. The 8-worker control passed all raw, normalized, per-worker input, and downstream comparisons. At 12, all per-group inputs still matched, but the raw discrepancies remained on Q heads 22/23 and all K heads; V remained exact. Normalized Q/K and downstream stages failed parity.

**Interpretation:** The mismatch is not caused merely by looping over multiple heads in one threadgroup: one head per group per globally synchronized round still fails. The independent per-head kernel remains exact. Need directly verify payload publication in the 12-group global and three-group KV barriers before narrowing the arithmetic schedule further.

**Next probe:** Add a small deterministic barrier stress test for `sync_all` at 12 groups and `kv_team_sync` with four teams of three groups; each group will publish a phase-tagged payload, synchronize, verify peer payloads, and repeat.

### 2026-10-27 — Cross-group payload visibility and coherent scratch

**Procedure:** Extended the M1 barrier stress kernel to use the current atomic-RMW polling barrier and 256 publication/verification iterations. Tested 8 and 12 resident groups, both the generation and optimized global barriers, and KV-team barriers, with payloads accessed through ordinary `device` pointers or `device coherent(device)` pointers. For each path, repeated each selected case three times. Then integrated the validated coherent-pointer transformation into the optimized M1 kernel only for experimental worker counts 12 through 36 (default 8 remains unchanged), and ran the normal native-vs-Rec sustained validator at 8, 12, 16, and 32 workers.

**Barrier results:** At 12 groups, ordinary-pointer global tests consistently missed peer payloads: each of groups 0–7 reported 1,024 mismatches and groups 8–11 reported 2,048; the first-iteration masks were `[3840] * 8 + [255] * 4`. The ordinary-pointer KV-team test reported mismatches in groups 6–8 (255, 255, and 510 respectively). With coherent pointers for both the writer and reader, all global (generation and optimized) and KV-team probes reported zero mismatches. The 8-group controls also reported zero mismatches with ordinary pointers. One-sided coherent-pointer trials varied by barrier type; production therefore qualifies the shared buffers on both producer and consumer paths rather than relying on a single-sided qualification.

**Implementation:** Added `engine/src/rec_mlx/kernels/coherent_scratch.py`. Its transform qualifies the cross-group Q/K/V, attention partial/merge, KV, router result, and expert intermediates as coherent, and adds coherent Q4/load helpers for device-resident input/output buffers. `build_decoder` applies it after the existing QAT transform only on M1 Max for worker counts 12, 16, 20, 24, 28, 32, and 36; the checked-in 8/8/8 profile is unchanged. The barrier algorithm itself was not changed in this experiment. Updated the stress test so it checks the expected ordinary-pointer failure and coherent-pointer success at 12 workers, while requiring the 8-worker control to pass.

**End-to-end results:** `tests/validate_resident.py` compared full layer traces and logits against native MLX using a one-token prompt and repeated token 150006. At 8/8/8, a 2-step control passed. At 12/8/8, 2 steps passed, then 3 repeats of 16 steps and 2 repeats of 64 steps passed. At 16/8/8, 2 steps and 3 repeats of 16 steps passed. At 32/8/8, 2 steps, 3 repeats of 16 steps, and 2 repeats of 64 steps passed. All checked steps matched exactly; these are correctness/repeatability checks, not throughput measurements.

**Interpretation:** The prior 12-worker divergence is repaired for the tested M1 worker configurations by coherent qualification of shared device scratch. The matching payload stress results and full-model traces implicate cross-threadgroup memory visibility for ordinary scratch pointers, not an inherent Q4 arithmetic limitation or proof that the barrier protocol is unusable. The precise Metal/compiler mechanism is still not independently proven. The 12-worker configuration is now a working, repeatedly validated candidate; 16 and 32 also pass the same sustained parity checks.

**Unresolved:** No throughput or GPU-saturation measurements have been taken for the coherent 12/16/32-worker kernels. Worker counts 20–36, 40+, full-context prefill, long-context KV-block boundaries, and peak memory remain untested. Keep the checked-in profile at 8 until separate performance and broader-context validation are complete.

### 2026-10-27 — 16K prefill parity and worker-scaling benchmark

**Context procedure/results:** Added `--prompt-tokens` to the sustained native validator so it can compare prefill KV/logits and decode traces at a chosen prompt length. Ran a repeated-token 16,384-token prompt with 2 decode steps at 8/8/8, 12/8/8, 16/8/8, and 32/8/8. Each run passed exact native MLX parity for the prompt and both decode steps. This extends the tested context from 8,192 to 16,384; it does not test the 65,536-token KV allocation boundary or the full 256K context.

**Benchmark procedure:** Added `engine/tests/benchmark_resident.py`. For each worker count, it uses an 8,192-token deterministic varied-ID prompt, performs 4 warmup decode steps, then times 128 fixed-token resident decode steps for 3 fresh sessions. It reports prefill and decode separately. Decode is raw resident-kernel throughput with a fixed input token and excludes token sampling and frontend overhead. The initial warmup run is discarded. This is an experimental comparison, not a controlled power/occupancy measurement or a direct comparison with llama-bench.

**Results (median of 3 runs):**

| Resident workers | Prefill tok/s at 8K | Decode tok/s at 8K |
| ---: | ---: | ---: |
| 8 | 516.19 | 23.08 |
| 12 | 519.94 | 26.17 |
| 16 | 527.20 | 29.95 |
| 32 | 527.82 | 36.29 |

All profiles kept `mlp_workers=8` and `head_workers=8`. Decode increased by 13.4% at 12 workers, 29.8% at 16, and 57.2% at 32 relative to 8 in this run. Prefill is the same native MLX path across these worker profiles; its small measured spread is run variation, not a worker-count effect.

**Test-suite note:** The full `unittest` suite passed: 53 tests, 1 skipped. Its M4/M5-specific routing test was mistakenly enabled on M1 because its device predicate accepted every recognized profile; the first run produced failures on this unsupported test platform. Narrowed the test predicate to M4 Pro/M5 Pro/M5 Max, matching its declared requirement. M1-specific barrier and kernel-routing tests pass separately. M4/M5 behavior was not tested on this M1 machine.

**Interpretation:** Decode scaling continues through 32 workers, so these measurements do not show a throughput plateau at the tested counts. This alone does not prove the GPU is unsaturated; a direct utilization/occupancy capture remains needed. The worker-count comparison is valid for the tested repeated-input kernel workload because correctness and repeatability passed first, but real prompt distributions, sampled generation, and longer contexts may change the rates.

**Next steps:** Continue with 32K and then longer context parity checks while staying below the prohibited 261,120-token prompt; measure throughput at longer validated contexts; retry a direct GPU utilization/occupancy measurement now that the 12/16/32 paths are stable. Do not change the default 8-worker profile based only on this decode microbenchmark.

### 2026-10-28 — Sampled-generation parity and M1 worker profile

**Issue found:** The 32-worker scalar decode path passed long sustained parity, but its packet sampler initially failed. The sampler adds a vocabulary argmax after the main scratch-coherence transform, so the added `statistics` accesses were not coherently qualified. After qualifying them, some multi-token packets still showed group-local token disagreement.

**Fix:** For M1 Max profiles above 8 workers, the packet argmax now has group 0 reduce the per-group maxima, publishes the selected token in the existing device-atomic control buffer, and has every group read it using zero-increment atomic RMW polling. The control barrier uses slots 0–2; the token uses slot 3. Metal rejected the release/acquire memory-order names; the token store and RMW use relaxed semantics after the global barrier. This path passed the full sampled validator, though the exact Metal/compiler visibility mechanism remains unproven. Added `--greedy-packet-steps` to `tests/validate_resident.py` for focused packet-vs-scalar checks, with optional `--capture-group-tokens` diagnostics.

**Correctness procedure/results:** Ran the focused 32-worker greedy packet validator for 520 generated tokens, three fresh sessions; all logits, layer traces, and KV states matched scalar decode. The full `validate_resident.py` matrix then passed at both 8 and 32 workers. It covers the 34-token natural-language prompt, 513- and 8,192-token contexts, greedy and filtered sampling settings, presence penalties, EOS truncation, and cancellation; the 520-token greedy case checks native MLX on its first eight generated tokens and scalar/packet equality through the full run. These results do not validate sampled generation at the 256K context limit.

**Sampled decode procedure:** The validator times 256 generated tokens after the 34-token prompt, with four repeats per mode; it discards the first timing and reports the median of the remaining three. Prefill is outside this decode timing. Rates below are packet-mode tokens/s:

| Sampling mode | 8 workers | 32 workers | Change |
| --- | ---: | ---: | ---: |
| Greedy | 31.27 | 41.79 | +33.6% |
| Temperature 0.6, unfiltered | 31.09 | 41.53 | +33.6% |
| Temperature 1, unfiltered | 31.36 | 41.95 | +33.8% |
| Temperature 0.6, top-p 0.95, top-k 20 | 30.45 | 40.53 | +33.1% |
| Temperature 1, top-p 0.95, top-k 20 | 30.45 | 40.43 | +32.8% |
| Temperature 1, top-p 0.95, top-k 20, presence penalty 1.5 | 30.42 | 40.46 | +33.0% |

The separate 8K fixed-input benchmark remains the prefill/decode measurement: 516.19 prefill tok/s at 8 workers and 527.82 at 32; prefill uses the same native MLX path, so this small difference is run variation, not a worker effect. Its raw decode rates were 23.08 and 36.29 tok/s, respectively. Do not combine these prefill and decode rates.

**Profile change:** Set the checked-in M1 Max resident worker count to 32; MLP and head workers remain 8. This is the fastest sampled-generation configuration measured here and passed the full sampling validator after the argmax publication fix. Other non-default counts 20, 24, 28, and 36 and full 256K sampled-generation correctness remain uncharacterized. Keep the 261,120-token prompt prohibited.

**GPU interpretation:** The measured decode improvement shows headroom beyond the old 8-worker profile, but does not prove GPU saturation or unsaturation. Existing Instruments GPU-state/counter samples are GPU-wide and not process-attributed; no saturation claim is justified.

**Final verification:** The full M1 `unittest` suite passed with the new default profile: 55 tests, 1 skipped. The 32-group coherent global-scratch payload stress passed three repeats. `py_compile` and `git diff --check` passed.
