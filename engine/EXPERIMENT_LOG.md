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
