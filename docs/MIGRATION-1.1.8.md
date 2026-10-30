# Migration: TensorFold 0.6.0 (patched Python) → 1.1.8 (native binary)

Status: **planned, not executed.** `scripts/config.sh:69` still pins `TF_VERSION=v0.6.0`. Nothing in the
launcher has changed. This document is the evidence base and the plan for the bump requested on
2026-10-30, and the reference for the upstream PR.

## 1. Why this is not a version-pin edit

| Fact | Evidence |
| --- | --- |
| 1.1.8 is a native binary (Zig + CUDA kernels). The Python engine is frozen at 0.6.6. | `~/Workspace/TensorFold/setup.py`: "TensorFold 1.0.0 and later install as a native binary … The Python engine stays at 0.6.6" |
| Release archives are published per platform, including `linux-aarch64` (GB10 / DGX Spark). | `~/Workspace/TensorFold/packaging/README.md:20-25` |
| 1.1.8 has no `tensorfold/` Python package. | `ls ~/Workspace/TensorFold` — no such directory; trees are `zig/`, `app/`, `tools/`, `docs/`, `packaging/` |
| This recipe's 123 patches are `patch -p0` unified diffs against **Python** `site-packages`. | `scripts/prepare.sh:169` ("unified diffs against site-packages, applied with patch -p0"), `scripts/prepare.sh:196` (apply loop) |
| Every patch targets Python engine paths. | `grep '^diff --git' patches/*.patch` → 253 hunks under `tensorfold/families/glm5_next/cuda`, 61 `tensorfold/cuda`, 31 `tensorfold/server`, 13 `tensorfold/vision`, … **0** files outside `tensorfold/*.py` / `tensorfold/cuda/kernels/*` |
| Therefore **0 of 123 patches apply** to 1.1.8. | `patch -p0 --dry-run` finds no target files |
| The container pipeline (NVIDIA PyTorch base + pip install + patches) has no 1.1.8 counterpart. | `scripts/prepare.sh:188-199` (Dockerfile, `patch -p0`, `LABEL tf.patches`), `scripts/config.sh:71-83` (`BASE_IMAGE`, `IMAGE`, `GHCR_IMAGE`, `IMAGE_TAG`, `IMAGE_DIGEST`) |

Consequence: the bump replaces the build pipeline, the launcher, and the rank-placement model. It does not
replace the model or the hardware.

## 2. What 1.1.8 gives us natively

Reference: `~/Workspace/TensorFold`, tag `v1.1.8` = commit `03d08f1ef5e385f6ecccb49d54e31b44e75ed422`
(local HEAD is `main` at `d33dcfc7`, not the tag — use the tag).

- **Native CUDA GLM-5.3-Flash, TP=2 on two GB10s**, from Mia-AiLab's EXL3/TR3 checkpoint — the same
  checkpoint family this recipe serves. `docs/recipes/glm-5.3-flash-cuda.md`.
- **Rank placement**: `TF_GLM_RANK` of `TF_GLM_WORLD` plus `NCCL_COMM_ID`; `docs/multi-rank-receipts.md`.
  The native CUDA build "serves neither `--tp` nor `--rank`" — `start.sh`'s `--tp 2 --rank N --master`
  is the frozen engine's model.
- **`--dashboard`** — "Enable the local /dashboard page and /stats endpoint": `zig/src/server/cli.zig:56`,
  handler `zig/src/server/dashboard.html` (embedded, `dashboard_test.zig:3`), same server port as the
  API, `/stats` JSON snapshot. This is the original ask from 2026-10-30 and it is native in 1.1.8.
- **`--learn` / `--learn-dir`** exist natively (`zig/src/server/cli.zig`, flag table). Not present in 0.6.0.
- Native equivalents for most of what the patch stack adds: `--drafter`, `--kv-cache bf16|fp8`,
  `--context`, `--parallel`, `--max-tokens`, `--vision`, `--vision-image-tokens`, `--vision-video-tokens`,
  `--prompt-cache-gib`, `--prompt-cache-over-cap`, `--parallel`, `--compact-*`, `--slide`, `--loop-guard`,
  `--api-key`, `--metrics-open`, `--policy key=value` (repeatable), `TF_ROCE` for two-rank gathers.

## 3. Coverage gap (measured, not estimated)

The recipe sets 53 distinct `TF_GLM_*` / `TENSORFOLD_*` names (`scripts/config.sh`, `start.sh`). Of those,
**7 exist in 1.1.8**:

```
TENSORFOLD_MEMORY_RESERVE_GIB  TENSORFOLD_NO_UPDATE_CHECK  TENSORFOLD_VISION_WORKSPACE_MIB
TF_GLM_CACHE_ENTRIES  TF_GLM_CACHE_GIB  TF_GLM_CLEAR_THINKING  TF_GLM_MTP
```

**46 are fork-only** (no occurrence anywhere in 1.1.8's `zig/src`, `docs/`, `README.md`, `RUNBOOK.md`):
`TF_GLM_DENSE`, `_KV`, `_KVB`, `_COMM`, `_COPY_DRAFTS`, `_COPY_MAX`, `_COPY_REPLY_MATCH`, `_DFLASH_POLICY`,
`_DISPLAY_KV_BACKEND`, `_DISPLAY_KV_MIB`, `_EFFORT_TAIL`, `_EXL3_DEC_ORDER`, `_EXL3_LOADS`, `_EXL3_ORDER`,
`_FILL_BUDGET_MS`, `_FILL_DRAFTS`, `_FILL_ROWS`, `_HC_SPLIT`, `_KDA_CHUNKED`, `_KEEP_PER_CHAT`,
`_KEEP_REASONING`, `_KEPT_BYTES_GIB`, `_KV_SHARD`(via `_KV`), `_L2PF`, `_MAX_QUEUED`, `_MULTI_LONE`,
`_MULTI_PREFILL`, `_MULTI_WATCHDOG_S`, `_MULTI_WINDOW`, `_PREFILL_OVERLAP`, `_PREFILL_ROWS`, `_PREFIX_WAIT`,
`_SHARED_PREFIX`, `_SPILL_QUOTA`, `_SPILL_RELEASE_MIB`, `_SPILL_TURNS`, `_STREAM_SMOOTH`,
`_STREAM_SMOOTH_MS`, `_WIDE_GRAPHS`, `_CACHE_SHARE_PCT`, and `TENSORFOLD_GLM_IMAGE_TOKENS`,
`_MAX_IMAGES`, `_REQUEST_IMAGE_TOKENS`, `_VIDEO_FRAMES`, `_VIDEO_TOKENS`, `TENSORFOLD_NUCLEUS_UNION`,
`TENSORFOLD_SSE_KEEPALIVE_S`.

Each needs a decision: map to a 1.1.8 flag/var, drop, or re-implement upstream in Zig. That table is
section 5 and is filled in during the spike, not guessed here.

## 4. Open questions the spike must answer

1. Does 1.1.8 serve the **Ablit** checkpoint this recipe pins
   (`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit`, `ABLIT=1`, thinking default off), or only the
   documented `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`? Load-time and quality differ if it must change.
2. What is the affordable window at BF16 KV on 1 GB10 with vision on? Current live figure to beat:
   `context_length 1040368`, `pool_tokens 2080768` (0.6.0 + patches, 2 ranks).
3. Which `/health` fields survive? `deploy`-side tooling reads `context_length`, `pool_tokens`,
   `pool_free_tokens`, `kept_prompts`, `kept_bytes`, `spill.*`, `kv_shard.*`
   (`tools/verify_deployment.py`, `justfile` `status`). 1.1.8 exposes `/stats` (dashboard) and `/memory`.
4. Is the 4-stream (`--parallel 4`) scheduler still the right default, and do the fork's verify-window
   knobs (`TF_GLM_MULTI_WINDOW=32`) have a 1.1.8 counterpart?
5. Do the two-rank RoCE gathers (`TF_ROCE`, `TF_ROCE_HCA`) reproduce `TF_GLM_COMM=roce`'s +6% decode, and is
   the rail layout the same (`start.sh` Rail probe, `scripts/config.sh` HCA list)?
6. Spill tier: does `--spill-gib` on 1.1.8 cover `SPILL_GIB=64` / `SPILL_TURNS=4`, or does per-conversation
   state on local NVMe need re-implementing?
7. Deployment shape: 1.1.8 is a release archive, not a container. Does the recipe keep Docker (own image
   wrapping the archive) or move to `systemd` + bare archive on each Spark? `justfile`, `DEPLOYMENT.md`,
   `stop.sh`, and the `docker logs` paths all depend on this answer.

## 5. Stages

| Stage | Deliverable | Gate |
| --- | --- | --- |
| **S0** | This document; repo ready for a PR (branch, decision log, `continue.md`, CHANGELOG entry); pin unchanged | reviewer agreement on the plan |
| **S1** | Hardware spike on spark0/spark1: 1.1.8 `linux-aarch64` archive, Ablit EXL3 at TP=2, `--dashboard`, `/stats` captured, gap table filled | Q1-Q6 answered with receipts in `logs/` |
| **S2** | Native launcher: `scripts/prepare.sh`, `start.sh`, `stop.sh`, `scripts/config.sh` gain a native path; knob mapping applied; docker path kept as legacy until S1 passes twice | two consecutive clean boots, smoke test, `tools/verify_deployment.py` green |
| **S3** | Fork-only knobs that 1.1.8 cannot express: either Zig PRs upstream or explicit drops, each with a measured cost | decisions.md entries per knob |
| **S4** | PR to `git@github.com:volfco/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git` | branch pushed, CHANGELOG/README updated, CI (`ci/patch-apply-and-pin-check`) adjusted for the native path |

S2 and S3 are deliberately separate: a launcher rewrite that silently drops 46 tuning knobs would look like
a version bump while removing the recipe's measured advantage (`scripts/config.sh` documents decode gains
per knob, e.g. prose 38.9 → 44.4 tok/s for `DENSE=q4`, +6% for RoCE).

## 6. References

- Upstream tag: local `~/Workspace/TensorFold` @ `v1.1.8` (`03d08f1e…`); release archives
  `packaging/README.md`; native GLM recipe `docs/recipes/glm-5.3-flash-cuda.md`; placement
  `docs/multi-rank-receipts.md`; dashboard/`--dashboard` `docs/api.md:101-102`, `zig/src/server/cli.zig:56`.
- This recipe: `README.md` (§ "KV pool and memory", § Performance), `CHANGELOG.md`, `DEPLOYMENT.md`,
  `patches/` (123 files), `scripts/prepare.sh`, `scripts/config.sh`, `start.sh`, `stop.sh`, `justfile`,
  `tools/verify_deployment.py`.
- Fork/pin history: `CHANGELOG.md:1-12` (`scripts/prepare.sh` pulls the image "by the digest pinned in
  `scripts/config.sh`"), `scripts/config.sh:69-83`.
