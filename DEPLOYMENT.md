# Spark deployment

From this checkout, `just setup` syncs to `~/glm53-tensorfold` on spark0 and creates `.env` from `.env.spark.example` if absent. `just start` runs the upstream TensorFold launcher on spark0; it prepares the image and weights and starts both ranks. `just restart`, `just stop`, `just status`, and `just logs` operate the same installation. Set `GLM53_HEAD`, `GLM53_WORKER`, `GLM53_REMOTE_DIR`, or `GLM53_PORT` to change the orchestration targets; update the remote `.env` separately if the worker or API port changes. Sync preserves `.env`, `scripts/local.sh`, and logs. Inspect an existing `.env` and `scripts/local.sh`: local.sh takes precedence, and exported environment variables take precedence over both.

The profile uses spark0's verified Tailscale address `100.95.49.21`, spark1's private address `10.50.0.11`, and the live shared `/nfs1/hub` model cache. Spark1's management commands run through spark0. The head downloads the pinned Mia-AiLab TensorFold Ablit checkpoint and DFlash2 into `/nfs1`; spark1 reads the same export through the dedicated Docker NFS volume `glm53-tensorfold-hf` (`10.50.0.1:/nvme1`). The spill directory is local on each node, with a 64 GiB limit per node; retain the upstream minimum free-space safeguard. Accept the gated model's terms and set a Hugging Face read token on spark0 (for example in its private `scripts/local.sh`) before starting. DFlash2 has a non-commercial license; see the upstream README.

The profile selects FP8 dense weights, BF16 KV, KV sharding, DFlash2, and four concurrent requests. TensorFold chooses its BF16 context window and memory reserve based on these settings and available memory; it can reduce the window at startup. The older checkout's measurements guide the profile through the corresponding TensorFold controls; engine-specific flags and measured capacities are not copied verbatim. Validate capacity, memory headroom, and performance on the actual pair before treating these settings as optimal. `ABLIT=1` selects the specified Mia-AiLab checkpoint, not the different MikeRoz checkpoint used by the old vLLM preset.


## Cache and DFlash measurement review

| Evidence in `../spark-glm53` | Transfer to this deployment |
| --- | --- |
| `docs/LIVE-TUNING.md` §3.2: k=5 lost 6.4% at C1 and 3.5–5.2% loaded versus k=7, except prose. README reports adaptive 2/4/7 verification improving prose and code. | `DRAFT_POLICY=fnc7:0.3` retains a seven-draft ceiling with TensorFold's sampling-aware stop rule. `TF_GLM_MULTI_WINDOW=32` allows four full eight-row verification windows. Exact optimum requires a same-workload A/B on this checkpoint. |
| README: draft TP2 was roughly neutral alone and improved loaded structured decode (65.1 vs 61.7 tok/s in the cited lab comparison). | Keep the two-rank TensorFold DFlash2 path; the vLLM `DFLASH_DRAFT_TP` flag has no equivalent in this recipe. Verify acceptance and loaded throughput on both ranks. |
| `docs/apc-retention-qualification.md`: explicit sparse retention lost 111,104 cached tokens for a late edit/branch, with TTFT increasing 7.65x/29.81x. | Keep shared prefixes, 32 kept states, and `TF_GLM_MULTI_LONE=0` so lone requests do not evict histories. Preserve four spill turns for rewinds. Avoid aggressive per-chat and byte caps that drop states without spilling. |
| `decisions.md`: per-request cache details were absent despite ~95% engine cache hits. | Verify TensorFold health/cache counters and repeated-turn timing together; absent client cache telemetry is not evidence of a miss. |
| README: 14 GiB vLLM KV cap left ~5 GiB free; 15 GiB left only 0.8–2.2 GiB. `decisions.md` records a 1.374 GiB graph reservation versus 0.33 GiB measured allocation. | Explicit TensorFold `KV_POOL_GIB=12.5`, `MEMORY_RESERVE_GIB=14.5`, with BF16 sharding and local spill. Measure free memory and swap deltas during cold prefill and four-stream decode; do not transfer vLLM graph-estimator arithmetic. |
| `docs/improve-prefill.md`: 2048-token chunks improved cold prefill but coarsened decode interleaving; later fair scheduling fit a fixed cost plus a token cost. | Keep TensorFold's measured scheduling defaults and validate mixed prefill/decode pauses. A vLLM batch-token budget is not the same operation as a TensorFold prompt chunk. |

Live inspection on 2026-10-27 also confirmed that the previous `.env` used DFlash k=5, a batch-dependent 5/5/3 schedule, 131072 context and FP8 KV. This differs from its documented k=7 profile; comparisons must record the resolved runtime configuration.

The first live boot required more than 20 minutes of CUDA extension compilation and engine warmup. The deployment profile allows up to 3600 seconds for startup; later starts retain the compiled extension cache. The existing daemon places containers in `vllm.slice` on CPUs 10–19, with no CPU quota or throttling observed.


## First live verification

The first functional run passed drafted/undrafted arithmetic, a 21,350-token cold prompt, a follow-up with 21,312 cached tokens out of 21,370, and four concurrent 128-token completions. Cold/continued wall times were 51.76/1.58 seconds; the concurrent requests took 13.45–16.51 seconds. Aggregate draft acceptance across this small mixed test was 78%. These are functional checks, not a representative throughput or draft-policy A/B benchmark. The live pool held 1,818,624 tokens versus 966,656 unsharded at the same memory budget. The spill tier wrote one 0.33 GiB state without errors. Raw results are in `logs/live-verification-first-boot.json` on spark0 and in the local checkout.


The controlled restart loaded in 546.4 seconds and found the persisted spill state. The repeated long request restored 21,312 tokens from disk, answered correctly in 1.37 seconds, and reported one restore with zero restore failures. The continuation took 1.12 seconds, and all four concurrent requests passed again. Results are in `logs/live-verification-after-restart.json`.

Memory sampling every ten seconds through startup, both test runs, and restart recorded minimum MemAvailable of 9.04 GiB on rank 0 and 9.19 GiB on rank 1. Maximum sampled increases in swap occupancy were 5.25 MiB and 54.79 MiB respectively. These tests do not qualify maximum-length concurrent requests or sustained production traffic.

After verification, ten old active/backup environment files and generated launch scripts in spark0's `~/spark-glm53` and `~/glm53-kit` were archived at `~/glm53-tensorfold/logs/retired-vllm-configuration.tar.gz` (mode 0600), then removed. The `glm53-flash-mikeroz:dt-carry20` image was removed on both Sparks, and `vllm/vllm-openai:glm53-flash-arm64-cu130` was removed on spark0. Source checkouts, shared model weights, and the separate `vllm/vllm-openai:v0.32.0` image remain. Cleanup details are in `logs/cleanup.json`; post-cleanup health is in `logs/health-after-cleanup.json`. Both TensorFold ranks remain running. The previous `smaug-base` and `nemo35` containers remain stopped as authorized.

The deployed API port is 8000 (`http://100.95.49.21:8000/v1`). The deployment profile, orchestration recipes, and live verification harness use this port.

Port-8000 redeployment passed external health access and all eight inference/cache/concurrency checks. Port 8888 is no longer listening. Evidence: `logs/live-verification-port-8000.json`.
