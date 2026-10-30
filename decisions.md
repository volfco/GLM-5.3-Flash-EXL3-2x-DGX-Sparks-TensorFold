# Decisions

Short records of why the deployment is the way it is. Code/commit references, scope, options considered,
the decision, and the reasoning. Newest last.

---

## D1 — Engine pin stays at TensorFold 0.6.0 + `patches/`; 1.1.8 migration is staged, not a pin edit

- **Scope:** major (build + launcher + rank placement).
- **Refs:** `scripts/config.sh:69-83`, `scripts/prepare.sh:169-199`, `docs/MIGRATION-1.1.8.md`.
- **Options:** (a) change `TF_VERSION` to `v1.1.8` and keep everything else; (b) backport only `--dashboard`
  to 0.6.0 as patch `0124`; (c) migrate to the native binary in stages.
- **Decision:** (c). Pin unchanged until the native launcher passes a hardware spike (S1 in the migration
  doc).
- **Reasoning:** (a) is verified broken — 1.1.8 has no `tensorfold/` Python package, so 0 of the 123
  `patch -p0` diffs apply and `start.sh`'s `--tp/--rank/--master` placement is refused by the native CUDA
  build. (b) ships a UI today but adds a patch to a pin we intend to leave. 46 of the 53 config knobs the
  recipe sets have no 1.1.8 counterpart, so the migration is a porting project with a measured feature
  budget, not a version bump.

## D2 — Window set to the engine maximum with `CONTEXT=1048576`

- **Scope:** minor (deployment profile).
- **Refs:** `.env.spark.example` (`CONTEXT`), `scripts/config.sh:145-154`, spark0 `~/glm53-tensorfold/.env`.
- **Options:** leave the 163,840 default derived from `VISION=1` + BF16 KV + non-q4 dense; set the
  checkpoint's native 1,048,576; raise `KV_POOL_GIB` instead.
- **Decision:** `CONTEXT=1048576` on both ranks.
- **Reasoning:** the server itself reported the pool afforded 1M (`this memory admits up to 1048576`) while
  the derived default capped a 166,219-token request. Measured after the change: `context_length 1040368`
  per request (`TF_GLM_KV_SHARD` part limit), `pool_tokens 2080768`, `startup estimate 97.03 GiB within
  99.66 GiB`, no refit-down, 26.6 h uptime before a manual stop. Cost: kept prompt states fell 32 → 19
  (three-quarters of the shrunken pool, logged at start).
