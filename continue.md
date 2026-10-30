# Continue

Forward-looking only. Project docs: `README.md` (engine/recipe), `DEPLOYMENT.md` (this deployment),
`docs/MIGRATION-1.1.8.md` (planned engine bump), `decisions.md` (why).

## Current state

- Served stack: TensorFold **0.6.0** + `patches/` (123 files) in a container, `scripts/config.sh`
  `TF_VERSION=v0.6.0`. Engine pin unchanged.
- Window: `CONTEXT=1048576` → served `context_length 1040368`, `pool_tokens 2080768`, BF16 KV, 4 streams,
  vision on. Verified live 2026-10-29/30.
- Ranks were stopped cleanly at 2026-10-30T04:24:45Z and are **not running** (`docker ps -a` on spark0 empty).
  Start with `just start` (spark0 `~/glm53-tensorfold`), verify with `just verify`.
- Repo is on branch `feat/tensorfold-1.1.8` with the migration plan and PR scaffolding; `main` is the
  published fork state.

## Next

1. Fill the 46-row knob table in `docs/MIGRATION-1.1.8.md` §3 (map / drop / re-implement). Until it is
   filled, S2 cannot start: the launcher rewrite would silently drop measured tuning.
2. Answer the seven open questions in §4 of that doc; Q1-Q3 are blockers, Q4-Q7 shape S2.
3. Run S1 (hardware spike) in a maintenance window, receipts into `logs/`.

## Open questions (need answers before S2)

1. Does the deployment have a maintenance window for the S1 spike (1.1.8 `linux-aarch64` archive on both
   Sparks, Ablit EXL3 at TP=2)? See `docs/MIGRATION-1.1.8.md` §4.
2. Does the Ablit checkpoint load on 1.1.8, or must the profile move to `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`?
3. For S2: does the recipe keep Docker as the unit of deployment, or move to systemd + bare archive on each
   Spark? `justfile`, `DEPLOYMENT.md`, `stop.sh`, and the `docker logs` paths in `start.sh` all depend on it.
