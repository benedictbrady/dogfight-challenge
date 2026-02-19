# Modal Training Runs Log

> **Keep this log updated** after every training run launch, completion, or failure.
> Last updated: 2026-02-19

## Summary

| # | Name | Type | Updates | Status | Key Result |
|---|------|------|---------|--------|------------|
| 1 | baseline-v1 (local) | curriculum | 300 | COMPLETED | 19% win rate — gamma/reward bugs |
| 2 | local-curriculum | curriculum | 600 | COMPLETED | 99-100% vs all — the breakthrough |
| 3 | exp_smoke-test_* | smoke | 3 | COMPLETED | Pipeline verification |
| 4 | exp_smoke_* | smoke | 3 | COMPLETED | Pipeline verification |
| 5 | exp_production_1771462146 | production | 750 | UNKNOWN | Early Modal production attempt |
| 6 | exp_production_1771462548 | production | 750 | UNKNOWN | Early Modal production attempt |
| 7 | mean-saber-production | production | 750 | DIED ~50 | Only 1 checkpoint saved |
| 8 | sharp-talon-production | production | 750 | DIED ~470 | No --detach; local kill = remote kill |
| 9 | mean-knight-production | production | 750 | COMPLETED | 100% vs all (4% vs do_nothing) |
| 10 | vivid-cobra-selfplay | selfplay v1 | 2000 | DIED ~1 | Barely started |
| 11 | grim-talon-selfplay | selfplay v1 | 2000 | DIED ~11 | No checkpoints |
| 12 | grim-knight-selfplay | selfplay v1 | 2000 | DIED ~20 | No checkpoints |
| 13 | odd-lancer-selfplay | selfplay v1 | 2000 | DIED ~24 | No checkpoints |
| 14 | keen-knight-selfplay | selfplay v1 | 2000 | DIED early | Only config saved |
| 15 | grim-falcon-selfplay | selfplay v1 | 2000 | DIED ~54 | 1 checkpoint (update 50) |
| 16 | vivid-icarus-selfplay | selfplay v1 | 2000 | DIED ~1200 | Longest v1 run; ELO bug (all=1000) |
| 17 | loud-ghost-selfplay_v2 | selfplay v2 | 2000 | DIED ~381 | ELO working! Peak 1529 |
| 18 | tight-saber-selfplay_v2 | selfplay v2 | 2000 | UNKNOWN | Could not download (file conflict) |

---

## Phase 1: Local Training

### baseline-v1 (Experiment #1)
- **Date**: 2025-02-18
- **Config**: 16 envs, 10800 n_steps, action_repeat=1, 300 updates, gamma=0.99, ent_coef=0.003
- **Curriculum**: do_nothing+dogfighter(0-100) -> dogfighter+chaser(100-250) -> chaser+ace(250+)
- **Status**: COMPLETED
- **Result**: 19% win rate, -0.34 return
- **Postmortem**: Multiple bugs — gamma too low (0.99 with 10800-step episodes made terminal rewards invisible), no action_repeat, entropy bonus caused std explosion, sparse rewards only

### local-curriculum (the breakthrough run)
- **Date**: 2025-02-18
- **Config**: 32 envs, 2048 n_steps, action_repeat=10, 600 updates, gamma=0.999, ent_coef=0.0
- **Model**: 256 hidden, 0 residual blocks
- **Curriculum**: do_nothing(0-50) -> +dogfighter(50-150) -> +chaser(150-300) -> +ace(300-450) -> +brawler(450+, log_std reset)
- **Checkpoints**: `training/checkpoints/` — step_3276800 through final.pt, pre_brawler.pt
- **Status**: COMPLETED
- **Result**: 99-100% vs dogfighter/chaser/ace/brawler, ~70% vs do_nothing (timeout draws)

---

## Phase 2: Modal Curriculum Runs

### exp_smoke-test_1771454283 / exp_smoke_1771461124
- **Type**: Smoke tests (3 updates, 4 envs, 64 steps)
- **Status**: COMPLETED
- **Purpose**: Verified Modal pipeline works end-to-end

### exp_production_1771462146 / exp_production_1771462548
- **Type**: Production (128 envs, 2048 steps, 750 updates)
- **Status**: UNKNOWN (early runs, predating mnemonic names)
- **Notes**: Among the first Modal production attempts

### mean-saber-production
- **Config**: 128 envs, 2048 steps, 750 updates, 256h/0b
- **Status**: DIED at ~update 50
- **Artifacts**: 1 checkpoint (step_13107200). No eval.
- **Postmortem**: Likely killed early or crashed. Only saved 1 checkpoint.

### sharp-talon-production
- **Config**: 128 envs, 2048 steps, 750 updates, 256h/0b
- **Status**: DIED at ~update 470/750
- **Artifacts**: Checkpoints up to step_117964800 (~450 updates). No final.pt, no eval.
- **Postmortem**: Ran without `--detach`. Killing the local terminal killed the remote container. Led to the critical memory note: **ALWAYS use `modal run --detach`**.

### mean-knight-production
- **Config**: 128 envs, 2048 steps, 750 updates, 256h/0b
- **Status**: COMPLETED
- **Artifacts**: Checkpoints up to step_196608000 + final.pt + eval.txt
- **Eval Results**:
  | Opponent | Wins | Draws | Losses | Win% |
  |----------|------|-------|--------|------|
  | do_nothing | 2 | 48 | 0 | 4% |
  | dogfighter | 50 | 0 | 0 | 100% |
  | chaser | 50 | 0 | 0 | 100% |
  | ace | 50 | 0 | 0 | 100% |
  | brawler | 50 | 0 | 0 | 100% |
- **Notes**: Best curriculum run on Modal. The do_nothing result is expected — do_nothing falls from sky and is hard to hit, leading to timeout draws.

---

## Phase 3: Self-Play v1 (PFSP, ent_coef=0)

All v1 runs share the same config:
- **Model**: 384 hidden, 3 residual blocks
- **Training**: 256 envs, 2048 steps, 2000 updates, PFSP sampling
- **Known Bug**: ELO update was broken — `update_elo()` never updated opponent ELO, so all entries stayed at 1000. Fixed on 2026-02-19.

### vivid-cobra-selfplay
- **Status**: DIED at update ~1
- **Pool**: 2 entries (sp_0000, sp_0001), 256 total games
- **Postmortem**: Crashed almost immediately. Likely an early iteration bug.

### grim-talon-selfplay
- **Status**: DIED at update ~11
- **Pool**: 4 entries up to sp_0011. No checkpoints.

### grim-knight-selfplay
- **Status**: DIED at update ~20
- **Pool**: 7 entries up to sp_0020. No checkpoints.

### odd-lancer-selfplay
- **Status**: DIED at update ~24
- **Pool**: 10 entries up to sp_0024. No checkpoints.

### keen-knight-selfplay
- **Status**: DIED very early
- **Artifacts**: Only config.json saved. No pool, no checkpoints.

### grim-falcon-selfplay
- **Status**: DIED at update ~54
- **Pool**: 14 entries up to sp_0054. 1 checkpoint (update 50).

### vivid-icarus-selfplay (longest v1 run)
- **Status**: DIED at ~update 1200/2000
- **Pool**: 30 entries (maxed out) up to sp_1205. All ELOs stuck at 1000 (the bug).
- **Checkpoints**: 12 saved, up to step_629145600 (~1200 updates)
- **Notes**: The most successful v1 run by far. Still had the ELO bug so PFSP sampling was random. Unclear why it died — possibly timeout (8hr limit with 256 envs at 2000 updates may exceed it).

---

## Phase 4: Self-Play v2 (ELO bug fixed, mixed sampling)

v2 runs use `selfplay_v2.json` config:
- **Model**: 384 hidden, 3 residual blocks
- **Training**: 256 envs, 2048 steps, 2000 updates, **mixed** sampling, ent_coef=0.005
- **Pool**: max_size=40, snapshot_every=25
- **Key Fix**: Symmetric ELO updates — opponents now get proper ELO ratings

### loud-ghost-selfplay_v2
- **Status**: DIED at ~update 381/2000
- **Pool**: 40 entries (maxed out) up to sp_0381 with **real ELOs** (range 686-1529)
- **Checkpoints**: 7 saved, up to step_183500800 (~350 updates)
- **ELO Highlights**:
  - Highest: sp_0335 (ELO 1529), sp_0325 (ELO 1430)
  - Lowest: sp_0116 (ELO 686), sp_0040 (ELO 722)
  - Average late-pool ELO ~1000-1400
- **Notes**: First run with working ELO. Shows genuine skill progression. Died early — cause unknown.

### tight-saber-selfplay_v2
- **Status**: UNKNOWN (download failed due to file conflict)
- **Config**: Same as loud-ghost (selfplay_v2)

---

## Lessons Learned

1. **ALWAYS use `modal run --detach`** — sharp-talon loss proved this
2. **ELO bug was critical** — v1 self-play had broken opponent ratings; fixed with symmetric ELO update
3. **Architecture mismatch risk** — train.py has no --hidden/--n-blocks args, always uses defaults (256h/0b). Self-play config (384h/3b) needs train_selfplay.py which does accept these args
4. **Modal timeout** — 8hr timeout may not be enough for 2000 updates with 256 envs
5. **Many early self-play runs died quickly** — possibly crashes in the self-play infrastructure before it stabilized
6. **Curriculum training is solved** — mean-knight-production achieves 100% vs all scripted opponents

## Next Steps

- [ ] Run selfplay v2 with `--detach` and longer timeout to completion
- [ ] Consider unified pipeline (curriculum -> self-play in one run)
- [ ] Evaluate best self-play checkpoints against scripted opponents
- [ ] Export best model to ONNX for submission
