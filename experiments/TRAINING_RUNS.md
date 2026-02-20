# Training Runs Log

> Last updated: 2026-02-19

## Summary

| # | Name | Type | Updates | Status | Key Result |
|---|------|------|---------|--------|------------|
| 1 | baseline-v1 (local) | curriculum | 300 | COMPLETED | 19% win rate — gamma/reward bugs |
| 2 | local-curriculum | curriculum | 600 | COMPLETED | 99-100% vs all — the breakthrough |
| 3 | mean-knight-production | production | 750 | COMPLETED | 100% vs all (4% vs do_nothing) |
| 4 | quick-raptor | unified DR+GPU | 2300 | FAILED | ~10-20% WR after 300 updates — DR too early + model too large |
| 5 | eager-icarus-unified_dr_gpu_v2 | unified DR+GPU | 2000 | FAILED | 384h/2b KL explosion at curriculum shift |
| 6 | dark-ghost-unified_dr_gpu_v3 | unified DR+GPU | 2000 | RUNNING | 256h/0b + staged DR — proven arch |

---

## Completed Runs

### baseline-v1 (local)
- **Date**: 2025-02-18
- **Config**: 16 envs, 10800 n_steps, action_repeat=1, 300 updates, gamma=0.99, ent_coef=0.003
- **Model**: 256h/0b
- **Status**: COMPLETED
- **Result**: 19% win rate, -0.34 return
- **Notes**: First training run. Multiple bugs — gamma too low, no action_repeat, entropy explosion, sparse rewards only.

### local-curriculum (the breakthrough)
- **Date**: 2025-02-18
- **Config**: 32 envs, 2048 n_steps, action_repeat=10, 600 updates, gamma=0.999, ent_coef=0.0
- **Model**: 256h/0b
- **Checkpoints**: `training/checkpoints/` — step_3276800 through final.pt, pre_brawler.pt
- **Status**: COMPLETED
- **Result**: 99-100% vs dogfighter/chaser/ace/brawler, ~70% vs do_nothing (timeout draws)

### mean-knight-production (Modal, best curriculum)
- **Config**: 128 envs, 2048 steps, 750 updates, 256h/0b
- **Status**: COMPLETED
- **Artifacts**: `training/modal_results/mean-knight-production/` — checkpoints + eval.txt
- **Eval Results**:
  | Opponent | Wins | Draws | Losses | Win% |
  |----------|------|-------|--------|------|
  | do_nothing | 2 | 48 | 0 | 4% |
  | dogfighter | 50 | 0 | 0 | 100% |
  | chaser | 50 | 0 | 0 | 100% |
  | ace | 50 | 0 | 0 | 100% |
  | brawler | 50 | 0 | 0 | 100% |

---

## Failed Runs

### quick-raptor-unified_dr_gpu_v1 (Modal)
- **Date**: 2026-02-19
- **Config**: `experiments/configs/unified_dr_gpu_v1.json` — 768h/4b, DR narrow from start
- **Status**: FAILED — stopped after ~300 updates
- **Problem**: Win rate flat at ~10-20%. Model simultaneously learning to fight AND adapt to random physics. DR noise during curriculum made reward signal too noisy for hill-climbing. Model also 3x larger than proven 256h/0b.
- **Lesson**: Don't apply domain randomization during curriculum. Let the model learn to fight on stable physics first, then introduce variation.

---

## Live Runs

### eager-icarus-unified_dr_gpu_v2 (Modal)
- **Date**: 2026-02-19
- **Config**: `experiments/configs/unified_dr_gpu_v2.json` — 384h/2b, staged DR
- **Status**: FAILED — KL divergence explosion at curriculum shift (update ~75)
- **Problem**: 384h/2b model + lr=3e-4 too aggressive. Single PPO gradient step caused KL of 8-24 (500x above target_kl=0.05). Policy collapsed catastrophically at do_nothing→dogfighter transition. Win rate went 0→100→0 and never recovered.
- **Lesson**: Larger models amplify gradient magnitude. Must reduce LR proportionally or stick with proven architecture.

---

## Live Runs

### dark-ghost-unified_dr_gpu_v3 (Modal)
- **Date**: 2026-02-19
- **Type**: Unified DR with GPU physics sim (NVIDIA Warp), staged DR
- **Config**: `experiments/configs/unified_dr_gpu_v3.json`
- **Model**: 256h/0b, obs_dim=59 (proven architecture from mean-knight-production)
- **Training**: 256 envs, 2048 steps, H100 GPU, fp16=true
- **Phases**: curriculum(700) + transition(200) + selfplay(1100) = 2000 updates
- **DR schedule**: none until 700 → narrow until 900 → full after 900
- **Modal app**: `ap-2XebXrcvWcIV45hJDk6npS`
- **Status**: RUNNING — 65% WR at update 10 (strong start)

---

## Lessons Learned

1. **ALWAYS use `modal run --detach`** for real training runs
2. **fp16 needs safety**: logratio clamping + KL early stopping + warmup period
3. **Curriculum training is solved** — mean-knight-production achieves 100% vs all scripted opponents
4. **Don't apply DR during curriculum** — the model needs stable physics to learn basic fighting. Noisy physics + learning = flat win rates. Use staged DR: none → narrow → full.
5. **Don't jump to 3x larger models** — 768h/4b failed where 256h/0b succeeded. Use moderate increases (384h/2b).
