# Work Notes

## 2026-06-17 - Static demand inverse design, v4/v5 scenarios, random initialization

### Summary

Pivoted from moving-hotspot tracking (hard; major head kept failing to follow the
swarm) to **static demand with inverse design**: define the desired two-layer
deployment first, then tune the environment so that deployment is the unique cost
minimum, then verify RL emerges it.

### Key findings

**Scenario evolution (see HANDOFF.md §5 for full detail):**
- v3 (moving hotspot): learned demand-matching but hub lagged swarm by 1528 m;
  hub ablation negative on moving targets (major head overwhelmed 1/K gradient share).
- v4 (static hotspot, fixed 0.85R start): W₁ 2220→505 GOAL MET, hub ablation +71%
  (static target fixed the major head); but deployments showed a **systematic vertical
  band bias** — swarm always shifted toward the hotspot from the same direction.
  Root cause: fixed starting corner created a directional habit, not relative steering.
- v4-fixed (hotspot and start both fixed): exploration deadlock — group never left the
  staging point (accept ≈ 0). Shows that randomization is the curriculum.
- **v5 (random hotspot + random swarm/hub start, current main branch)**:
  Symmetric convergence around hotspot (band bias gone); hub-to-hotspot 1185→555 m.
  New issue: all 16 UAVs crowd the hotspot (13.8 of 16 within 1.5σ), background
  abandoned — root cause is the ×4 access bandwidth scaffold making hotspot marginal
  value too high (n*=5 trough is shallow at access×4).

**Phase-1 inverse design probe (phase1_design.py):**
Recipe [C]: K=16, σ=600, peak=0.008, bg=1e-5, C_U=0.35 (f_U=1.4 GHz),
C_H=10.5 (f_H=42 GHz), access bandwidth ×4 (320 MHz scaffold).
n*=5 verified: cost trough at n=3-5 hotspot UAVs, 6th UAV better on background;
single-UAV overflow → 3 UAVs clear it (soft-split sharing confirmed).
cap/offered=0.87, ovf=0 (capacity covers demand).

**Code changes (on top of v3 work):**
- `finite_k_env.py`: added `_initial_deployment()` — per-episode random hub/swarm
  centroid in [0.3,0.7]² box; 16 UAVs on a 1km×1km 4×4 grid (333m spacing, zero
  collision penalty); backward-compatible (falls back to yaml fixed values when
  `initial_deploy` key absent).
- `finite_k_env.py`: `_initial_demand_motion` supports `initial_center_frac_range`
  for per-episode random hotspot centre.
- `config_loader.py`: UAV generator `cluster` (compact staging formation at a given
  corner frac) and hub `swarm_centroid` (hub start = UAV centroid).
- New scenarios: `v4_static_demand.yaml`, `v4_fixed_demand.yaml`,
  `v5_static_randinit.yaml` (all derived from v3 via make_v4.py / direct edit).

**Access model clarification:**
Orthogonal FDMA (W/K per UAV, bare SNR) is the confirmed and final access model.
A full-reuse SINR interference model was attempted but produced 10× rate collapse
for 2 vs 1 co-located UAVs — interference too strong, reverted to orthogonal.
HANDOFF.md §6 notes this constraint explicitly.

**Open problem:** v5 all-hotspot crowding. Next step: reduce access bandwidth from
320 MHz back to physical 80 MHz (W/K shrinks → hotspot marginal rate drops with
added UAVs → two-layer becomes optimal). Use phase1_design.py at 80 MHz to verify
n* stays ≈ 5 before retraining.

### Verification

```bash
C:/Users/Administrator/anaconda3/envs/marl/python.exe -m pytest onpolicy/envs/mec/tests -q
# 12 passed, 1 skipped
```

v5 training (seed 1, 1.5e6 steps): reward −68→−42 (trend/noise 4.3), W₁ 565→360,
accept 14.4→17.2, U_src 4.0→1.27 (96% acceptance), ovf ≈ 0 throughout.

## 2026-06-16 - Archive checkpoint: MEC v3 training and PPO log-prob fix

### Summary

Created a Git checkpoint for the current MEC research branch. This archive captures
the transition from a physically locked but hard-to-learn v2 setup to the learnable
v3 scenario, plus the PPO continuous-action `action_log_probs` buffer fix needed for
correct actor-loss scaling.

### Scope

- MEC v3 scenario: K=12, doubled compute capacity, larger queues, and updated docs
  explaining why v2 remains a parity baseline while v3 is used for learnability.
- Training stability: low initial MEC Gaussian log-std, entropy annealing arguments,
  and MEC console/WandB metrics including demand-matching W1.
- Correctness fix: shared and separated replay buffers now store joint log-probs
  with width 1 for Box/Discrete actions, avoiding act-dim broadcast amplification.
- Documentation: README/spec/work notes now record the v2 flat-landscape diagnosis,
  v3 training command, verified result, and remaining follow-up items.

### Verification recorded

Previous run recorded in this note set:

```bash
conda run -n marl python -m pytest tests onpolicy/envs/mec/tests onpolicy/algorithms/mec/tests -q
# 26 passed, 1 skipped, 20 subtests passed
```

## 2026-06-15 - Fix action_log_probs broadcast (act_dim x policy-loss amplification)

### Summary

Found and fixed a real shape bug inherited from upstream on-policy: for continuous
(Box) action spaces the actor returns a single JOINT log-prob `[.,1]`, but the
replay buffers allocated `action_log_probs` with `act_shape` columns
(`get_shape_from_act_space(Box)=act_dim`). NumPy broadcasts the scalar into
`act_dim` identical columns on insert, and `r_mappo`'s `torch.sum(min(surr1,surr2),
dim=-1)` then sums those copies, multiplying the PPO **actor loss/gradient by
act_dim** (3x for MEC's `[vx,vy,beta]`). Discrete spaces (act_shape=1, e.g. MPE
simple_spread) are unaffected.

### Evidence (empirical, `verify_logprob_bug.py`)

- actor `get_actions` -> `action_log_probs` shape `[B,1]` (joint).
- buffer stored width 3, value broadcast to `[x,x,x]`.
- exact r_mappo math: `policy_loss` current 0.2914 vs correct 0.0971 = **3.0000x**;
  actor grad-norm 40.35 vs 13.45 = 3x.

### Impact (why v3 still learned)

With Adam, a global kx on the policy gradient ~cancels in `m/sqrt(v)`, and when the
grad-norm exceeds `max_grad_norm=10` it is fully absorbed by clipping. The only
material residue: the SEPARATE entropy term is not amplified, so the effective
entropy coefficient was ~1/3 of nominal. v3 deliberately uses low exploration
(`logstd_init=-1.9`, entropy anneal), so the prior v3 result was not invalidated.

**Confirmed on fixed code**: re-running v3 (seed 1) with `entropy_coef=0.01` softened
slightly (cost 0.568, W1 787, hub +15%) because the fix restored entropy to nominal
(too high here); setting `entropy_coef=0.003` (~= the effective value the bug had been
producing) recovered and slightly exceeded the original: **cost 0.509, W1 733 (12%
better than hover, eval "GOAL MET"), hub ablation +27.9%**. The final correct config
is v3 + `--mec_logstd_init -1.9 --use_entropy_anneal --entropy_coef 0.003 --ppo_epoch 5`.

### Fix

- `onpolicy/utils/shared_buffer.py` and `onpolicy/utils/separated_buffer.py`:
  allocate `action_log_probs` with last dim `1` for Discrete/Box (joint log-prob),
  keeping `act_shape` only for MultiDiscrete (per-dim, as `ACTLayer` emits). Single
  localized change; generators read `shape[-1]` so they adapt.

### Verification

```bash
conda run -n marl python -m pytest tests onpolicy/envs/mec/tests onpolicy/algorithms/mec/tests -q
# 26 passed, 1 skipped, 20 subtests passed (no regression; MPE Discrete unaffected)
```

## 2026-06-15 - MEC v3 learnable scenario, flat-landscape diagnosis, first successful training

### Summary

Trained MAPPO on the MEC env end-to-end. The locked v2 scenario does **not** learn
demand-matching: the cost landscape is nearly flat w.r.t. UAV positioning, so the
policy drifts to a worse-than-hover solution. Diagnosed the root cause, derived a
learnable **v3** scenario, added exploration fixes, and trained a policy that
learns demand-matching (W1 below the hand-designed heuristic). Synchronized all
docs/comments with the finding.

### Diagnosis (why v2 failed)

- PPO machinery is healthy (ratio==1, critic explained_variance ~0.95) -- not a bug.
- The trained v2 policy is **worse than hover** on both cost (0.795 vs 0.709) and
  W1 (835 m vs 603 m); entropy collapses onto a noise-driven, wrong-direction policy.
- Quantified cause: demand-matching (heuristic) beats hover by only **~3.6%** on team
  cost, below the ~11% per-episode exogenous noise -> the policy gradient follows noise.
- Two structural reasons positioning barely affects cost:
  1. **Geometric over-provisioning**: 24 UAVs x 861 m coverage over-cover the 6 km
     field (~1.55x); hovering already blankets the hotspot (hover W1 603 < sigma 800).
  2. **Severe overload**: offered ~33 vs capacity 13.5 Mbit/slot (2.4x); the binding
     constraint is processing, not coverage, and queue/energy costs ride on accepted
     data so equal lambda still leaves a mild under-collection incentive.

### Key changes

- Added **`onpolicy/envs/mec/scenarios/v3_iort_learnable.yaml`** (derived from v2):
  K=12, uav cpu 1->2 GHz (C_U 0.25->0.5), hap cpu 30->60 GHz (C_H 7.5->15, overload
  2.4x->1.6x), queues x1.5; speeds unchanged (UAV 40 > HAP 30). Verified positioning
  is now load-bearing: heuristic-vs-hover cost gap +26% (was +1.8%), cost becomes
  source-loss (coverage) dominated.
- Exploration fixes: `--mec_logstd_init` (default -1.9, sigma~0.15; was a fixed 0 ->
  sigma 1 max-speed careening) in `MECActor`; `--use_entropy_anneal` /
  `--entropy_coef_min` linear entropy-coef anneal in the shared runner; new args in
  `config.py`.
- Added a per-log-interval `[mec]` console trace and a training-time W1 metric
  (`mec/w1`) to `mec_runner` / `MEC_env` for live monitoring.
- Doc/comment sync: fixed the stale `test_health_check.py` run command; updated the
  yaml lambda-description comment; rewrote `docs/mec_env_port_spec.md` to add a
  prominent diagnosis note in section 0, refreshed training tips (8.1), v3 training
  command (8.3), progress (8.4), a decision-history row (9), and a new section 10
  documenting the diagnosis + v3 + results.

### v3 result (K=12, 1.5e6 steps, CPU ~1 h, seed 1)

(Numbers below are the FIRST success, produced before the action_log_probs fix
documented in the newer note above. The corrected-code final config -- fix +
`entropy_coef=0.003` -- recovers/slightly exceeds them: cost 0.509, W1 733
"GOAL MET", hub +27.9%.)

| controller | team cost/slot | W1 (m) | hub ablation |
|---|---|---|---|
| trained policy | 0.514 | **755** | +25.3% |
| heuristic | 0.500 | 763 | - |
| hover | 0.674 | 830 | - |
| random | - | 1155 | - |

Full-run trends (trend/noise): reward -133->-105 (7.7), W1 986->796 (2.6), accepted
13.7->16.5, U_src 10.6->7.4 (2.9). Velocity-direction probe: minor mean velocity now
points toward the demand (v2 pointed the wrong way). Demand-matching emerges from
team-cost minimization; W1 beats the hand-designed heuristic. A second seed is training.

### Verification

```bash
conda run -n marl python -m pytest tests onpolicy/envs/mec/tests onpolicy/algorithms/mec/tests -q
# 26 passed, 1 skipped, 20 subtests passed (unchanged by the edits)
conda run -n marl python -m onpolicy.scripts.eval.eval_mec --env_name MEC \
  --mec_scenario v3_iort_learnable --mec_eval_controller policy \
  --model_dir <run1/models> --mec_eval_episodes 8
```

## 2026-06-15 - MEC environment port, testing, and checkpoint config

### Summary

Integrated the current MEC implementation into the on-policy project, improved
the test entrypoint, and made saved model directories carry enough run
configuration for later evaluation or rendering.

### Key changes

- Added the MEC finite-fleet environment, scenario configuration, metrics,
  shared runner, policy wrapper, and train/eval/render entrypoints.
- Documented the MEC porting assumptions and expanded `README.md` with the
  unified pytest command for the active `marl` environment.
- Added `pytest` to project requirements.
- Removed the local absolute path dependency from the MEC parity test by using
  `MFMEC_ORIGIN_SRC`; the parity test is skipped when the reference source is
  unavailable.
- Saved `config.json` beside each run and under `models/`, then taught MEC
  eval/render scripts to load saved arguments unless the user explicitly
  overrides them on the command line.

### Verification

Executed successfully on 2026-06-15:

```bash
/opt/anaconda3/envs/marl/bin/python -m compileall -q onpolicy tests
/opt/anaconda3/envs/marl/bin/python -m pytest tests onpolicy/envs/mec/tests onpolicy/algorithms/mec/tests -q
MFMEC_ORIGIN_SRC="/Users/qiaonan/Projects/Mean Field Mec/src" /opt/anaconda3/envs/marl/bin/python -m pytest onpolicy/envs/mec/tests/test_finite_k_env.py -q
/opt/anaconda3/envs/marl/bin/python -m onpolicy.scripts.train.train_mec --env_name MEC --algorithm_name mappo --experiment_name config_smoke --mec_fleet_size 2 --n_rollout_threads 1 --episode_length 2 --num_env_steps 2 --ppo_epoch 1 --num_mini_batch 1 --hidden_size 8 --layer_N 1 --use_wandb
/opt/anaconda3/envs/marl/bin/python -m onpolicy.scripts.render.render_mec --model_dir /Users/qiaonan/Projects/on-policy/onpolicy/scripts/results/MEC/v2_iort_6km_mmwave/mappo/config_smoke/run1/models --episode_len 1 --out /tmp/mec_config_smoke.gif
/opt/anaconda3/envs/marl/bin/python -m onpolicy.scripts.eval.eval_mec --env_name MEC --mec_eval_controller policy --model_dir /Users/qiaonan/Projects/on-policy/onpolicy/scripts/results/MEC/v2_iort_6km_mmwave/mappo/config_smoke/run1/models --mec_eval_episodes 1
/opt/anaconda3/envs/marl/bin/python -m onpolicy.scripts.render.render_mec --model_dir /Users/qiaonan/Projects/on-policy/onpolicy/scripts/results/MEC/v2_iort_6km_mmwave/mappo/v2_run1/run2/models --hidden_size 128 --layer_N 2 --episode_len 1 --out /tmp/mec_legacy_config_smoke.gif
git diff --check
```

The unified pytest run reported `26 passed, 1 skipped, 2 warnings, 20 subtests
passed`. The skip is expected when `MFMEC_ORIGIN_SRC` is not provided.

## 2026-06-10 - MPE rendering setup corrections

### Summary

Improved the retained MPE rendering path so shared-policy rendering initializes
its output directory correctly and reports dependency or OpenGL setup failures
clearly.

### Key changes

- Initialized the shared runner's render-mode `run_dir` and `gifs` directory
  without depending on training or Weights & Biases setup.
- Added the compatible `pyglet>=1.5,<2` rendering dependency explicitly.
- Replaced invalid fallback print calls during rendering imports with actionable
  exceptions for missing pyglet and unavailable OpenGL.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
conda run -n marl python -c "import pyglet; print(pyglet.version)"
git diff --check
```

All 10 tests passed. The active `marl` environment reports pyglet version
`1.5.31`, which satisfies the pinned compatibility range.

## 2026-06-10 - Replay buffer indexing and sampling corrections

### Summary

Audited the shared and separated replay buffers after the Gymnasium migration
and corrected several indexing and sampling issues that could misalign rollout
transitions, omit training samples, or form invalid recurrent sequences.

### Key changes

- Standardized replay buffer indexing so transition data is stored at time
  `t`, while observations, recurrent states, and masks for the resulting state
  are stored at `t + 1`.
- Corrected return calculation to use each transition's actual next-value
  prediction and the mask belonging to its next state.
- Populated shared-buffer advantages for non-GAE return calculation.
- Changed feed-forward mini-batch sampling so remainder samples are retained
  instead of silently dropped.
- Added divisibility and size checks for naive-recurrent and recurrent
  mini-batches.
- Prevented recurrent chunks from crossing agent or environment trajectory
  boundaries.
- Corrected separated recurrent batches to flatten in time-major order expected
  by the recurrent layer.
- Removed the obsolete sequential-agent factor update path from the separated
  runner and added visible average-reward logging.
- Added focused regression tests for buffer insertion, return calculation,
  remainder sampling, recurrent ordering, and chunk-boundary validation.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
git diff --check
```

All 10 tests passed, including the 5 new replay-buffer indexing and sampling
regression tests and the existing 5 Gymnasium migration tests.

## 2026-06-10 - MPE Gymnasium migration and rollout semantics

### Summary

Completed the first substantial update after the repository was reduced to the
MPE-focused version. The work modernizes the retained MPE stack for Gymnasium
and NumPy 2, and corrects how episode termination and time-limit truncation are
handled during rollout and return calculation.

### Key changes

- Migrated MPE environments, spaces, wrappers, training, evaluation, and
  rendering flows from the legacy Gym API to Gymnasium.
- Replaced global scenario randomness with seeded per-environment generators,
  making seeded resets reproducible.
- Rebuilt the retained vector environment wrappers with Gymnasium-style reset,
  step, autoreset, seed, and final-observation handling.
- Separated true task termination from time-limit truncation:
  - recurrent state and GAE continuation stop at either boundary;
  - critic bootstrapping remains enabled for truncation;
  - critic bootstrapping is disabled for true termination.
- Updated shared and separated MPE runners and replay buffers to preserve and
  evaluate the actual final observation before autoreset.
- Added PPO diagnostics for approximate KL divergence, clipping fraction, and
  explained variance, plus optional `--target_kl` early stopping.
- Updated dependencies for Gymnasium and NumPy 2 compatibility.
- Added focused migration tests and documented the new semantics in `README.md`.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
git diff --check
```

All 5 migration tests passed. The tests cover all retained MPE scenarios,
reproducible seeded resets, time-limit truncation, vector autoreset final
observations, and truncation-versus-termination bootstrapping.

### Notes

- `origin` is the private archive repository `YijieWang-2024/on-policy`.
- `upstream` remains the original `marlbenchmark/on-policy` repository.
- MPE environments should report external time limits as truncation and reserve
  termination for true task-ending conditions.
