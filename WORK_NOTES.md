# Work Notes

## 2026-06-26 - Aligned MLP readout repaired; Mean/Flat training gate recovered

### Root cause

The poor `mean/flat/set` results from the 350-slot architecture sweep were not
caused by the PPO runner, buffer insertion, Gymnasium autoreset handling, or
the public-state/resource-context change.  The main regression was in the
aligned population readout path: `FusionMLP` was a bare two-layer
`Linear + activation` stack and did not honor the legacy MAPPO `MLPBase`
optimization contract.

Specifically, it bypassed:

- `use_feature_normalization` input LayerNorm;
- `use_orthogonal` / Xavier initialization choice;
- configured `layer_N` hidden depth;
- per-hidden-layer LayerNorm from `MLPLayer`.

This made the aligned Mean/Flat/Set runs numerically different from the proven
legacy `MLPBase` baseline even when the high-level MAPPO algorithm was
unchanged.

### Fix

- `onpolicy/algorithms/mec/set_networks.py`: `FusionMLP` now wraps the same
  `MLPLayer` contract as `MLPBase`, with optional input LayerNorm.
- `onpolicy/algorithms/mec/mec_policy.py`: aligned actor readouts and team
  critic readout now pass `layer_N`, `use_orthogonal`, and
  `use_feature_normalization`.
- `onpolicy/algorithms/mec/tests/test_set_mec_policy.py`: added a regression
  test locking the readout contract.

### Verification

- Focused policy/buffer/env tests: `25 passed, 20 subtests passed`.
- Full suite: `57 passed, 1 skipped, 20 subtests passed`.
- `compileall` passed.
- `git diff --check` passed, aside from expected LF/CRLF warnings.

### 350-slot one-seed regression gate after the fix

Both gates used seed 1, 350-slot episodes, 16 rollout workers, 160 PPO updates,
896k environment steps, validation seed 1000 for best-checkpoint selection, and
held-out test seed 100000 / stride 13 / 24 episodes for reporting.

| architecture | held-out cost/slot | accept | W1 diagnostic | HAP-freeze delta |
|---|---:|---:|---:|---:|
| mean, fixed readout | 3.2557 | 59.4% | 919.1 m | +2.2% |
| flat, fixed readout | 2.9574 | 62.6% | 883.9 m | -0.2% |
| legacy_mean reference, previous 3-seed mean | 2.7470 | 65.4% | 767.9 m | +15.2% |
| heuristic reference | 2.0694 | 73.7% | 734.1 m | n/a |

Training curves recovered clearly:

- Mean validation improved from about `-1519` early to `-1107` at 896k.
- Flat validation improved from about `-1484` early to `-986` at 896k.
- PPO diagnostics stayed healthy: approximate KL remained small after the
  first update, clip fraction did not explode, and explained variance reached
  about `0.98-0.99`.

### Decision

Mean and Flat are now using MAPPO correctly enough to learn.  Flat is close to
the historical `legacy_mean` reference, while Mean remains weaker but no longer
collapses.  The next step is not decoder/Sinkhorn/PPG yet; first run the fixed
`set` seed-1 gate, then rerun the formal 3-seed Mean/Flat/Set comparison if Set
also shows recovered movement.

## 2026-06-26 - 350-slot aligned architecture experiment completed

### Formal protocol

- Completed `mean/flat/set x seed 1/2/3` with 350-slot episodes,
  16 rollout workers, 160 PPO updates, and 896,000 environment steps per run.
- Best-checkpoint selection used only validation seed 1000 over 24 episodes.
- Final reporting used the disjoint held-out split with base seed 100000,
  stride 13, and 24 episodes.
- All nine training runs and held-out evaluations completed without stderr.

### Held-out result

| architecture | cost/slot | accept | spatial diagnostic | HAP-freeze delta |
|---|---:|---:|---:|---:|
| mean | 4.6491 +/- 0.1292 | 38.9% +/- 1.5% | 1285.7 +/- 14.6 m | -0.3% +/- 0.5% |
| flat | 4.5822 +/- 0.0900 | 39.7% +/- 1.2% | 1260.9 +/- 32.9 m | approximately 0% |
| set | 4.6230 +/- 0.0669 | 39.1% +/- 0.9% | 1287.0 +/- 9.3 m | approximately 0% |
| heuristic | 2.0694 | 73.7% | 734.1 m | n/a |

The Set policy is numerically stable across seeds, but task performance is not
ready for reconstruction: acceptance remains below 40%, the learned HAP
trajectory is not load-bearing, and the spatial diagnostic does not improve
meaningfully over hover.

### Regression localization

- Representative action ablations show that freezing learned UAV motion
  improves cost by 1.4% to 1.9%. Mean UAV velocity has a negative projection
  toward the hotspot, and mean HAP velocity has a negative projection toward
  the UAV centroid.
- Historical role-wise `legacy_mean` checkpoints were evaluated unchanged on
  the same 350-slot held-out split. They achieve cost/slot
  `2.5610/2.8504/2.8297`, acceptance `67.9%/64.2%/64.1%`, and HAP-freeze
  degradation `+19.4%/+10.2%/+16.0%`.
- Therefore the longer horizon is not the cause. The regression is in the new
  aligned representation/readout path.
- The new `FusionMLP` path bypasses the configured input feature
  normalization, per-layer normalization, orthogonal initialization, and
  `layer_N` depth used by the proven legacy `MLPBase`. This is the first
  architecture contract to repair and ablate.

### Decision

Do not add the decoder, Sinkhorn reconstruction loss, or PPG auxiliary phase
yet. First restore an optimization-equivalent normalized actor/critic readout,
then rerun a one-seed regression gate against `legacy_mean`; only after the
aligned Set policy recovers useful UAV and HAP motion should the three-seed
comparison be repeated.

## 2026-06-26 - 350-slot aligned architecture experiment launch

### Decisions

- The nearest-UAV spatial diagnostic remains an environment-design metric; it
  is not the future decoder's Sinkhorn reconstruction objective.
- Beta sensitivity is deferred to its dedicated mildly overloaded scenario and
  does not block the current representation experiment.
- The training horizon is increased from 200 to 350 slots. With 16 rollout
  threads, the budget is increased from 512k to 896k environment steps so each
  run retains 160 PPO updates.
- Checkpoint selection uses a fixed validation split (`base_seed=1000`,
  24 episodes). Final structure reporting uses a disjoint held-out split
  (`base_seed=100000`, stride 13, 24 episodes).

### Information contract

The public state now contains the seven physical HAP/demand features plus six
dimensionless fleet/resource features:

```text
K/K_ref,
W_ac,i/W_ac,total,
W_bh,i/W_bh,total,
K*C_U/(K*C_U+C_H),
C_H/(K*C_U+C_H),
(K*C_U+C_H)/D_ref.
```

Canonical local rows are 17-D and centralized state is `13 + 3K`. Historical
14-D checkpoints remain supported through the legacy adapter, which discards
the new resource context when reconstructing historical rows.

### Verification and launch

- Focused and full suites passed: `53 passed, 1 skipped, 20 subtests passed`.
- Parallel 350-slot Mean/Flat/Set smoke completed training, validation
  checkpointing, and held-out JSON evaluation.
- Workstation calibration: three concurrent runs use about 4.5 GB GPU memory;
  CPU, rather than GPU memory, is the limiting resource.
- Formal `mean/flat/set x seed 1/2/3` experiments are running in three waves
  through `scripts/run_v6_h350_arch_parallel.ps1`.
- `summarize_h350_arch.py` produces cross-seed statistics and explicit
  reconstruction-readiness gates after held-out evaluation.

## 2026-06-26 - Phase archive: aligned MEC population policies

### Archived scope

This archive closes the engineering phase that began with HAP load-bearing
calibration and ended with an information-matched Set-MAPPO phase-1 stack.

Included work:

- resumable latest/step/best checkpoints and saved run configuration;
- role-balanced major/minor MAPPO optimization;
- beta identifiability and stress-scenario diagnostics;
- theory review against the implemented finite-K MEC dynamics;
- canonical 11-D local observations and `7 + 3K` centralized state;
- switchable `legacy_mean`, aligned `mean`, ordered `flat`, and invariant `set`;
- shared set encoder ownership without duplicate actor/critic Adam states;
- old 14-D actor/critic checkpoint compatibility;
- training, evaluation, rendering, analysis, and experiment launch scripts;
- structural, checkpoint, environment, and beta diagnostic tests.

### Documentation review

No Markdown file was removed because the current files have distinct roles:

- `HANDOFF.md`: current state, evidence, and next action;
- `WORK_NOTES.md`: chronological engineering record;
- `docs/mec_runbook.md`: reproducible commands and experiment procedure;
- `docs/mec_env_port_spec.md`: system contract and historical decisions;
- `docs/setrec_architecture.md`: current algorithm and optimizer contract;
- `docs/diagnosis_layout_not_loadbearing.md`: archived negative evidence.

### Validation and next stage

The focused suite passed with `51 passed, 1 skipped, 20 subtests passed`.
Aligned mean, flat, and set each completed an end-to-end training smoke.
New set checkpoints and historical legacy-mean checkpoints both reloaded; the
historical checkpoint also restored optimizer/ValueNorm state and resumed
training.

The next stage is a fresh equal-budget, matched-seed `mean/flat/set`
comparison. Reconstruction and the PPG-style auxiliary phase remain deferred
until Set-MAPPO is stable across seeds.

## 2026-06-25 - Information-matched mean/flat/set refactor

### Decision

The historical mean implementation was not a clean representation baseline:
its actor used a shared 14-D role-conditioned trunk while its critic consumed
the ordered flattened team rows. The aligned comparison now holds actor
readouts, team critic, grouped PPO batching, and optimizer ownership fixed.

### Implementation

- Changed local MEC observations to `[role, own(3), p(7)]` (11 dimensions).
- Added canonical centralized state `[p(7), s_1(3), ..., s_K(3)]`.
- Aligned actor inputs:
  - HAP: `p + representation`;
  - UAV: `s_i + p + representation`.
- Aligned critic input to `p + representation`.
- Added aligned `mean`, ordered `flat`, and invariant `set` representations.
- Renamed the old network semantics to `legacy_mean`.
- Added adapters that reconstruct historical 14-D rows so old actor/critic
  state dictionaries still load exactly.
- Added automatic `legacy_mean` selection for model directories whose saved
  config predates `mec_policy_arch`.

### Experiment implication

The old mean runs remain historical engineering references. A fair
representation ablation requires fresh equal-budget `mean/flat/set` runs.

### Verification

- `51 passed, 1 skipped, 20 subtests passed`.
- Mean, flat, and set each completed a one-update end-to-end training smoke.
- A new set checkpoint reloaded and completed deterministic evaluation.
- A historical 16-UAV checkpoint with no architecture metadata was inferred as
  `legacy_mean`, evaluated, restored with optimizer/ValueNorm state, and trained
  for one further update.

## 2026-06-25 - SetRec theory review and architecture freeze

### Summary

Reviewed the proposed SetRec theory and algorithm against the actual
`v6_hap_loadbearing` state, action, transition, queue, and cost implementation.

Conclusions:

- The finite-K environment is exchangeable under joint permutation of UAV states
  and actions. The empirical-measure reformulation remains a sound method-level
  starting point.
- The current infinite-horizon Lipschitz theorem is conditional rather than a
  verified property of v6. A normalized same-action perturbation probe observed a
  one-step distance ratio up to about `4.93`, so the current natural metric does
  not justify `gamma * L_F < 1`.
- The public algorithm interface needs only one invariant UAV population
  descriptor. Equivariant element tokens may exist inside the encoder but must
  not bypass the descriptor into the actors.
- The final phase-1 policy is `pi_H(a_H | p, xi)`,
  `pi_U(a_i | s_i, p, xi)`, with one team value `V(p, xi)`.
- Reconstruction will be added only after Set-MAPPO is stable, using a decoder
  that reads only `xi`, debiased Sinkhorn divergence, and a PPG-style auxiliary
  phase with policy-KL protection.

### Baseline contract

Four architecture switches are retained:

- `legacy_mean`: historical mean-descriptor MAPPO, checkpoint compatibility.
- `mean`: aligned low-capacity mean descriptor baseline.
- `flat`: information-complete ordered concatenation baseline. It is fixed-K and
  label-sensitive, and is not described as decentralized-execution MAPPO.
- `set`: the proposed public invariant population encoder.

The detailed contract is now isolated in `docs/setrec_architecture.md`.

### Immediate implementation order

1. Add `legacy_mean/mean/flat/set` architecture selection.
2. Preserve complete team groups in PPO minibatches.
3. Implement the public Set Transformer encoder, role-specific actor readouts,
   and one invariant team critic.
4. Add permutation, ratio, shape, and gradient-ownership tests.
5. Run smoke tests before adding the reconstruction decoder.

### Phase-1 implementation result

Implemented:

- `--mec_policy_arch legacy_mean|mean|flat|set`;
- ordered full-state FlatConcat actors/critic;
- one public Set Transformer population encoder shared by HAP/UAV actors and
  consumed with stop-gradient by one team critic;
- common role-specific fusion MLPs for aligned mean/flat/set readouts;
- grouped PPO minibatches that preserve complete `(time, environment, team)` rows;
- critic-value recomputation after an encoder-changing actor update;
- checkpoint-compatible actor/critic ownership.

The environment physics was not changed. Local rows are now canonical 11-D
physical observations, and the runner builds the canonical team state. A
compatibility adapter reconstructs the historical 14-D rows only inside
`legacy_mean`.

### Verification

```text
New and existing MEC policy tests: passed
Full focused repository suite: 51 passed, 1 skipped, 20 subtests passed
Mean training smoke: passed
Set training smoke: passed
Flat training smoke: passed
Set checkpoint restore smoke: passed
Set policy evaluation from saved config/checkpoint: passed
```

The full-suite first attempt reported two setup errors caused by denied access to
the default Windows pytest temp directory; both tests passed when rerun with a
writable `--basetemp`. No code failure remained.

### Next step

Run fresh equal-budget `mean/flat/set` seed 1/2/3 experiments. Keep existing
role-wise mean results as historical references only. Do not add the decoder
until Set-MAPPO is stable.

## 2026-06-24 - Role-wise MAPPO and resumable checkpoint validation

### Summary

Implemented the first two algorithm-engineering steps after the
`v6_hap_loadbearing` diagnosis:

- resumable latest/step checkpoints plus fixed-seed best-checkpoint selection;
- equal-role MEC advantage, PPO surrogate, and entropy normalization.

Then repeated the 512k-step seed 1/2/3 experiment with the same environment and
training budget.

### Checkpoint changes

- Retain `models/checkpoints/step_<env_steps>/`.
- Save actor, critic, optimizer state, ValueNorm state, step count, and config.
- Evaluate on fixed `eval_seed=1000` episodes and retain `models/best/`.
- Verified restoring training from a numbered checkpoint.

All three role-wise runs selected the final 512k checkpoint. Therefore checkpoint
selection did not change the chosen model in this experiment, but it now prevents
future structural experiments from silently depending on a degraded latest model.

### Role-wise result

| setting | seed | cost/slot | accepted Mbit/slot | W1 | HAP freeze |
|---|---:|---:|---:|---:|---:|
| original | 1 | 2.8311 | 96.0 | 836.1 m | +4.7% |
| original | 2 | 2.4619 | 102.9 | 782.8 m | +11.2% |
| original | 3 | 2.3951 | 104.6 | 774.5 m | +4.6% |
| role-wise | 1 | 2.5011 | 102.5 | 802.0 m | +7.4% |
| role-wise | 2 | 2.3863 | 104.5 | 761.9 m | +10.4% |
| role-wise | 3 | 2.5010 | 102.4 | 797.7 m | +11.2% |

Interpretation:

- Mean cost improved by 3.9% when comparing the mean costs directly
  (2.5627 -> 2.4628).
- HAP freeze improved from +6.8% to +9.7% on average. Major learning is
  materially better, but seed 1 still misses the +10% criterion.
- UAV motion remains strongly load-bearing.
- Per-UAV beta remains ineffective: replacing each step's beta vector by its
  fleet mean changes cost by about -1.0%, +0.03%, and +0.30%.

The next step is therefore SetRec population encoding plus a permutation-invariant
critic, while retaining role-wise training and the new checkpoint protocol.

### Verification

```text
Focused tests: 8 passed
Full suite: 34 passed, 1 skipped, 20 subtests passed
Checkpoint restore smoke: passed
Role-wise 512k training: 3/3 completed without stderr
Matched-seed 24-episode diagnostics: 3/3 completed
```

## 2026-06-24 - Three-seed evaluation and HAP load-bearing calibration

### Summary

Completed the first full training/evaluation stage for `v6_continuous_workload`.
MAPPO seed 1/2/3 each reached 1.5M environment steps and were evaluated on the same
24 deterministic episodes. UAV demand matching is reproducible, but beta/queue
control remains weaker than the heuristic and HAP motion is not robustly learned.

The next cross-machine stage therefore uses `v6_hap_loadbearing` for a 300k-500k
step, 3-seed confirmation before any final long-run experiment.

### Three-seed result

| controller | cost/slot | accept | W1 | queue share | HAP freeze |
|---|---:|---:|---:|---:|---:|
| MAPPO seed 1 | 2.4254 | 72.1% | 756.1 m | 13.1% | +24.0% |
| MAPPO seed 2 | 2.3041 | 72.5% | 765.3 m | 9.9% | +0.5% |
| MAPPO seed 3 | 2.5190 | 71.1% | 761.2 m | 13.1% | +0.2% |
| heuristic | 2.1776 | 72.3% | 762.2 m | 3.6% | - |

Interpretation:

- UAV trajectory learning is load-bearing and reproducible.
- Offloading is active but queue-aware beta control is not yet consistently good.
- The old backhaul budget leaves HAP motion nearly irrelevant in two of three seeds.
- These are environment/algorithm diagnostics, not final paper results.

### Engineering-constrained HAP scan

Kept a realistic FR2 total bandwidth of 400 MHz and changed the net link budget
instead of shrinking mmWave bandwidth to 40-50 MHz. The new candidate uses:

```text
carrier = 28 GHz
W_bh_total = 400 MHz
P_tx = 23 dBm
combined Tx/Rx antenna gain = 20 dB
link margin = 7 dB
noise figure = 8 dB
path-loss exponent = 2.2
hard cutoff = false
```

Ten paired heuristic episodes gave:

```text
HAP freeze: +28.18% cost
beta=0:     +213.30% cost
UAV hover:  +185.99% cost
backhaul utilization: 44.08%
HAP compute utilization: 75.16%
overflow: 0
accepted: 120.526 Mbit/slot
```

This validates that all three action groups affect the mechanism. It does not prove
that the current MAPPO architecture can learn all three simultaneously.

### Code and documentation

- Added `v6_hap_loadbearing.yaml` and `scan_v6_hap_loadbearing.py`.
- Added explicit `link_margin_db` handling to the backhaul link budget.
- Added access/backhaul/UAV-compute/HAP-compute utilization logging.
- Made `design_v6_sanity.py` accept `--scenario`.
- Fixed random evaluation so one RNG stream is retained for an entire episode.
- Reorganized the Markdown documentation around `HANDOFF.md` and
  `docs/mec_runbook.md`; historical diagnosis remains archived rather than deleted.

### Verification

```text
MEC/config/policy tests: 18 passed, 1 skipped
candidate config/env tests: 14 passed, 1 skipped
```

### Next step

On the new machine, run tests, the candidate probe, a smoke test, then seed 1/2/3
for 300k steps. Continue to 500k only while all seeds improve. Do not start the
final 1.5M/5-seed/SetRec experiment until HAP freeze and beta diagnostics are
stable across seeds.

## 2026-06-23 - v6 continuous workload scenario and training-preflight diagnostics

### Summary

Completed the pre-training environment redesign for the MEC task. The current main
candidate is **`v6_continuous_workload`**: a normalized continuous workload field
with finite-K UAV control, continuous 28 GHz backhaul, conservative sub-6 access,
and calibrated compute capacity. This is now ready to move to a stronger machine
for training; the local machine was used only for probes and unit tests.

### Key changes

- Replaced the finite-device scaffold (`base_probability`, hotspot peak increment,
  packet size, per-device bandwidth) with direct workload density
  `lambda(omega, Z_t)` in bits/(m^2 slot). The total fresh workload is controlled by
  `A_tot=150 Mbit/slot`, with `zeta=0.70` in the hotspot and `sigma_h=700 m`.
- Set the access layer to a conservative `W_ac_total=40 MHz` pool, i.e.
  `2.5 MHz/UAV` for `K=16`. Access uses fixed effective PSD `p0=1e-8 W/Hz`,
  `gamma_ref=8 dB` as a service-attractiveness reference, and `tau=0.2`.
- Switched backhaul to continuous 28 GHz Shannon rate with no hard SNR cutoff:
  `W_bh_total=400 MHz`, `25 MHz/UAV`, `P_UH=27 dBm`, and an effective gain of 15 dB.
  This gain must be explained as beamforming gain plus fixed implementation losses,
  not as a bare antenna-gain number.
- Rebalanced compute to `cycles_per_bit=500`, `F_U=2 GHz/UAV`, `F_H=45 GHz`, so
  offered compute load is `150e6 * 500 / (16*2e9 + 45e9) ~= 0.97`.
- Added diagnostic info for source-loss decomposition, hotspot/background
  offered/accepted/source, real access spectral-efficiency percentiles, access and
  backhaul rates, compute/backhaul utilization, UAV hotspot/background counts, and
  hub-to-hotspot distance.
- Added `onpolicy/scripts/analysis/design_v6_sanity.py`, a training-preflight probe
  that uses true signal integration rather than a fixed reference spectral efficiency.

### Probe result

The v6 deploy-and-hold probe over 32 random hotspot centers reports:

```text
best n_hot counts over 32 centers: {5: 14, 6: 10, 7: 8}
```

Representative aggregate rows:

```text
* n_hot=5: accepted=125.7M, source=24.3M(out=15.8M, cap=8.4M), util=0.56
  n_hot=6: accepted=125.4M, source=24.6M(out=19.1M, cap=5.5M), util=0.55
  n_hot=7: accepted=122.9M, source=27.1M(out=22.7M, cap=4.4M), util=0.53
  n_hot=16: accepted=104.0M, source=46.0M(out=46.0M, cap=0.0M), util=0.38
```

The main conclusion is not "exactly five UAVs", but rather a stable 5-7 hotspot
UAV regime with the rest covering background workload. Full hotspot crowding is now
a clear static loss because background workload falls into the outside option.

### Verification

```bash
PYTHONPATH=/Users/qiaonan/Projects/on-policy /opt/anaconda3/envs/marl/bin/python -m pytest \
  onpolicy/envs/mec/tests/test_config_loader.py \
  onpolicy/envs/mec/tests/test_finite_k_env.py \
  onpolicy/algorithms/mec/tests/test_mec_policy.py -q
# 17 passed, 1 skipped, 1 warning

PYTHONPATH=/Users/qiaonan/Projects/on-policy /opt/anaconda3/envs/marl/bin/python -m compileall -q \
  onpolicy/envs/mec onpolicy/scripts/analysis/design_v6_sanity.py

git diff --check
```

### Next step

Run training for `v6_continuous_workload` on a stronger machine. Keep diagnostic
logging enabled and judge success by hotspot/background counts, source-loss
decomposition, accepted workload, overflow, access/backhaul/compute utilization,
hub geometry, and W1 together. Do not use W1 alone.

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
