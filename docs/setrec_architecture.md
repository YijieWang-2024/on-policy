# SetRec-MAPPO Architecture Contract

Status: diagnostic implementation contract, archived 2026-06-28.

This document defines the current algorithm design. Historical environment
decisions remain in `mec_env_port_spec.md`; current experiment commands remain in
`mec_runbook.md`.
The next readout gate is specified in `readout_contract.md`.

## 0. Archive status after recent diagnostics

The current codebase contains several useful MEC/SetRec diagnostics, but the
algorithmic contract below should not be treated as the final paper algorithm.
Recent 1.5M-step gates showed:

- the original pooled Set actor path fails robustly;
- the failure is actor-side, and more specifically UAV-readout-side;
- HAP can use a Set descriptor when the UAV branch keeps Flat information;
- giving UAVs equivariant tokens helps;
- UAV-conditioned cross-attention helps more, but still does not match the
  Flat-UAV references.

Representative held-out results:

| variant | cost/slot | accept | W1 | note |
|---|---:|---:|---:|---|
| Flat actor + Set critic | 2.2201 | 73.2% | 710.8 m | Set critic can work when actor has Flat information |
| Set-HAP + Flat-UAV actor | 2.2900 | 73.6% | 710.4 m | HAP-side Set is not the main blocker |
| Flat-HAP + Set-UAV pooled | 4.2534 | 45.5% | 1133.8 m | broken UAV Set readout |
| Flat-HAP + Set-UAV relational | 3.5498 | 55.8% | 941.5 m | tokens help, not enough |
| Flat-HAP + Set-UAV cross-attention | 3.1444 | 60.8% | 865.3 m | best Set-UAV result so far, still short |

The immediate archive decision is to pause further algorithm branches that
would import an ordered Flat teacher, heuristic behavior cloning, or more
auxiliary-control hybrids.  Those may improve engineering performance, but
they do not yet answer the paper's core claim: how a learned public,
order-invariant representation supports control in this heterogeneous
HAP/UAV setting.  The next phase should first rewrite the theory/architecture
contract: what representation is public, what local/equivariant decoder queries
are allowed, and what minimal trainable architecture follows from that claim.

## 1. State and information contract

For UAV `i`, the normalized physical state atom is

```text
s_i = [x_i / L_x, y_i / L_y, Q_i / Q_U_max].
```

The public HAP/demand state is

```text
p_phys = [y_x / L_x, y_y / L_y, Q_H / Q_H_max,
          c_x / L_x, c_y / L_y, c_dot_x / V_ref, c_dot_y / V_ref].
```

The fixed-width fleet/resource context is

```text
r = [K / K_ref,
     W_ac,i / W_ac,total,
     W_bh,i / W_bh,total,
     K C_U / (K C_U + C_H),
     C_H / (K C_U + C_H),
     (K C_U + C_H) / D_ref].
```

`K_ref` is the scenario's native fleet size before a fleet-size override.
The public state is `p = [p_phys(7), r(6)]`. Making cardinality and resource
scaling explicit is required because normalized attention pooling alone cannot
in general recover multiplicity.

The environment emits a 17-dimensional local row:

```text
[role, own(3), p(13)].
```

The MEC runner constructs one canonical centralized state and repeats it across
the `K+1` buffer rows:

```text
[p(13), s_1(3), ..., s_K(3)].
```

No population descriptor is emitted by the environment. Mean, flat, and set
representations are all computed inside the policy from the same UAV atoms.
The physical environment and queue/communication equations are unchanged.

## 2. Comparison architectures

Four explicit architecture switches are retained.

| CLI value | Actor information | Critic information | Purpose |
|---|---|---|---|
| `legacy_mean` | historical shared trunk over `own + p + UAV mean` | historical ordered flattened 14-D rows | old-checkpoint compatibility only |
| `mean` | HAP: `p + mean`; UAV: `s_i + p + mean` | `p + mean` | aligned low-capacity descriptor baseline |
| `flat` | HAP: `p + s_1:K`; UAV: `s_i + p + s_1:K` | `p + s_1:K` | information-complete, fixed-K, label-sensitive baseline |
| `set` | HAP: `p + xi`; UAV: `s_i + p + xi` | `p + xi` | proposed invariant/equivariant architecture |

The aligned `mean`, `flat`, and `set` variants use the same role-specific actor
heads, team critic, PPO batching, and optimizer ownership. Only the population
representation changes. `flat` is an information-complete fixed-K control
baseline, but is label-sensitive and not a decentralized-execution policy.

Historical mean results remain useful engineering references, but they are not
a clean representation ablation because their actor trunk and ordered critic
differ from the aligned variants.

## 3. Population encoder

There is one public UAV population encoder shared by the HAP actor, all UAV
actors, and the team critic:

```text
{s_i}
  -> shared AtomMLP
  -> two Set Attention Blocks
  -> Pooling by Multihead Attention with M learned population seeds
  -> latent Set Attention Block
  -> xi in R^(M x d)
```

Initial configuration:

```text
d = 64, heads = 4, M = 4,
element SAB blocks = 2, latent SAB blocks = 1.
```

The encoder may create permutation-equivariant element tokens internally.  The
original phase-1 contract exposed only the permutation-invariant descriptor
`xi`; later diagnostics also exposed tokens to the UAV readout.  That exposure
is best understood as a decoder/readout diagnostic, not yet as a settled final
paper contract.

## 4. Actor and critic

All three aligned variants use the same role-specific fusion readout contract.
The readouts are intentionally optimization-equivalent to the legacy MAPPO
`MLPBase`: optional input LayerNorm, configured orthogonal/Xavier
initialization, configured `layer_N` hidden blocks, and per-hidden-layer
LayerNorm.

```text
HAP: f_H([p, representation]) -> velocity distribution
UAV: f_U([s_i, p, representation]) -> velocity + beta distributions
```

All UAVs share the same UAV fusion network and action heads. For `set`, the
invariant `M x d` descriptor is flattened only after invariant pooling; its
latent slot order is learned and fixed, not tied to UAV labels.

There is one centralized team critic:

```text
V([p, representation]).
```

### Phase-1 gate update, 2026-06-26

The first 350-slot `mean/flat/set` sweep failed because the aligned readout did
not follow this `MLPBase` contract.  After repairing `FusionMLP`, a seed-1
350-slot gate recovered normal training:

| architecture | held-out cost/slot | accept | W1 diagnostic |
|---|---:|---:|---:|
| mean, fixed readout | 3.2557 | 59.4% | 919.1 m |
| flat, fixed readout | 2.9574 | 62.6% | 883.9 m |

This means the PPO framework and environment output contract are not the main
cause of the previous collapse.  Decoder, Sinkhorn reconstruction, and PPG
auxiliary training should still wait until the fixed Set policy passes the same
one-seed gate and the formal 3-seed comparison is rerun.

Its scalar value is repeated across the `K+1` buffer rows for compatibility with
the existing shared runner. The value gradient is stopped at `xi` in phase 1:
the policy/representation optimizer owns the population encoder; the critic
optimizer owns only the critic readout.

## 5. Set reconstruction

Reconstruction is phase 2, after Set-MAPPO passes structural and short-training
tests. The public decoder reads only `xi` and target fleet size `K`:

```text
D(xi, K) -> {s_hat_1, ..., s_hat_K}.
```

It must not read original UAV states or intermediate element tokens. The loss is
the debiased Sinkhorn divergence between the normalized empirical measures:

```text
mu     = (1/K) sum_i delta_(s_i)
mu_hat = (1/K) sum_j delta_(s_hat_j)
L_rec  = S_epsilon(mu, mu_hat).
```

Reconstruction must preserve permutation symmetry. A row-wise MSE loss is not
valid for unordered UAV atoms because it penalizes equivalent permutations.
The minimum viable loss is symmetric Chamfer distance; the final theory path
can promote this to debiased Sinkhorn divergence after the staged training
contract is validated.

Reconstruction should not be treated as just another term inside every PPO
actor update. Historical UAV atom sets remain valid for representation
learning after PPO's on-policy window expires, while actor and critic control
losses still require on-policy PPO data. Therefore reconstruction uses a
staged/PPG-style contract: replay or pretrain the encoder-decoder, initialize
PPO from that representation, and use KL-protected auxiliary phases if the
encoder continues to move during RL.

## 6. Training phases

Phase 1, implemented first:

```text
policy optimizer: population encoder + HAP actor + UAV actor
critic optimizer: critic readout only
losses: role-balanced PPO and value regression
```

Diagnostic variants now available:

- `--mec_set_encoder_type mean_pool` replaces the latent-slot encoder with a
  simpler atom-MLP + self-attention + invariant mean-pooling encoder.
- `--mec_set_encoder_type flat_mlp` replaces the set encoder with an ordered
  fixed-K MLP descriptor. It is a diagnostic closest to the Flat baseline while
  retaining the Set-style encoder/readout interface.
- `--mec_set_actor_encoder separate` gives the HAP actor and UAV actor
  independent population encoders. It is a diagnostic for role-gradient
  interference and requires `--mec_set_critic_encoder separate`.
- `--mec_set_critic_encoder separate` gives the critic an independent
  population encoder trained by the value loss.
- `--mec_set_critic_encoder shared_grad` lets the critic value loss update the
  actor's shared population encoder too. This is a diagnostic path rather than
  the preferred final algorithm.
- `--mec_set_actor_context relational` gives each UAV its equivariant
  self-attention token in addition to the invariant pooled descriptor.
- `--mec_set_actor_context slot_attention` lets each UAV form a local query
  from `[s_i, p]` and attend only to invariant latent population slots.  This
  is the clean EqDec candidate for testing whether the public descriptor is
  control-readable without per-UAV token memory.
- `--mec_set_actor_context cross_attention` lets each UAV form a local query
  from `[s_i, p, token_i]` and read from the equivariant token memory before
  fusing `[s_i, p, xi, token_i, context_i]`. This is the strongest Set-UAV
  diagnostic so far, but it remains below the Flat-UAV references.

Phase 2, revised after the first Chamfer MVP:

```text
offline/replay stage:
    collect canonical UAV atoms from random/heuristic/policy rollouts
    train encoder + decoder with permutation-invariant Chamfer/Sinkhorn loss

PPO initialization:
    initialize Set actor encoder from the pretrained encoder
    optionally freeze the encoder for a short actor/critic warm-up

auxiliary phase, if continued during RL:
    auxiliary loss = lambda_rec * L_rec
                   + beta_KL * KL(pi_reference || pi_current)
    updated parameters = encoder + decoder (+ readouts only if KL-protected)
    then refit the critic to the updated descriptor
```

## 7. Required tests

Before training:

1. Permuting UAV rows leaves `xi`, HAP action distribution, and team value unchanged.
2. The UAV action distributions follow the same permutation.
3. Default Set critic parameters do not include the public actor encoder.
4. Default value loss produces no actor-encoder gradients.
5. With `mec_set_critic_encoder=separate`, value loss does produce gradients
   for the critic-owned encoder and not for the actor encoder.
6. With `mec_set_critic_encoder=shared_grad`, actor and critic optimizers both
   contain the shared actor encoder and value loss reaches that encoder.
7. With `mec_set_encoder_type=mean_pool`, the descriptor remains invariant and
   UAV actions remain equivariant.
8. With `mec_set_encoder_type=flat_mlp`, the descriptor is fixed-size,
   order-sensitive, and rejects relational UAV actor context.
9. PPO forward/evaluate log probabilities agree before an update.
10. PPO minibatches preserve complete `(time, environment, K+1 agents)` groups.

## 8. Experiment order

1. Preserve completed old mean results under the `legacy_mean` label.
2. Smoke-test aligned `mean`, `flat`, and `set`.
3. Run new equal-budget, three-seed `mean/flat/set` comparisons.
4. Add reconstruction and compare Set-MAPPO against SetRec-MAPPO.
5. Only after stable three-seed evidence, run long training, cross-K evaluation,
   and final paper experiments.

Archive update, 2026-06-28: this original experiment order is superseded for
the current branch.  The available evidence is no longer pointing to "add one
more auxiliary trick"; it points to a mismatch between the theoretical
representation claim and the actor decoder/readout needed for control.  Do not
advance to final paper experiments or teacher/imitation warmups from this
branch without first rewriting that contract.

## 9. Phase-1 gate result (2026-06-26)

The 350-slot, 896k-step, three-seed aligned comparison is complete. All three
architectures train without numerical failure, but none passes the control
quality gate:

```text
architecture  held-out cost/slot  accept   HAP-freeze
mean          4.6491 +/- 0.1292    38.9%    -0.3% +/- 0.5%
flat          4.5822 +/- 0.0900    39.7%     approximately 0%
set           4.6230 +/- 0.0669    39.1%     approximately 0%
heuristic     2.0694               73.7%     n/a
```

Set has the lowest cross-seed cost variance, so its implementation is not
numerically unstable. However, freezing its HAP does not hurt performance and
freezing learned UAV motion slightly improves representative checkpoints.
Reconstruction is therefore deferred.

The immediate phase-1 repair is to make the aligned actor and critic readouts
optimization-equivalent to the proven legacy trunk. In particular,
`FusionMLP` currently does not honor `use_feature_normalization`,
`use_orthogonal`, or `layer_N`, and omits the per-layer normalization used by
`MLPBase`. After this contract is repaired, run a one-seed gate against
`legacy_mean`, followed by the full three-seed comparison only if useful HAP
and UAV motion returns.

## 10. Set diagnosis update (2026-06-26)

The `FusionMLP` contract was repaired and Mean/Flat recovered. Set did not:
with `mec_logstd_init=-1.2`, the 448k high-variance Set gate stayed near
validation reward `-1531`, while Flat reached `-874.83` by 448k.

Three 112k Set diagnostics were run:

```text
variant                  best validation
separate critic encoder  -1531.60
small Set encoder        -1531.20
relational UAV context   -1534.83
```

Current interpretation: Set's issue is not one missing PPO readout trick, too
many layers, critic encoder detachment, or the pooled-only UAV input alone.
Pure PPO is not shaping the attention population encoder into a useful control
representation. The next Set-specific step should therefore be auxiliary
representation learning: start the decoder/reconstruction path before spending
large compute on pure Set-MAPPO.

## 11. Simple encoder / optimizer-path update (2026-06-27)

The Mean/Flat/Set sharing difference was isolated further. Mean/Flat have no
learnable shared trunk across HAP actor, UAV actor, and critic; Set does.
Therefore a simpler `mean_pool` population encoder was tested with three
critic update paths:

```text
encoder: UAV atoms -> atom MLP -> multi-head self-attention -> mean pool

critic path             best validation, 112k
actor_detached          -1540.76
separate critic encoder -1539.08
shared_grad             -1546.69
```

This did not recover Set learning. Directly allowing value loss to update the
shared actor encoder also did not help. The evidence now supports treating the
decoder/reconstruction/PPG stage as a necessary representation-learning
component, not merely a later embellishment after pure Set-MAPPO.

## 12. HAP/UAV actor encoder sharing check (2026-06-27)

The HAP/UAV actor sharing hypothesis was checked with a new diagnostic:

```text
--mec_set_encoder_type mean_pool
--mec_set_actor_encoder separate
--mec_set_critic_encoder separate
```

This gives HAP actor, UAV actor, and critic three disjoint population encoders.
Structural tests verify that major-only actor loss reaches only the HAP encoder
and minor-only actor loss reaches only the UAV encoder.

The 112k, 350-slot, seed-1 gate did not recover:

```text
best validation:   -1567.28
held-out cost/slot: 4.5959
accept:             39.5%
W1:                 1287.3 m
HAP-freeze:         -0.0%
```

This weakens the hypothesis that simple HAP/UAV encoder sharing is the main
failure mode. Continue toward auxiliary representation learning rather than
expanding this pure Set-MAPPO branch.

## 13. Full 896k confirmation gate (2026-06-27)

Before the full confirmation gate, the population encoder implementation was
made numerically consistent with the MAPPO readout contract: atom encoding now
uses optional feature normalization and `MLPLayer`, and attention / FFN blocks
are initialized according to `use_orthogonal`.

The full budget was 350 slots, 896k environment steps, seed 1, and
`mec_logstd_init=-1.2`.

```text
architecture                      best validation   held-out cost   accept   W1     HAP-freeze
Mean                              -766.40           2.2695          73.5%    702m   +28.2%
Flat                              -815.46           2.3591          73.0%    729m   +5.0%
Set mean_pool actor_detached      -1478.39          4.4842          41.1%   1220m   -0.1%
Set mean_pool separate critic     -1525.51          4.5496          40.3%   1249m   -0.1%
Set mean_pool shared_grad         -1511.63          4.3408          43.2%   1209m   -0.1%
```

This confirms that Set mean-pool is not simply slower under the same pure
MAPPO objective.  Even with a full training budget and the encoder numerical
contract repaired, it does not approach Mean/Flat and does not learn a
load-bearing HAP trajectory.  The next architecture step should be auxiliary
representation learning: first a cheap decoder/reconstruction phase, then
Sinkhorn/PPG scheduling once that signal is validated.

## 14. Flat-MLP encoder diagnostic (2026-06-27)

To check whether the remaining failure was specifically caused by the
self-attention / mean-pool implementation, a plain ordered fixed-K encoder was
added:

```text
--mec_set_encoder_type flat_mlp
```

This encoder flattens the UAV atoms in their current order and maps them through
a MAPPO-style `MLPLayer` into a fixed-width descriptor. It is intentionally not
permutation invariant and supports only pooled actor context. The first gate
used the no-sharing setting:

```text
--mec_set_encoder_type flat_mlp
--mec_set_actor_encoder separate
--mec_set_critic_encoder separate
```

The 400k gate used 350 slots, seed 1, `mec_logstd_init=-1.2`, 403.2k
environment steps / 72 PPO updates, validation seed 1000, and held-out seed
100000 with stride 13 over 24 episodes.

```text
best validation:    -1470.92 at 403.2k steps
held-out cost/slot:  4.1688
accept:              45.6%
W1:                  1166 m
HAP-freeze:          -0.8%
```

This is slightly better than mean-pool Set but still far behind Mean/Flat and
still does not learn a load-bearing HAP trajectory. The failure is therefore
unlikely to be just a self-attention or mean-pool implementation bug. The
remaining suspect is the learnable two-stage population descriptor/readout
interface under pure MAPPO, so the next main step remains auxiliary
representation learning rather than more pure Set-MAPPO sweeps.

## 15. Chamfer reconstruction MVP (2026-06-27)

The first reconstruction auxiliary used `mean_pool` and a symmetric squared
Chamfer set loss:

```text
--mec_policy_arch set
--mec_set_encoder_type mean_pool
--mec_set_reconstruction_coef 0.1
```

This was intentionally a minimal joint-loss diagnostic: the actor optimizer
updated the population encoder and decoder with PPO actor loss plus
`lambda_rec * L_chamfer`.

The symmetry contract passed structural tests: Chamfer loss is invariant to
target atom order, and the mean-pool reconstruction loss is unchanged when the
input UAV rows are permuted.

The 400k control gate did not recover:

```text
best validation:     -1511.02 at 201.6k steps
held-out cost/slot:   4.6367
accept:               39.0%
W1:                   1251 m
HAP-freeze:           -0.0%
```

The reconstruction loss itself did learn, decreasing from `0.2840` at 5.6k
steps to `0.0103` at 397.6k steps. This means the auxiliary task is learnable,
but naive joint optimization is not enough to improve control. Future SetRec
runs should use replay/pretraining and KL-protected auxiliary phases rather
than scaling this direct joint-loss path.

## 16. Staged reconstruction pretraining result (2026-06-27)

The staged pretraining path is now implemented:

```text
python -m onpolicy.scripts.train.pretrain_mec_set_reconstruction
--mec_set_pretrained_actor <actor.pt>
--mec_set_freeze_pretrained_encoder_updates <N>
```

`--mec_set_pretrained_actor` is intentionally separate from `--model_dir`.
It loads only Set representation tensors from a pretrained actor, not critic
weights, readouts, optimizer state, or ValueNorm. This keeps reconstruction
pretraining distinct from resumable PPO checkpoints.

The first staged gates show that pure geometry reconstruction is not sufficient:

```text
mean_pool pretrain:
    replay samples:           22,400
    Chamfer loss:             0.0383 -> 0.0020
    PPO freeze-12 cost/slot:  4.7404
    PPO freeze-12 HAP-freeze: -0.3%
    PPO no-freeze best val:   -1566.83 at 112k

latent_slots pretrain:
    replay samples:           22,400
    Chamfer loss:             0.0441 -> 0.0017
    PPO no-freeze cost/slot:  4.6197
    PPO no-freeze HAP-freeze: -0.2%
```

Both encoders learn the reconstruction task, but neither yields control
recovery. The archive decision is to pause new algorithm branches rather than
continue toward policy distillation, heuristic behavior cloning, or more
auxiliary-control hybrids. Those directions would test different training
problems, but they would not explain the current failure: introducing a
learnable descriptor repeatedly prevents the otherwise successful Mean/Flat
PPO path from learning a load-bearing controller.

The next scientifically useful step is to analyze the descriptor/readout
interface itself: gradient scales, feature statistics, action sensitivity to
individual UAV atoms, and optimizer/loss coupling across Mean, Flat, mean_pool,
flat_mlp, and latent_slots. A new algorithmic stage should wait until that
failure mode is identified.

## 17. Slot-EqDec readout recovers the Set actor path (2026-06-29)

The next gate changed the UAV actor readout rather than adding another
auxiliary loss.  `--mec_set_actor_context slot_attention` lets each UAV form a
query from `[s_i, public]` and attend to invariant population memory.  With
`mean_pool` the memory is a single invariant slot; with `latent_slots` the
memory is the learned invariant slot set.  This keeps the descriptor invariant
and the UAV decoder equivariant.

Protocol: v6 HAP-loadbearing, 350-slot episodes, 1.5M environment steps,
validation seed 1000 over 24 episodes, held-out seed 100000 with stride 13
over 24 episodes.

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| mean_pool Slot-EqDec + Flat critic, seed 1 | -867.14 | 1.2096M | 2.6655 | 67.4% | 791.7 m | +7.0% |
| mean_pool Slot-EqDec + Flat critic, seed 2 | -928.45 | 1.1088M | 2.9208 | 64.5% | 863.0 m | +4.5% |
| mean_pool Slot-EqDec + Flat critic, seed 3 | -860.95 | 1.4112M | 2.5813 | 68.8% | 810.7 m | +16.2% |
| latent_slots Slot-EqDec + Flat critic, seed 1 | -877.59 | 1.4952M | 2.6018 | 68.0% | 805.8 m | +7.1% |
| mean_pool Slot-EqDec + Set critic separate, seed 1 | -965.55 | 1.4112M | 2.8969 | 65.0% | 875.4 m | +0.8% |
| mean_pool Slot-EqDec + Set critic shared_grad, seed 1 | -1068.07 | 1.2096M | 3.1975 | 60.0% | 941.2 m | +16.6% |

For the three mean_pool Slot-EqDec + Flat critic seeds, held-out cost/slot is
`2.7225` on average, with average acceptance `66.9%` and average W1 `821.8 m`.
This is the first Set actor result that is both substantially better than the
pooled / relational / cross-attention Set-UAV readout gates and replicated
across multiple seeds.  The latent-slot seed-1 result is especially important:
learned invariant slots are now control-readable when paired with the local
query decoder.

The critic interpretation must be careful.  The Flat critic is not the final
SetRec claim; it is a stabilizing diagnostic that isolates whether the actor
descriptor/readout can work.  The theory-facing algorithm still needs an
order-insensitive value function, because a value estimate should not depend on
the arbitrary UAV row order.  The current negative evidence is narrower:
naive `shared_grad` critic coupling hurts, and the current separate Set critic
is still weaker than the Flat critic when paired with a Set actor.  This does
not kill the invariant critic goal; it says the critic must be repaired without
letting critic gradients corrupt the actor's control-readable descriptor.

Current working conclusion:

- The next manuscript algorithm should be based on invariant population slots
  plus an equivariant local readout, not pooled descriptor concatenation.
- Do not claim low-dimensional broadcast yet; the present claim is fixed-size,
  permutation-invariant / label-free representation.
- Do not abandon the Set critic objective.  Treat Flat critic runs as
  actor-readout evidence, then run targeted invariant-critic repair experiments.
- Do not revive control auxiliaries as the main story until the readout and
  critic contracts are aligned.
