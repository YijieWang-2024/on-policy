# SetRec-MAPPO Architecture Contract

Status: implementation contract, updated 2026-06-25.

This document defines the current algorithm design. Historical environment
decisions remain in `mec_env_port_spec.md`; current experiment commands remain in
`mec_runbook.md`.

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

The encoder may create permutation-equivariant element tokens internally. They
are not exposed to the actor, critic, or decoder. The only public population
interface is the permutation-invariant descriptor `xi`.

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

Reconstruction will use a PPG-style auxiliary phase with policy KL protection,
not a second optimizer that independently updates the shared encoder.

## 6. Training phases

Phase 1, implemented first:

```text
policy optimizer: population encoder + HAP actor + UAV actor
critic optimizer: critic readout only
losses: role-balanced PPO and value regression
```

Phase 2:

```text
auxiliary loss = lambda_rec * L_rec
               + beta_KL * KL(pi_reference || pi_current)
updated parameters = encoder + decoder + actor readouts
then refit the critic to the updated descriptor
```

## 7. Required tests

Before training:

1. Permuting UAV rows leaves `xi`, HAP action distribution, and team value unchanged.
2. The UAV action distributions follow the same permutation.
3. Set critic parameters do not include the public encoder.
4. Value loss produces no encoder gradients.
5. PPO forward/evaluate log probabilities agree before an update.
6. PPO minibatches preserve complete `(time, environment, K+1 agents)` groups.

## 8. Experiment order

1. Preserve completed old mean results under the `legacy_mean` label.
2. Smoke-test aligned `mean`, `flat`, and `set`.
3. Run new equal-budget, three-seed `mean/flat/set` comparisons.
4. Add reconstruction and compare Set-MAPPO against SetRec-MAPPO.
5. Only after stable three-seed evidence, run long training, cross-K evaluation,
   and final paper experiments.

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
