# Readout Contract for the Next SetRec Gate

Status: internal research contract, not yet manuscript text.

## 1. Current Claim

The next algorithm gate should not claim low-dimensional broadcast.  In the
current K=16 MEC setup, the raw UAV population state has only `3 * 16 = 48`
dimensions, while the default Set descriptor can be `4 * 64 = 256`
dimensions.  The current defensible claim is therefore:

```text
permutation-invariant, label-free, fixed-size population representation
```

The low-dimensional broadcast claim should be revisited only after a descriptor
size sweep shows that smaller descriptors preserve control quality.

## 2. Execution Mode

The current algorithm is not strict decentralized execution.  It is better
described as HAP-coordinated execution:

```text
HAP observes/collects the UAV population state,
computes a population descriptor,
and makes that descriptor available to UAV actors.
```

This is consistent with the HAP role in the MEC system, but it must be stated
separately from strict CTDE.

## 3. Descriptor and Readout

The invariant public descriptor is:

```text
xi = E({s_1, ..., s_K})
```

It must not depend on the label order of UAV rows.  The UAV actor may still
produce different actions for different UAVs through a shared local decoder:

```text
m_i = rho(s_i, public, xi)
a_i ~ pi_U(. | s_i, public, m_i)
```

The descriptor is invariant; the UAV readout is equivariant.

## 4. Diagnostic Readout Families

The next experiments must separate these cases:

| readout | memory used by the UAV decoder | role |
|---|---|---|
| `pooled` | flattened invariant descriptor only | failed baseline |
| `flat_descriptor` | ordered Flat state compressed to one broadcast descriptor | bottleneck diagnostic, failed |
| `relational` | invariant descriptor plus per-UAV equivariant token | token diagnostic |
| `cross_attention` | per-UAV query attends equivariant token memory | strongest token-assisted diagnostic so far |
| `slot_attention` | per-UAV query attends invariant latent slots only | clean EqDec candidate |

`slot_attention` is the cleanest next candidate for testing whether invariant
population slots are control-readable without relying on per-UAV token memory.

## 5. Auxiliary Losses

Control auxiliary losses are not ruled out, but they are not the next main
step.  The current Wasserstein reconstruction theorem motivates
reconstruction-aware descriptors; it does not directly justify task-aware
control auxiliaries.

Control auxiliaries may later be tested as empirical regularizers only after
the readout architecture gate is clear.

## 6. Critic Contract After the Slot-EqDec Gate

The flat critic is a diagnostic stabilizer, not a final theoretical commitment.
It isolates the actor/readout question: can an invariant descriptor or invariant
slots support equivariant UAV control at all?  The latest answer is yes.

The manuscript-level SetRec story still needs an order-insensitive value
function.  The critic should therefore remain part of the intended algorithmic
contract:

```text
V = V(public, E_V({s_1, ..., s_K}))
```

The current evidence only rules out the naive coupling choices:

- `shared_grad` lets value loss update the actor descriptor and hurts control.
- `actor_detached` and `separate` Set critics are usable diagnostics, but the
  current Set critic still lags the flat critic when paired with a Set actor.

The next critic work should repair this gap without abandoning the invariant
critic goal.  Plausible directions are a stronger invariant critic readout,
critic-only slots, value normalization / target diagnostics, and delayed or
two-stage critic fitting.  The key constraint is that critic improvements must
not corrupt the actor's control-readable descriptor.

## 7. Reset-Permutation Descriptor Bottleneck Evidence

The 2026-06-30 reset-permutation diagnostics make the actor-readout constraint
sharper.  `flat_descriptor + Flat critic` still uses the ordered Flat UAV state
and a Flat critic, but compresses the UAV population through a learned global
descriptor before broadcasting it to each UAV readout.  It averaged held-out
cost/slot `4.4281`, acceptance `42.24%`, and W1 `1212.0 m` over seeds 1/2/3.

This rules out a too-simple explanation that SetRec failed only because of
permutation invariance or because the Set critic was weak.  The global
descriptor broadcast interface is itself a bad control bottleneck for the UAV
branch.  The algorithmic claim should therefore stay with:

```text
invariant population slots + equivariant local decoder
```

and should avoid pooled/broadcast descriptor actors as main candidates.

W1 remains useful evidence: successful policies in this phase are near
`700-850 m`, while failed descriptor policies are near `1100-1250 m`.  Any W1
reconstruction auxiliary should be tested as a controlled critic/representation
regularizer, not as a replacement for the local equivariant actor readout.

## 8. Mean Baseline and the Revised Descriptor Contract

The 2026-07-01 reset-permutation results changed the baseline hierarchy.  The
fair Mean actor + Mean critic baseline averaged cost/slot `2.0991`, acceptance
`75.07%`, and W1 `684.6 m` over seeds 1/2/3.  It outperformed Flat actor +
Flat critic (`2.7326`) and all learned descriptor variants under the same
reset-permutation protocol.

This means the current environment is likely close to first-moment sufficient:
`[s_i, public, mean_uav]` is a very strong control interface.  Learned
descriptors should no longer be framed as direct replacements for mean in this
scenario.  The next descriptor contract should be residual:

```text
a_i = pi_i(s_i, public, mean_uav, C_i({s_1, ..., s_K}))
V   = V(public, mean_uav)          # first repaired actor-side gate
```

where `C_i` is a local-query, permutation-invariant slot context.  This keeps
the proven first-moment signal and asks whether learned slots add higher-order
empirical-measure information beyond the mean.

The critic evidence is also now more nuanced:

- `Mean actor + Mean critic`, seed 1: cost/slot `2.1286`.
- `Mean actor + Flat critic`, seed 1: cost/slot `2.5767`.
- `Mean actor + separate Set critic`, seed 1: cost/slot `3.2847`.
- Earlier `Flat actor + Set critic`, seed 1: cost/slot `2.2201`.

So the Set critic is not impossible, but it is fragile and interface-dependent.
For the next actor-side repair, use Mean critic first.  Then revisit an
invariant critic once the mean-residual actor can at least match or improve on
Mean-MAPPO in either the base scenario or a mean-insufficient stress scenario.
