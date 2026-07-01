# Work Notes

## 2026-06-30 - Reset-permutation isolation and flat-descriptor held-out

The reset-permutation round is now complete for the main actor baselines and
for the flat-descriptor bottleneck diagnostic.  The scenario is
`v6_hap_loadbearing` with `initial_deploy.random_uav_permutation: true`,
350-slot episodes, 1.5M environment steps, validation seed 1000, and held-out
seed 100000 with stride 13 over 24 episodes.

The new B diagnostic is:

```text
mec_policy_arch=flat_descriptor
actor: flat ordered UAV state -> FusionMLP descriptor -> pooled PopulationActor readout
critic: flat ordered centralized critic
```

This actor is still order-sensitive and keeps the Flat critic, so it isolates
whether the descriptor/readout bottleneck itself is harmful.

| variant | held-out cost/slot mean | accept mean | W1 mean | HAP-freeze mean | validation cost mean |
|---|---:|---:|---:|---:|---:|
| Flat actor + Flat critic | 2.7326 | 66.55% | 800.6 m | +12.9% | 2.5979 |
| Latent Slot-EqDec actor + Flat critic | 3.1458 | 60.54% | 972.8 m | +8.1% | 3.0125 |
| Latent Slot-EqDec actor + invariant Set critic | 4.3119 | 44.87% | 1145.4 m | -0.7% | 4.0225 |
| Flat descriptor actor + Flat critic | 4.4281 | 42.24% | 1212.0 m | +0.3% | 4.1802 |

Flat descriptor actor + Flat critic held-out rows:

| seed | selected step | validation cost | held-out cost/slot | accept | W1 | HAP-freeze |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 806400 | 4.1728 | 4.3981 | 42.4% | 1219.6 m | -0.4% |
| 2 | 1495200 | 4.2223 | 4.3427 | 43.7% | 1178.2 m | +2.1% |
| 3 | 302400 | 4.1454 | 4.5433 | 40.6% | 1238.3 m | -0.7% |

Artifacts:

```text
eval_outputs/resetperm_flat_descriptor_1500k/resetperm1500_flat_descriptor.summary.json
eval_outputs/resetperm_flat_descriptor_1500k/resetperm1500_flat_descriptor_flatcrit_seed{1,2,3}.json
training_logs/resetperm1500_flat_descriptor_flatcrit_seed{1,2,3}.heldout.log
```

Interpretation:

- Reset permutation was the right fairness change.  It removes the fixed
  initialization-row shortcut, but it does not make Flat actor + Flat critic
  collapse; Flat remains the strongest reset-permutation reference.
- Latent Slot-EqDec + Flat critic is stable and much better than the old pooled
  descriptor path, but under reset permutation it still trails Flat by about
  `0.41` cost/slot and about `172 m` W1.
- The invariant Set critic is a real bottleneck.  `latent Slot-EqDec + Set
  critic` is much worse than the same actor with Flat critic.
- The B diagnostic is the strongest evidence that the main actor failure is
  not merely permutation invariance, Set encoder complexity, or Set critic
  weakness.  A learned global descriptor that is broadcast to each UAV readout
  breaks control even when the descriptor is produced from the ordered Flat
  state and the critic is Flat.
- The historical role-isolation results fit the same story: HAP-side Set is
  acceptable when UAVs keep Flat information, while UAV-side pooled Set readout
  fails.  Query-conditioned/equivariant UAV readout is the useful repair.
- W1 is not just an aesthetic metric.  Good policies stay around `700-850 m`,
  while failed pooled/descriptor policies are around `1100-1250 m`.  However,
  reconstruction loss should not be reintroduced as a main actor-side bottleneck
  until the readout and critic contracts are stable.

Current research decision:

- Keep reset-level UAV permutation as the default experimental setting.
- Stop expanding pooled/broadcast descriptor actor variants.
- Keep Slot-EqDec as the actor-side theory candidate: invariant population
  slots plus equivariant local UAV decoder.
- Keep Flat critic as a diagnostic stabilizer only, not as the final SetRec
  claim.
- Repair the invariant critic as a separate subproblem: critic-only slots,
  stronger invariant value readout, delayed critic fitting, and no shared
  gradient path back into the actor descriptor.
- If W1 reconstruction is used next, prefer critic/representation diagnostics
  or a carefully weighted critic-side auxiliary.  Do not make it the main actor
  representation bottleneck again.

## 2026-06-29 - Slot-EqDec recovers the Set actor, critic remains open

The latest work separated two questions that had been entangled:

1. Can an invariant Set descriptor / slot memory support UAV control if the UAV
   readout is query-conditioned and equivariant?
2. Can the value function also be estimated from an order-insensitive Set
   representation without damaging the actor?

The first answer is now positive.  `--mec_set_actor_context slot_attention`
uses a UAV-local query from `[s_i, public]` and reads only invariant population
memory.  It is cleaner than token `cross_attention` because it does not rely on
per-UAV token memory as the main actor interface.

1.5M, 350-slot, validation seed 1000 / held-out seed 100000 results:

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| mean_pool Slot-EqDec + Flat critic, seed 1 | -867.14 | 1.2096M | 2.6655 | 67.4% | 791.7 m | +7.0% |
| mean_pool Slot-EqDec + Flat critic, seed 2 | -928.45 | 1.1088M | 2.9208 | 64.5% | 863.0 m | +4.5% |
| mean_pool Slot-EqDec + Flat critic, seed 3 | -860.95 | 1.4112M | 2.5813 | 68.8% | 810.7 m | +16.2% |
| latent_slots Slot-EqDec + Flat critic, seed 1 | -877.59 | 1.4952M | 2.6018 | 68.0% | 805.8 m | +7.1% |
| mean_pool Slot-EqDec + Set critic separate, seed 1 | -965.55 | 1.4112M | 2.8969 | 65.0% | 875.4 m | +0.8% |
| mean_pool Slot-EqDec + Set critic shared_grad, seed 1 | -1068.07 | 1.2096M | 3.1975 | 60.0% | 941.2 m | +16.6% |

Mean-pool Slot-EqDec + Flat critic over seeds 1/2/3 averages cost/slot
`2.7225`, acceptance `66.9%`, and W1 `821.8 m`.  This is not yet as strong as
the best Flat-UAV references, but it is a genuine recovery of the Set actor
path and is no longer a one-seed accident.  The latent-slot seed-1 result shows
that learned invariant slots can also be control-readable when decoded through
the local equivariant query.

The critic lesson should not be overstated.  Flat critic is a diagnostic
stabilizer, not the final SetRec value-function claim.  The paper still wants a
value estimate such as `V(public, E_V({s_i}))`, because arbitrary UAV row order
should not change the value.  What the latest results say is narrower:

- `shared_grad` is harmful because value loss can distort the actor descriptor.
- `separate` Set critic improves over the original Set critic but still lags
  the Flat critic in this paired Set-actor setting.
- The invariant critic should be repaired, not abandoned.

Next plan:

1. Run `latent_slots + slot_attention + flat critic` seeds 2 and 3 to check
   whether the true latent-slot version is stable.
2. Design an invariant critic repair gate: stronger Set value readout,
   critic-only slots, delayed critic fitting, or a stop-gradient interface that
   preserves actor descriptor quality.
3. Keep manuscript wording away from "low-dimensional broadcast" for now.  The
   defensible claim is fixed-size, permutation-invariant representation with
   equivariant local readout under HAP-coordinated execution.
4. Do not put control auxiliary losses back on the main path until the actor
   readout and invariant critic contracts are both resolved.

## 2026-06-28 - Set-UAV cross-attention readout 1.5M diagnostics

Implemented `--mec_set_actor_context cross_attention` for Set actors and the
`flat_hap_set_uav` role-hybrid actor.  The encoder still produces the invariant
public descriptor `z=f({uavs})` plus equivariant UAV tokens; the new
query-conditioned computation is only in the UAV decoder/readout.  Each UAV
forms a local query from its own state, public state, and corresponding token,
attends over the encoded UAV token memory, then fuses:

```text
[uav_i, public, invariant_descriptor, token_i, attended_context_i]
```

Verification:

```text
33 passed
git diff --check passed with CRLF warnings only
```

1.5M seed-1 comparison, all with `flat_hap_set_uav`, `mean_pool` Set encoder,
Flat critic, 350-slot episodes, validation seed 1000 over 24 episodes,
held-out seed 100000 stride 13 over 24 episodes:

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| pooled Set-UAV readout | -1440.57 | 1.4112M | 4.2534 | 45.5% | 1133.8 m | -0.7% |
| relational token Set-UAV readout | -1269.49 | 1.4952M | 3.5498 | 55.8% | 941.5 m | +6.7% |
| cross-attention Set-UAV readout | -1061.36 | 1.4952M | 3.1444 | 60.8% | 865.3 m | +20.8% |

Artifacts:

```text
scripts/run_v6_cross_attention_readout_1500k.ps1
eval_outputs/cross_attention_readout_1500k/diag1500_crossreadout.summary.json
training_logs/diag1500_crossreadout_*.log
```

Interpretation:

- The original pooled Set-UAV path is confirmed as the broken mechanism.
- Relational/equivariant tokens help, and cross-attention helps more.  This
  supports the diagnosis that the UAV actor needs a local/equivariant decoder
  over the population representation, not only a shared pooled descriptor.
- Cross-attention improves held-out cost/slot by about 11.4% versus relational
  and about 26.1% versus pooled.
- Cross-attention still does not reach the earlier references:
  `Set-HAP + Flat-UAV` cost/slot 2.2900 and `Flat actor + Set critic` 2.2201.
  So the root cause is improved but not fully solved.
- The high HAP-freeze sensitivity (+20.8%) suggests the cross-attention UAV
  branch learned a more HAP-dependent policy.  That is a useful sign, but it
  also points to HAP/critic coordination as the next bottleneck.

Archive decision:

1. Make cross-attention the main Set-UAV branch and stop spending budget on
   pooled Set-UAV PPO sweeps.
2. Do not proceed to Flat-teacher warmup, behavior cloning, or heuristic
   imitation. Those may improve a controller, but they move away from explaining
   why the proposed public descriptor architecture should work.
3. Pause new algorithm branches and treat this as an unresolved theory/architecture
   mismatch: local/equivariant readout helps, but the current learned descriptor
   stack still does not match the Flat-UAV references.
4. The next useful work is a written redesign of the algorithmic contract:
   what representation is claimed by the theory, what information the decoder
   is allowed to query, and what minimal architecture can make that claim
   trainable without importing a Flat teacher.

## 2026-06-28 - Actor role-isolation 1.5M diagnostics

The grouped actor binding concern was converted into tests before running
more PPO.  Added coverage for:

- `_features()` preserving `[HAP, UAV1, ...]` row binding after grouped
  reshape/flatten;
- role-specific `evaluate_actions()` log-prob selection, including the fact
  that HAP log-prob ignores the dummy beta while UAV log-prob uses beta;
- hybrid actor contracts and gradient routing for the Set-UAV branch.

Verification:

```text
30 passed
git diff --check passed
```

We then ran three 1.5M-step role-isolation diagnostics in parallel:

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| Set-HAP + Flat-UAV actor, Flat critic | -765.42 | 1.1088M | 2.2900 | 73.6% | 710.4 m | +3.5% |
| Flat-HAP + Set-UAV actor, Flat critic | -1440.57 | 1.4112M | 4.2534 | 45.5% | 1133.8 m | -0.7% |
| Flat-HAP + Set-UAV relational actor, Flat critic | -1269.49 | 1.4952M | 3.5498 | 55.8% | 941.5 m | +6.7% |

Artifacts:

```text
scripts/run_v6_actor_role_isolation_1500k.ps1
eval_outputs/actor_role_isolation_1500k/diag1500_roleiso.summary.json
training_logs/diag1500_roleiso_*.log
```

Interpretation:

- The grouped tensor binding and `is_major` log-prob split are very unlikely
  to be the root cause.  Tests now cover the critical alignment assumptions.
- The failure is not primarily HAP-side.  A Set HAP branch with Flat UAV
  branch learns well.
- The robust bottleneck is the UAV actor readout from Set information.  When
  the UAV branch uses a pooled Set descriptor, training remains in the failed
  regime even with 1.5M steps.
- Relational/equivariant UAV tokens help substantially, but they still do not
  close the gap.  This says the right direction is stronger UAV-local
  relation/readout alignment, not more pure PPO on a single global latent.

Next algorithm direction at that point was to keep the invariant/public
descriptor for global coordination, but replace the UAV readout with a
control-readable relational module.  Cross-attention was then tested and did
improve the Set-UAV branch, but it did not close the gap to the Flat-UAV
references.  This should now be framed as partial evidence about the failure
mode, not as the final paper algorithm.

## 2026-06-28 - Public descriptor diagnostics extended to 1.5M

The 400k gate was too short for one of the three diagnostic questions.  We
reran the same three variants to 1.5M environment steps, in parallel with
three Python workers.  The machine handled the parallel run cleanly; all three
training stderr logs stayed empty.  The PowerShell wrapper misclassified the
run as failed because this Windows `Start-Process` path can leave `ExitCode`
empty even when artifacts exist; held-out evaluation was therefore completed
manually from `models\best`, and the wrapper was patched to accept
artifact-complete runs with empty `ExitCode`.

Protocol: 350 slots, seed 1, validation seed 1000 over 24 episodes, held-out
seed 100000 with stride 13 over 24 episodes.

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| sort_flat actor + sort_flat critic | -829.19 | 1.3104M | 2.5370 | 69.5% | 800.0 m | +8.6% |
| Set mean_pool actor + Flat critic | -1417.86 | 1.4952M | 4.0030 | 48.4% | 1087.8 m | +8.4% |
| Flat actor + Set mean_pool critic | -750.21 | 1.4112M | 2.2201 | 73.2% | 710.8 m | +11.2% |

Artifacts:

```text
eval_outputs/descriptor_diagnostics_400k/diag1500_desc.summary.json
training_logs/diag1500_desc_*.log
onpolicy/scripts/results/MEC/v6_hap_loadbearing/mappo/diag1500_desc_*/run1/models/best
```

Updated interpretation:

- The public descriptor idea is not dead.  `sort_flat` is public,
  order-invariant, information-preserving, and improves substantially with a
  longer budget.
- The 400k conclusion that a Set critic independently poisons MAPPO was too
  strong.  `Flat actor + Set critic` needs more steps, but by 1.5M it becomes
  the best of these three diagnostics.  A Set critic can work when the actor
  still receives the ordered Flat observation.
- The robust failure is actor-side: `Set actor + Flat critic` remains far
  behind even after 1.5M.  This isolates the main problem to the path
  "learn one public Set descriptor, then have every actor read control from
  that descriptor plus local/env features".
- HAP is load-bearing in the two working-ish runs: freezing HAP increases
  held-out cost by +8.6% for `sort_flat` and +11.2% for `Flat actor + Set
  critic`.

Next algorithm direction after this diagnostic was to keep the paper-compatible
public descriptor `z=f({uavs})`, but not rely on a vanilla mean-pooled Set
descriptor as the actor's only control-readable public interface.  Later
cross-attention results confirmed that readout alignment matters, but the
remaining performance gap means the project should pause before adding
imitation or teacher-driven objectives.

## 2026-06-28 - Public descriptor / actor-critic coupling 400k diagnostics

### Question

The next check was deliberately kept inside the paper's public descriptor
framing.  We did not switch to query-conditioned per-agent descriptors.  The
goal was to identify whether the failure comes from:

1. the public descriptor + readout interface itself;
2. the learnable Set actor descriptor;
3. the learnable Set critic descriptor / advantage path.

### Implementation

New diagnostic switches:

```text
--mec_policy_arch sort_flat
--mec_critic_arch same|mean|flat|sort_flat|set
```

`sort_flat` lexicographically sorts UAV atoms by normalized `(x, y, queue)`
and then flattens them.  It is a non-learned, fixed-K, order-invariant public
descriptor.  This is not proposed as the final algorithm; it is a control test
for whether a public `z=f({uavs})` descriptor can be read by the existing
actor/critic interface when it preserves all UAV state information.

The logging test was also updated to match the production `add_scalar` call.

Verification:

```text
74 passed, 1 skipped, 20 subtests passed
compileall passed
```

### 400k seed-1 gate

Protocol: 350 slots, seed 1, `mec_logstd_init=-1.2`, 403.2k environment steps
/ 72 PPO updates, validation seed 1000, held-out seed 100000 with stride 13
over 24 episodes.

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| sort_flat actor + sort_flat critic | -1181.68 | 403.2k | 3.5106 | 55.1% | 968.9 m | +0.3% |
| Set mean_pool actor + Flat critic | -1514.10 | 302.4k | 4.6383 | 40.0% | 1230.4 m | -1.6% |
| Flat actor + Set mean_pool critic | -1433.38 | 201.6k | 4.2400 | 44.4% | 1155.3 m | -0.0% |

Artifacts:

```text
scripts/run_v6_descriptor_diagnostics_400k.ps1
eval_outputs/descriptor_diagnostics_400k/diag400_desc.summary.json
training_logs/diag400_desc_*.log
```

### Interpretation

`sort_flat` partially recovers UAV coverage and acceptance, so a public
order-invariant descriptor is not intrinsically impossible.  However, it still
does not recover a load-bearing HAP trajectory and remains much worse than the
working Mean/Flat high-variance gates.  This points to a readout/interface
difficulty introduced by canonicalizing the full set before the actor and
critic consume it, not just to information loss.

`Set actor + Flat critic` remains near the failed Set regime.  Therefore the
main actor-side problem is not explained away by a bad Set critic or a bad
advantage estimate alone: the learnable Set actor descriptor is not producing
control-readable action features.

`Flat actor + Set critic` also degrades badly.  This means the Set critic path
can independently poison the advantage signal even when the actor has the
successful ordered Flat information.  Actor and critic descriptor failures are
both real; the actor-side failure is not the only bottleneck.

Current root-cause hypothesis:

- public descriptor theory is still viable, but the descriptor must be
  explicitly control-readable, not merely reconstructive or permutation
  invariant;
- learnable Set descriptors under vanilla MAPPO are not being shaped into such
  a representation on either actor or critic side;
- the next useful step is an offline descriptor/readout probe and gradient /
  sensitivity audit, not another larger pure-PPO Set sweep.

## 2026-06-27 - Staged reconstruction pretraining pipeline

### Implementation

The naive joint reconstruction loss was replaced with a staged training path:

```text
python -m onpolicy.scripts.train.pretrain_mec_set_reconstruction
--mec_set_pretrained_actor <actor.pt>
--mec_set_freeze_pretrained_encoder_updates <N>
```

The pretrain script collects replay UAV atoms from MEC rollouts and trains only
`population_encoder + reconstruction_decoder` with Chamfer set loss. PPO can
then initialize only the Set representation weights from the pretrained
`actor.pt`; critic, actor readouts, optimizer states, and ValueNorm are not
loaded. The optional freeze switch keeps the actor population encoder fixed for
the first `N` PPO updates so the randomly initialized readouts and critic can
adapt before finetuning the representation.

A TensorBoard logging bug was also fixed: scalar tags with slashes such as
`mec/uav_compute_utilization` must use `add_scalar`, not `add_scalars`, because
TensorBoardX otherwise tries to create nested event-writer paths and can fail
on Windows.

### Mean-pool staged gate

Offline pretraining:

```text
encoder: mean_pool
replay: 64 mixed hover/random episodes, 22,400 UAV-set samples
epochs: 40
Chamfer: 0.0383 -> 0.0020
```

PPO with pretrained encoder and 12-update frozen warm-up:

```text
best validation:    -1620.48 at 134.4k steps
held-out cost/slot:  4.7404
accept:              37.7%
W1:                  1318.4 m
HAP-freeze:          -0.3%
```

PPO with the same pretrained encoder and no freeze, 112k quick gate:

```text
best validation: -1566.83 at 112k steps
```

### Latent-slot staged quick gate

Offline pretraining:

```text
encoder: latent_slots, 4 slots x 128 dim
replay: 64 mixed hover/random episodes, 22,400 UAV-set samples
epochs: 40
Chamfer: 0.0441 -> 0.0017
```

PPO with the pretrained latent-slot encoder, no freeze, 112k quick gate:

```text
best validation:    -1537.72 at 56k steps
held-out cost/slot:  4.6197
accept:              39.2%
W1:                  1268.6 m
HAP-freeze:          -0.2%
```

### Interpretation

The staged pipeline is now code-complete and self-consistent, but pure
geometry reconstruction is not enough. Both mean-pool and latent-slot encoders
learn to reconstruct the UAV set well, yet neither produces a useful control
policy under PPO. This suggests the bottleneck is not just whether `xi`
contains recoverable UAV atoms; the descriptor/readout must be shaped by a
control-relevant signal.

Archive decision: pause new algorithm branches here. The recent work has not
produced a substantive algorithmic improvement, and the core failure is still
not explained: introducing a learnable descriptor repeatedly breaks the
Mean/Flat-style PPO learning path even when the descriptor can reconstruct the
UAV set. Do not proceed to policy distillation, heuristic behavior cloning, or
more auxiliary-control hybrids for now; those would change the question rather
than explain why the descriptor path fails.

The next useful work should be analytical rather than another training sweep:

- inspect gradient scale and feature statistics at the descriptor/readout
  interface for Mean, Flat, mean_pool, flat_mlp, and latent_slots;
- compare whether actor logits/actions are sensitive to individual UAV atoms
  after the descriptor bottleneck;
- identify whether the problem is descriptor compression, readout conditioning,
  optimizer coupling, or loss assignment before proposing a new algorithmic
  stage.

## 2026-06-27 - Mean-pool Chamfer reconstruction auxiliary MVP

### Question

The reconstruction objective must preserve the Set/mean-pool permutation
contract. A simple row-wise MSE would incorrectly penalize equivalent UAV-set
permutations, so the auxiliary MVP uses a symmetric squared Chamfer-style set
loss instead.

The first implementation intentionally used the smallest possible training
change: add `lambda_rec * L_chamfer` to the actor optimizer during the usual
PPO minibatch update. This was a diagnostic, not the final SetRec training
architecture.

### Implementation

New pieces:

```text
--mec_set_reconstruction_coef <float>
PopulationReconstructionDecoder
chamfer_set_loss(prediction, target)
```

The decoder is attached to the shared Set actor path and reconstructs the UAV
state atoms from the population descriptor. It is currently restricted to
`mec_set_actor_encoder=shared` so the auxiliary target has a single clear actor
encoder to shape.

Tests verify that:

- Chamfer loss is invariant to target UAV order;
- mean-pool reconstruction loss is unchanged by permuting the input UAV rows;
- reconstruction gradients reach the actor population encoder and decoder;
- separate HAP/UAV actor encoders reject the current reconstruction auxiliary.

### 400k joint-loss gate

Protocol: 350 slots, seed 1, `mec_logstd_init=-1.2`, 403.2k environment steps
/ 72 PPO updates, validation seed 1000, held-out seed 100000 with stride 13
over 24 episodes.

```text
--mec_policy_arch set
--mec_set_encoder_type mean_pool
--mec_set_reconstruction_coef 0.1
```

The reconstruction loss itself learned:

```text
mec_set_reconstruction_loss: 0.2840 at 5.6k -> 0.0103 at 397.6k
```

But control did not recover:

```text
best validation:    -1511.02 at 201.6k steps
held-out cost/slot:  4.6367
accept:              39.0%
W1:                  1250.8 m
HAP-freeze:          -0.0%
```

Interpretation: the set reconstruction task is learnable and the Chamfer loss
is the right symmetry-preserving minimum viable loss, but simply mixing the
auxiliary loss into every PPO actor update is not a good final training
architecture. The auxiliary can dominate or drift the representation without a
policy-consistency constraint, and it still throws away useful off-policy
reconstruction data whenever the PPO rollout buffer is discarded.

### Revised training architecture

The next SetRec implementation should be staged:

1. Collect a replay dataset of canonical UAV atoms from random/heuristic/policy
   rollouts. This data is valid for representation learning even after the PPO
   on-policy window expires.
2. Pretrain `encoder + decoder` on replay with Chamfer/Sinkhorn reconstruction
   only. No actor or critic control loss is used in this stage.
3. Initialize PPO from the pretrained encoder. Either freeze the encoder for a
   short warm-up or use a much smaller auxiliary coefficient during PPO.
4. If auxiliary updates continue during policy training, run them in a
   PPG-style phase with a policy KL constraint, rather than as an
   unconstrained term inside every PPO actor update.
5. Promote from Chamfer to debiased Sinkhorn only after the staged Chamfer
   version improves representation probes or held-out control.

## 2026-06-27 - Flat-MLP population encoder 400k gate

### Question

The `mean_pool` Set failure could still be caused by the self-attention /
mean-pooling implementation rather than by the broader learnable population
representation framework. To check this, a simpler ordered fixed-K encoder was
added:

```text
--mec_set_encoder_type flat_mlp
```

It flattens the `K` UAV state atoms in their current order and maps them through
a MAPPO-style `MLPLayer` into a fixed-width descriptor. This is intentionally
not permutation invariant; it is a diagnostic closest to the existing ordered
Flat baseline while still preserving the Set-style two-stage
encoder/readout interface.

### Implementation

`flat_mlp` supports the same numerical contract as the other population
encoders: optional feature LayerNorm, `MLPLayer`, `layer_N`, `use_orthogonal`,
and `use_ReLU`. It supports only pooled actor context because it has no
equivariant per-UAV tokens.

### 400k no-sharing gate

The first gate used the strongest "remove sharing" setting:

```text
--mec_set_encoder_type flat_mlp
--mec_set_actor_encoder separate
--mec_set_critic_encoder separate
```

Protocol: 350 slots, seed 1, `mec_logstd_init=-1.2`, 403.2k environment steps
/ 72 PPO updates, validation seed 1000, held-out seed 100000 with stride 13
over 24 episodes.

```text
best validation:   -1470.92 at 403.2k steps
held-out cost/slot: 4.1688
accept:             45.6%
W1:                 1166.1 m
HAP-freeze:         -0.8%
```

The flat MLP encoder is slightly better than mean-pool Set but still far behind
Mean/Flat and still learns no load-bearing HAP trajectory. This weakens the
hypothesis that the failure is specifically caused by self-attention or
mean-pooling code. The remaining difference from the successful Flat baseline
is the two-stage learnable population descriptor bottleneck/readout interface
itself, not just the particular Set encoder internals.

## 2026-06-27 - HAP/UAV separate Set actor encoder gate

### Question

The shared Set actor encoder was checked as a potential framework-level cause
of the Set failure. Mean/Flat have no learnable shared population trunk, while
Set used one actor-side `population_encoder` for both HAP and UAV actors. This
could create role-gradient interference because the HAP actor and UAV actor
need different control information.

### Implementation

Added a diagnostic switch:

```text
--mec_set_actor_encoder {shared,separate}
```

`shared` is the previous default. `separate` builds independent
`major_population_encoder` and `minor_population_encoder` modules. It requires
`--mec_set_critic_encoder separate`, so the critic has its own third encoder
instead of implicitly choosing one actor encoder.

Tests now verify that:

- HAP/UAV actor encoders have disjoint parameters;
- both actor encoders are owned by the actor optimizer only;
- the critic encoder is owned by the critic optimizer only;
- a major-only actor loss reaches only the HAP encoder;
- a minor-only actor loss reaches only the UAV encoder.

### 112k gate

Protocol: 350 slots, seed 1, `mec_logstd_init=-1.2`, 112k environment steps /
20 PPO updates, validation seed 1000, held-out seed 100000 with stride 13 over
24 episodes.

```text
Set mean_pool + separate HAP/UAV actor encoders + separate critic encoder
best validation: -1567.28 at 28k steps
held-out cost/slot: 4.5959
accept: 39.5%
W1: 1287.3 m
HAP-freeze: -0.0%
```

Conclusion: separating HAP/UAV actor encoders did not produce an early recovery
signal. This weakens the hypothesis that the Set failure is mainly caused by
HAP/UAV sharing the same actor encoder. Do not expand this branch to 896k unless
new evidence appears; continue toward auxiliary reconstruction / representation
learning.

## 2026-06-27 - Full high-variance Mean/Flat vs Set mean-pool gate

### Code audit and repair before the full gate

The user's concern about parameter sharing was accepted as valid. Mean/Flat
have no learnable shared trunk between the HAP actor, UAV actor, and critic;
Set introduces a learnable `population_encoder` shared by HAP/UAV actor
branches, and the default critic only reuses that encoder under `no_grad`.

Before running the longer experiment, the mean-pool encoder was audited:

- permutation invariance/equivariance and actor/critic dimensions were already
  correct;
- the readouts already used the repaired `FusionMLP` MAPPO contract;
- the population encoder itself did not yet honor the same initialization and
  normalization conventions.

The encoder contract was repaired so both `latent_slots` and `mean_pool`
population encoders now use:

- optional atom input LayerNorm from `use_feature_normalization`;
- `MLPLayer` for the atom encoder, honoring `layer_N`, `use_orthogonal`, and
  `use_ReLU`;
- initialized multi-head attention / FFN blocks following `use_orthogonal`.

Regression tests were extended to lock this contract.

### Full 896k protocol

All runs used 350-slot episodes, seed 1, `mec_logstd_init=-1.2`, 16 rollout
workers, 160 PPO updates / 896k environment steps, validation seed 1000, and a
disjoint held-out test split with seed 100000, stride 13, 24 episodes.

Validation best checkpoints:

| architecture | best validation reward | selected step |
|---|---:|---:|
| Mean | -766.40 | 784k |
| Flat | -815.46 | 896k |
| Set mean_pool, actor_detached | -1478.39 | 224k |
| Set mean_pool, separate critic encoder | -1525.51 | 672k |
| Set mean_pool, shared_grad | -1511.63 | 672k |

Held-out deterministic evaluation of the selected checkpoints:

| architecture | cost/slot | accept | W1 | HAP-freeze delta |
|---|---:|---:|---:|---:|
| Mean | 2.2695 | 73.5% | 702.2 m | +28.2% |
| Flat | 2.3591 | 73.0% | 729.4 m | +5.0% |
| Set mean_pool, actor_detached | 4.4842 | 41.1% | 1220.4 m | -0.1% |
| Set mean_pool, separate critic encoder | 4.5496 | 40.3% | 1248.6 m | -0.1% |
| Set mean_pool, shared_grad | 4.3408 | 43.2% | 1209.3 m | -0.1% |

### Interpretation

The longer run supports the user's request for stronger evidence: Mean and
Flat continue improving over the full 896k budget, while all three
Set-mean_pool paths remain far behind and do not learn a load-bearing HAP
trajectory.  The `shared_grad` path is the best of the Set mean-pool variants
on held-out cost, but it is still roughly 1.9x Mean's cost and has no useful
HAP-freeze signal.

This substantially weakens the hypothesis that Set only needs more steps
because it has more parameters. The current evidence favors the conclusion that
pure MAPPO does not give the shared population encoder enough structured
representation signal. The next Set-specific step should therefore be an
auxiliary decoder / reconstruction phase, then Sinkhorn / PPG once the cheaper
auxiliary signal is validated.

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

## 2026-06-26 - Exploration scale diagnosis and high-variance Flat gate

### Questions checked

Two concerns were tested:

1. the repaired Mean/Flat curves might still be climbing at 896k steps;
2. learned trajectories might be too slow because the Gaussian exploration
   scale is too small, not because the task intrinsically prefers slow flight.

### Learning-curve evidence

For repaired 350-slot Mean/Flat seed 1, the best validation point appeared at
the last evaluation:

- Mean: validation improved to `-1107.247` at 896k steps.
- Flat: validation improved to `-986.458` at 896k steps.

Therefore the repaired baselines had not reached a clean plateau. Longer runs
are justified after the current gate decisions.

### Speed-scale ablation

`scripts/diagnose_v6_speed_scale.py` was added. It leaves the learned direction
and beta fixed, multiplies deterministic velocity commands by a scale factor,
and lets the environment project to the physical speed limits.

For repaired Flat seed 1, using 8 held-out episodes:

| UAV speed scale | cost/slot | cost delta | accept | W1 | mean UAV speed | safety violations/slot |
|---:|---:|---:|---:|---:|---:|---:|
| 0x | 4.8565 | +51.5% | 36.0% | 1337.7 m | 0.0 m/s | 0.000 |
| 1x | 3.2064 | 0.0% | 59.0% | 923.7 m | 5.0 m/s | 0.063 |
| 2x | 3.0363 | -5.3% | 61.6% | 879.0 m | 7.9 m/s | 0.088 |
| 4x | 3.5537 | +10.8% | 55.7% | 936.0 m | 13.9 m/s | 0.501 |
| 8x | 4.4404 | +38.5% | 43.7% | 1148.9 m | 24.0 m/s | 2.935 |

Interpretation: the learned policy is under-using moderate speed, but forcing
very high speed overshoots the hotspot and starts creating safety issues. The
right intervention is more exploration around moderate speeds, not a hard-coded
full-speed controller.

### High-variance Flat gate

A Flat seed-1 gate was run with `mec_logstd_init=-1.2`
(`sigma ~= 0.30`, around 12 m/s for UAVs) instead of the previous `-1.9`
(`sigma ~= 0.15`, around 6 m/s). All other training settings were unchanged.
Only 448k steps were used.

| architecture | steps | held-out cost/slot | accept | W1 | HAP-freeze delta |
|---|---:|---:|---:|---:|---:|
| Flat, fixed readout, sigma 0.15 | 896k | 2.9574 | 62.6% | 883.9 m | -0.2% |
| Flat, fixed readout, sigma 0.30 | 448k | 2.6442 | 68.8% | 785.9 m | +1.0% |
| legacy_mean reference | 512k old / eval at 350 slots | 2.7470 | 65.4% | 767.9 m | +15.2% |

The higher-variance Flat gate outperformed the previous Flat run and slightly
beat the historical legacy-mean cost reference on the held-out split, while
remaining below the heuristic cost reference. Its deterministic mean UAV speed
rose from about `5.0 m/s` to `7.2 m/s`, and final action sigmas stayed near
`0.29`.

### Set high-variance gate

The same `mec_logstd_init=-1.2`, 448k-step seed-1 gate was run for Set. It did
not recover:

- best validation remained around `-1531`, essentially unchanged from the old
  Set gate;
- explained variance reached only about `0.64-0.71`, compared with `0.92-0.96`
  for high-variance Flat;
- actor gradients/policy losses were weak.

This points to a Set-specific representation/critic training issue, not merely
an exploration-scale issue. The current Set critic consumes the actor's
population encoder under `torch.no_grad()`, so the value loss cannot shape the
Set encoder. That design should be revisited before adding decoder, Sinkhorn
reconstruction, or PPG.

### Set structure ablations

Three 112k-step Set diagnostics were then run with the same 350-slot seed-1
high-variance setting (`mec_logstd_init=-1.2`) to check whether the problem was
simply critic ownership, excessive encoder depth, or a pooled-only UAV readout.

| Set variant | changed factor | best validation reward |
|---|---|---:|
| baseline high-variance Set | pooled Set actor, actor encoder detached in critic | -1531.43 |
| separate critic encoder | critic owns and trains an independent population encoder | -1531.60 |
| small Set encoder | dim 32, 2 latent seeds, 1 element block, 0 latent blocks | -1531.20 |
| relational UAV context | each UAV also receives its equivariant attention token | -1534.83 |

Flat under the same high-variance setting was already much better at the first
comparable evaluation (`-1414.47` at 112k) and reached `-874.83` by 448k.
Therefore Set is not merely slower because of parameter count, critic encoder
detachment, or lack of a per-UAV token. The evidence now favors a broader
representation-learning issue: the Set attention encoder is not being shaped
into a useful control representation by the sparse PPO objective alone.

### Simple encoder and optimizer-path ablation

To separate the update-path question from the more complex latent-slot Set
encoder, a simple population encoder was added:

```text
UAV atoms -> atom MLP -> multi-head self-attention blocks -> mean pool
```

It is enabled with `--mec_set_encoder_type mean_pool`.  Three 112k-step
high-variance gates were then run to compare where the learnable population
encoder is updated:

| encoder | critic encoder path | who updates actor encoder | best validation reward |
|---|---|---|---:|
| mean_pool | actor_detached | actor PPO only | -1540.76 |
| mean_pool | separate | actor PPO only; critic has its own encoder | -1539.08 |
| mean_pool | shared_grad | actor PPO + critic value optimizer | -1546.69 |

This directly addresses the parameter-sharing concern:

- Mean/Flat still have no learnable shared trunk between HAP actor, UAV actor,
  and critic.
- Set introduces a learnable shared population encoder for HAP and UAV actor
  branches.
- Allowing the value loss to update a simple shared encoder did not rescue
  learning; if anything, the two-optimizer `shared_grad` diagnostic was worse.

Therefore the current failure is not just caused by the latent-slot pooling
architecture, nor by the critic's default `no_grad` encoder path.  The likely
issue is that PPO's sparse control objective does not provide a sufficiently
structured representation-learning signal for a shared population encoder.

### Decision

Use `mec_logstd_init=-1.2` as the new default candidate for the next Mean/Flat
gates. Do not hard-code faster velocity. Do not spend more long runs on pure
Set-MAPPO until the Set representation receives an auxiliary shaping signal.
The next algorithmic task should be the planned decoder / reconstruction path.
Start with a cheap auxiliary reconstruction phase on the simple `mean_pool`
and/or default `latent_slots` encoder, then add Sinkhorn/PPG scheduling once
the auxiliary signal demonstrably improves representation quality.

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

## 2026-07-01 - Reset-Permutation Mean Baseline and Critic Isolation

Protocol: `v6_hap_loadbearing`, 350-slot episodes, 1.5M environment steps,
validation seed 1000 over 24 episodes, held-out seed 100000 with stride 13 over
24 episodes, and reset-time UAV-row permutation.

### Main reset-permutation results

| variant | seeds | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|
| Mean actor + Mean critic | 1/2/3 | 2.0991 | 75.07% | 684.6 m | +59.2% |
| Flat actor + Flat critic | 1/2/3 | 2.7326 | 66.55% | 800.6 m | +12.9% |
| Latent Slot-EqDec actor + Flat critic | 1/2/3 | 3.1458 | 60.54% | 972.8 m | +8.1% |
| Latent Slot-EqDec actor + Set critic | 1/2/3 | 4.3119 | 44.87% | 1145.4 m | -0.7% |
| Flat descriptor actor + Flat critic, dim 256 | 1/2/3 | 4.4281 | 42.24% | 1212.0 m | +0.3% |

The fair Mean descriptor baseline is now the strongest reset-permutation
result.  It beats Flat actor + Flat critic by `0.6335` cost/slot and improves
acceptance by `8.52` percentage points.  This indicates that the reset
permutation removes a useful fixed-row shortcut from Flat, while the mean
descriptor supplies a robust first-moment inductive bias that is highly matched
to the current scenario.

### Flat-descriptor dimension sweep

All entries below use seed 1 and Flat critic:

| descriptor dim | held-out cost/slot | accept | W1 | HAP-freeze |
|---:|---:|---:|---:|---:|
| 16 | 4.2389 | 45.48% | 1165.0 m | -1.0% |
| 32 | 4.1678 | 46.91% | 1134.9 m | -2.9% |
| 64 | 4.5236 | 40.72% | 1218.0 m | -0.8% |
| 128 | 4.6397 | 39.18% | 1243.1 m | -1.0% |
| 256 | 4.3981 | 42.40% | 1219.6 m | -0.4% |

Lowering the descriptor dimension does not rescue the learned bottleneck.
Dim 32 is the least bad, but it is still far behind Mean, Flat, and
Slot-EqDec.  The likely issue is not descriptor width.  It is that PPO does not
reliably learn a control-readable broadcast descriptor through this weak
end-to-end signal.

### Critic isolation with a strong Mean actor

| variant | seed | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|
| Mean actor + Mean critic | 1 | 2.1286 | 74.63% | 700.9 m | +59.1% |
| Mean actor + Flat critic | 1 | 2.5767 | 67.38% | 828.9 m | +19.5% |
| Mean actor + separate Set critic | 1 | 3.2847 | 59.74% | 872.4 m | +8.9% |

This isolates two facts.  First, the Set critic is still a real bottleneck when
paired with a non-Set Mean actor, even with a separate critic encoder.  Second,
Mean + Flat critic also lags Mean + Mean, so the Mean baseline's strength is
not purely actor-side; the simple mean value representation is also very well
matched to this environment.

### Paper-facing interpretation

The current environment appears to be largely first-moment sufficient.  A
simple permutation-invariant mean descriptor plus local UAV state explains most
of the control problem.  Learned descriptors and slots are not yet justified as
replacements for mean in this scenario.

The manuscript should therefore pivot from "learned descriptor replaces mean"
to a more defensible question: can an invariant learned descriptor provide
residual higher-order empirical-measure information beyond the strong mean
baseline, especially in stress settings where a first moment is insufficient?

Immediate next experiments:

1. Implement a mean-residual Slot-EqDec actor that always receives `mean_uav`
   and adds a local-query slot context as residual information.
2. Run `mean_residual_slot + Mean critic`, seed 1 full 1.5M, then compare to
   Mean + Mean seed 1 and Mean + Flat critic seed 1.
3. Add a post-hoc representation probe for true set reconstruction W1, separate
   from the demand-matching W1 currently reported by evaluation.
4. Design a mean-insufficient stress scenario, such as split/multi-hotspot
   demand, where higher-order fleet geometry should matter.
