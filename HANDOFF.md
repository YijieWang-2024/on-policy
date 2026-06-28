# MEC 当前阶段交接摘要（2026-06-26）

> 本文是当前阶段的唯一交接入口。实际操作见
> [`docs/mec_runbook.md`](docs/mec_runbook.md)，系统模型和历史设计依据见
> [`docs/mec_env_port_spec.md`](docs/mec_env_port_spec.md)，当前 SetRec
> 网络与训练契约见
> [`docs/setrec_architecture.md`](docs/setrec_architecture.md)。

## 本次阶段归档

本次归档覆盖从 `v6_hap_loadbearing` 机制校准到 Set-MAPPO phase 1
落地的完整工作：

- 完成 resumable checkpoint、固定验证集 best selection 和 role-wise MAPPO；
- 完成 beta 可辨识性诊断，并将轻微超负载配置保留为可选机制场景；
- 重新审查 empirical-measure 理论与实际系统动力学的适用边界；
- 确定公共 invariant population descriptor、共享 UAV actor 和单一 team
  critic 的架构契约；
- 将环境观测和 centralized state 改为 canonical 物理信息接口；
- 实现可切换的 `legacy_mean/mean/flat/set`，其中新 `mean/flat/set`
  只替换 population representation；
- 保留旧 14 维 mean checkpoint 的自动兼容路径；
- 完成结构测试、端到端训练 smoke、新旧 checkpoint 加载和恢复训练验证。

验证基线为 `51 passed, 1 skipped, 20 subtests passed`。当前阶段的代码目标
已经完成，下一阶段是重新训练同预算 `mean/flat/set` 三种子结构对照。

## 0. 2026-06-25 表示对齐重构

本轮进一步确认：旧 mean 基线同时使用了共享 actor trunk 和有序 centralized
critic，不能作为只比较 population representation 的严格对照。因此当前代码契约为：

- 环境局部观测统一为 `[role, own(3), p_phys(7), resource(6)]`，共 17 维；
- runner 统一构造 `[p(13), s_1(3), ..., s_K(3)]` centralized state；
- 新 `mean/flat/set` 的 HAP actor、共享 UAV actor、team critic 和 grouped PPO
  完全一致，只替换 population representation；
- `mean` 输入分别为 `p+mean`、`s_i+p+mean`、`p+mean`；
- 历史网络改名为 `legacy_mean`，仅用于旧 checkpoint 复现；旧 config 缺少
  `mec_policy_arch` 时，训练恢复、评估和渲染入口会自动选择该模式。

因此，下一轮公平结构实验必须重新训练 `mean/flat/set` 三组；过去的 mean
结果保留为历史工程基线，不能直接与新 flat/set 解释为纯表示消融。

## 1. 当前阶段

已经完成：

- v6 continuous workload 环境、接入、队列、计算和连续 28 GHz 回传实现。
- 可复现性修复和 CPU 训练参数确认。
- `v6_continuous_workload` 的 MAPPO seed 1/2/3，各训练 1.5M steps。
- 三个模型在相同 24 个测试 episode 上的统一评估。
- HAP 承重区间扫描，以及工程约束下的新候选场景。
- `v6_hap_loadbearing` 的 512k-step、seed 1/2/3 诊断训练。
- latest、step checkpoint、优化器/ValueNorm 恢复和固定验证集 best selection。
- MEC major/minor role-wise advantage、PPO loss 和 entropy 等权归一化。
- role-wise 改造后的同预算 512k-step、seed 1/2/3 对照训练。
- beta 可辨识性诊断与轻微超负载候选场景分析。默认环境不因此改变；
  `A_tot=165 Mbit/slot, F_U=1.8 GHz` 只保留为可选机制验证场景。
- 基于实际 v6 动力学重新审查 SetRec 理论和算法。确认 empirical-measure
  交换性主线成立，但现稿的 `gamma L_F < 1` 不能直接视为已被系统满足。
- 完成 SetRec 第一阶段架构定案：一个公共 UAV population encoder、
  独立 HAP/UAV readout、一个 team critic，以及后续 PPG-style
  reconstruction auxiliary phase。
- 已实现 `--mec_policy_arch legacy_mean|mean|flat|set`、canonical observation/
  centralized state、grouped PPO minibatch、Set Transformer population encoder、
  对齐 Mean/Flat 对照和单一 team critic。
- 结构测试、Set/Flat 训练 smoke、Set checkpoint 恢复和 policy eval 均已通过。

尚未完成：

- 对齐 `mean/flat/set` 的同预算 3-seed 结构训练与统一评估。
- Set reconstruction decoder、Sinkhorn divergence 和 auxiliary phase。
- 论文最终 5-seed、跨 K 和统计置信区间实验。

当前不应直接跑论文最终实验。beta 诊断暂不作为主线阻塞条件。Set-MAPPO
第一阶段的结构正确性已经通过测试；当前执行 350-slot、同预算
`mean/flat/set` 三 seed 对照。best checkpoint 使用 validation seed 1000，
最终报告改用 held-out test seed 100000。reconstruction 只在 Set-MAPPO
稳定后加入。

## 2. `v6_hap_loadbearing` 短程确认与 role-wise 对照

统一 24-episode deterministic evaluation：

| 训练设置 | seed | cost/slot | accept | W1 | HAP freeze |
|---|---:|---:|---:|---:|---:|
| 原 MAPPO | 1 | 2.8311 | 64.0% | 836.1 m | +4.7% |
| 原 MAPPO | 2 | 2.4619 | 68.6% | 782.8 m | +11.2% |
| 原 MAPPO | 3 | 2.3951 | 69.7% | 774.5 m | +4.6% |
| role-wise MAPPO | 1 | 2.5011 | 68.4% | 802.0 m | +7.4% |
| role-wise MAPPO | 2 | 2.3863 | 69.7% | 761.9 m | +10.4% |
| role-wise MAPPO | 3 | 2.5010 | 68.3% | 797.7 m | +11.2% |
| heuristic | - | 2.1776 | 72.3% | 762.2 m | - |

判断：

- role-wise 后平均 cost 从 2.5627 降至 2.4628，accepted 增加约
  2.0 Mbit/slot。
- HAP freeze 平均从 +6.8% 提升到 +9.7%；seed 2/3 超过 10%，seed 1
  仍只有 +7.4%。major 学习明显改善，但尚未三 seed 全部达标。
- UAV 轨迹继续稳定承重；hover 消融平均从 +71.9% 提升到 +77.9%。
- beta 仍未学到有效的逐 UAV 控制：每步把 beta 替换为全 UAV 均值，
  cost 变化约为 -1.0%、+0.03%、+0.30%。
- 三个 role-wise run 的固定验证 reward 最优点都在 512k；本轮没有末期退化，
  但 step/best checkpoint 机制已经可用于后续结构实验。

## 3. 已完成训练的历史结论

旧主场景：`v6_continuous_workload`。

统一 24-episode deterministic evaluation：

| 控制器 | cost/slot | accept | W1 | queue share | HAP freeze |
|---|---:|---:|---:|---:|---:|
| MAPPO seed 1 | 2.4254 | 72.1% | 756.1 m | 13.1% | +24.0% |
| MAPPO seed 2 | 2.3041 | 72.5% | 765.3 m | 9.9% | +0.5% |
| MAPPO seed 3 | 2.5190 | 71.1% | 761.2 m | 13.1% | +0.2% |
| heuristic | 2.1776 | 72.3% | 762.2 m | 3.6% | - |

判断：

- 三个 seed 都学会了 UAV demand matching，W1 相比 hover/random 改善约 32%-33%。
- 总成本仍比 heuristic 高，主要差距来自队列和 beta 控制。
- seed 2 的 beta 对 UAV/HAP 队列响应最合理，seed 1/3 不稳定。
- 旧 v6 的回传利用率过低，HAP 轨迹不是稳定承重变量。
- 上述历史模型仍是旧 3 维 UAV 均值 descriptor；当前代码已新增对齐的
  `mean/flat/set`，但尚无正式训练结果。

这些结果是算法和环境诊断基线，不是论文最终结果。

## 4. HAP 承重扫描

扫描工具：

```text
onpolicy/scripts/analysis/scan_v6_hap_loadbearing.py
```

旧 v6 回传参数：

```text
W_bh_total = 400 MHz
P_tx = 27 dBm
combined/effective gain = 15 dB
explicit link margin = 0 dB
```

旧 v6 的实际回传利用率约 16%，冻结 HAP 几乎没有影响。问题不是带宽制式，
而是净链路预算过于宽裕。

根据 FR2 带宽和功率等级约束重新扫描后，保留 400 MHz，得到候选配置：

```text
W_bh_total = 400 MHz
P_tx = 23 dBm
combined Tx/Rx antenna gain = 20 dB
link margin = 7 dB
noise figure = 8 dB
path-loss exponent = 2.2
hard cutoff = false
```

该配置位于：

```text
onpolicy/envs/mec/scenarios/v6_hap_loadbearing.yaml
```

10 个配对 episode 的机制验证：

| 消融/指标 | 结果 |
|---|---:|
| 冻结 HAP | cost +28.2% |
| beta=0 | cost +213.3% |
| UAV hover | cost +186.0% |
| backhaul utilization | 44.1% |
| HAP compute utilization | 75.2% |
| overflow | 0 |
| accepted | 120.5 Mbit/slot |

注意：这些是启发式机制验证，不等于 RL 已经学会三类联合动作。

## 5. 当前代码入口

| 文件 | 作用 |
|---|---|
| `onpolicy/envs/mec/scenarios/v6_hap_loadbearing.yaml` | 换机短程训练候选 |
| `onpolicy/envs/mec/scenarios/v6_continuous_workload.yaml` | 已完成 3-seed 的旧基线 |
| `onpolicy/envs/mec/finite_k_env.py` | workload、接入、回传、队列和代价 |
| `onpolicy/envs/mec/config_loader.py` | 配置推导，含显式 `link_margin_db` |
| `onpolicy/envs/mec/MEC_env.py` | 多智能体适配和系统指标 |
| `onpolicy/algorithms/mec/mec_policy.py` | MEC mean/flat/set policy 入口 |
| `onpolicy/algorithms/r_mappo/r_mappo.py` | MEC role-wise advantage/loss |
| `docs/setrec_architecture.md` | 当前算法、输入、参数归属与训练阶段契约 |
| `scripts/diagnose_v6_learned_policy.py` | 三类动作 matched-seed 消融 |
| `onpolicy/scripts/analysis/design_v6_sanity.py` | 场景静态/rollout probe |
| `onpolicy/scripts/analysis/scan_v6_hap_loadbearing.py` | HAP 链路预算扫描 |
| `onpolicy/scripts/eval/eval_mec.py` | policy/heuristic/hover/random 评估 |

## 6. 当前执行顺序

1. 保留 role-wise loss，重新跑同预算 `mean/flat/set` 三 seed 结构对照；
   已完成的旧 mean role-wise 结果只作为历史工程对照。
2. 使用固定验证集和相同 24 episodes 做统一评估及动作消融。
3. Set-MAPPO 稳定后加入 decoder、Sinkhorn divergence 和 PPG-style
   auxiliary phase。
4. 只有结构消融和三类动作均跨 seed 稳定，才进入 1.5M/5-seed 最终实验。

短程实验命名建议：

```text
v6_hap_lb_probe_seed1
v6_hap_lb_probe_seed2
v6_hap_lb_probe_seed3
```

完整命令见 [`docs/mec_runbook.md`](docs/mec_runbook.md)。

## 7. 短程训练通过标准

不能只看 reward。至少要求：

- cost 随训练下降，且不存在 accepted 主动下降的投机策略。
- `n_hotspot_uav` 大致保持在 5-7，背景 UAV 不消失。
- `mec/backhaul_utilization` 对 HAP 几何有响应。
- beta 对 UAV 队列正向响应，对 HAP 拥塞有抑制响应。
- 冻结 learned HAP 后 cost 明显增加，目标暂定大于 10%。
- overflow 不长期爆炸。
- 三个 seed 中不能只有一个 seed 使用 HAP。

当前 role-wise 已缓解 major 样本占比过低的问题，结构阶段重点处理：

1. actor 的 UAV 均值描述子存在群体构型混叠；
2. centralized critic 使用有序拼接，尚非置换不变；
3. PPO minibatch 必须保留完整 team group；
4. Set/critic 改造后重新检查三 seed HAP freeze 和 beta 行为。

## 8. 文档职责

| 文档 | 职责 |
|---|---|
| `README.md` | 项目总览 |
| `HANDOFF.md` | 当前阶段状态和下一步 |
| `docs/mec_runbook.md` | 换机命令和运行步骤 |
| `docs/mec_env_port_spec.md` | 系统、参数、历史决策和技术依据 |
| `docs/setrec_architecture.md` | 当前算法架构与训练契约 |
| `WORK_NOTES.md` | 按日期记录实际工作 |
| `docs/diagnosis_layout_not_loadbearing.md` | v2/v3 历史失败证据，仅作 archive |

没有删除 Markdown：现有文件职责仍然不同，历史失败证据也仍用于解释场景
演进。后续不再把当前算法设计继续堆进历史环境规范。

## 9. 2026-06-26：350-slot 三结构实验最终结论

正式 `mean/flat/set x seed 1/2/3` 已全部完成。每个 run 使用 350 slots、
16 rollout workers、160 次 PPO 更新和 896k environment steps；validation
仅用于选择 best checkpoint，最终结果来自独立的 seed 100000 held-out split。

| 架构 | cost/slot | acceptance | W1 诊断 | 冻结 HAP |
|---|---:|---:|---:|---:|
| Mean | 4.6491 +/- 0.1292 | 38.9% +/- 1.5% | 1285.7 +/- 14.6 m | -0.3% +/- 0.5% |
| Flat | 4.5822 +/- 0.0900 | 39.7% +/- 1.2% | 1260.9 +/- 32.9 m | 约 0% |
| Set | 4.6230 +/- 0.0669 | 39.1% +/- 0.9% | 1287.0 +/- 9.3 m | 约 0% |
| heuristic | 2.0694 | 73.7% | 734.1 m | - |

Set 的跨 seed 方差最小，说明代码可以稳定优化；但三种策略都没有学出有效的
轨迹控制。代表性 checkpoint 中，冻结 UAV 运动反而降低成本约 1.4%-1.9%，
UAV 平均速度背离热点，HAP 平均速度背离 UAV 质心。

旧 role-wise `legacy_mean` checkpoint 在完全相同的 350-slot held-out split
上仍达到 cost/slot `2.5610/2.8504/2.8297`，acceptance `67.9%/64.2%/64.1%`，
冻结 HAP 会恶化 `19.4%/10.2%/16.0%`。因此 episode 延长不是根因，回归来自
新的 aligned representation/readout 路径。

当前决定：不进入 decoder、Sinkhorn reconstruction 或 PPG auxiliary phase。
下一步先修复 aligned `FusionMLP` 与 legacy `MLPBase` 不等价的问题：
它目前没有遵守 `use_feature_normalization`、`use_orthogonal`、`layer_N`，
也缺少 legacy trunk 的逐层 LayerNorm。修复后先做一个 seed 的
`legacy_mean/mean/flat/set` 回归门控；只有恢复有效 UAV/HAP 运动后，才重跑
三 seed 并进入 reconstruction。

## 10. 2026-06-26: aligned readout repair gate

The `FusionMLP` regression was repaired. The aligned readouts now follow the
legacy MAPPO `MLPBase` contract: input LayerNorm when enabled, configured
orthogonal/Xavier initialization, configured `layer_N`, and per-hidden-layer
LayerNorm.

Seed-1 full 350-slot gates now recover normal learning:

| architecture | held-out cost/slot | acceptance | W1 | HAP freeze |
|---|---:|---:|---:|---:|
| Mean, fixed readout | 3.2557 | 59.4% | 919.1 m | +2.2% |
| Flat, fixed readout | 2.9574 | 62.6% | 883.9 m | -0.2% |

The PPO framework, grouped minibatch path, environment output contract, and
validation/test split passed regression tests. Full verification:
`57 passed, 1 skipped, 20 subtests passed`; `compileall` and
`git diff --check` passed.

Next step: run the fixed Set seed-1 350-slot gate before adding decoder,
Sinkhorn reconstruction, or PPG. If Set also recovers, rerun the formal
3-seed Mean/Flat/Set comparison with the repaired readout.

## 11. 2026-06-26: exploration scale gate

The slow-trajectory concern was tested directly.  A speed-scale diagnostic
showed that doubling learned UAV velocities improves repaired Flat cost by
about 5%, but scaling by 4x or 8x hurts due to overshoot and safety violations.
So the right intervention is moderate exploration, not hard-coded full-speed
flight.

Flat with `mec_logstd_init=-1.2` (`sigma ~= 0.30`) was trained for only 448k
steps and already reached held-out cost/slot `2.6442`, acceptance `68.8%`, and
W1 `785.9 m`.  This beats the previous repaired Flat 896k result
(`2.9574`) and is slightly better than the legacy_mean cost reference
(`2.7470`).

The same high-variance 448k gate did not rescue Set: best validation stayed near
`-1531`.  Set now looks like a representation/critic-training issue.  The
current Set critic consumes the actor population encoder under `torch.no_grad()`,
so critic value loss cannot shape the Set encoder.

Next step: keep `mec_logstd_init=-1.2` for Mean/Flat gates, and ablate Set
encoder ownership/training before adding reconstruction/Sinkhorn/PPG.

## 12. 2026-06-26: Set diagnosis after structure ablations

Set was checked for the obvious architecture issues before moving to
reconstruction. The aligned readouts still use the repaired MAPPO-friendly
`FusionMLP`; the Set-specific attention encoder is larger than Mean/Flat, but
shrinking it did not fix learning.

New diagnostic switches:

- `--mec_set_critic_encoder {actor_detached,separate}`: keeps the old
  detached actor-encoder critic by default, or trains an independent critic
  population encoder with the value loss.
- `--mec_set_actor_context {pooled,relational}`: keeps the old pooled-only UAV
  readout by default, or also gives each UAV its equivariant self-attention
  token.

112k-step, 350-slot, seed-1, `mec_logstd_init=-1.2` Set diagnostics:

| variant | best validation |
|---|---:|
| high-variance Set baseline | -1531.43 |
| separate critic encoder | -1531.60 |
| small Set encoder | -1531.20 |
| relational UAV context | -1534.83 |

For comparison, high-variance Flat reached `-1414.47` at 112k and `-874.83` at
448k. The Set failure therefore is not explained by one missing readout trick,
critic encoder detachment, too many layers, or the pooled-only minor input
alone. Current conclusion: pure PPO is not shaping the Set attention encoder
into a useful control representation. Do not spend large compute on pure
Set-MAPPO now; move to the decoder/reconstruction/auxiliary representation
phase and keep Flat high-variance as the working control baseline.

## 13. 2026-06-27: simple encoder / optimizer-path ablation

The Mean/Flat/Set parameter-sharing concern was checked explicitly. The user's
interpretation is correct:

- Mean/Flat have no learnable shared trunk between HAP actor, UAV actor, and
  critic. Their shared population descriptor is non-parametric.
- Set shares a learnable actor `population_encoder` between HAP and UAV actor
  branches. The default critic reuses that encoder in forward under
  `torch.no_grad()`, so value loss does not update it.

New switches:

- `--mec_set_encoder_type {latent_slots,mean_pool}`. `mean_pool` is a simple
  diagnostic encoder: atom MLP, multi-head self-attention, invariant mean pool.
- `--mec_set_critic_encoder shared_grad`. This registers the actor encoder in
  the critic too, so both actor and critic optimizers update the same encoder.
  It is diagnostic only; it is less clean than a PPG-style auxiliary phase.

112k-step, 350-slot, seed-1, `mec_logstd_init=-1.2` mean-pool diagnostics:

| critic path | best validation |
|---|---:|
| actor_detached | -1540.76 |
| separate critic encoder | -1539.08 |
| shared_grad | -1546.69 |

So the simple encoder did not train stably either, and directly letting value
loss update the shared actor encoder did not help. The Set issue is broader
than latent-slot complexity or critic `no_grad`. Current recommendation:
do not continue pure Set-MAPPO sweeps. Proceed to auxiliary representation
learning, starting with a cheap reconstruction/decoder phase, then add
Sinkhorn/PPG once the auxiliary signal is validated.

## 14. 2026-06-27: full 896k confirmation gate

The quick-gate conclusion was rechecked with a full training budget after
auditing the `mean_pool` code. The audit found that mean-pool's invariance,
equivariance, dimensions, and critic paths were correct, but the population
encoder had not fully inherited the MAPPO-friendly numerical contract. This
was repaired:

- atom encoder now uses optional feature LayerNorm and `MLPLayer`;
- `layer_N`, `use_orthogonal`, and `use_ReLU` are honored;
- attention / FFN blocks are initialized according to `use_orthogonal`;
- regression tests lock the encoder contract.

Full protocol: 350 slots, seed 1, `mec_logstd_init=-1.2`, 896k environment
steps / 160 PPO updates, validation seed 1000, held-out seed 100000 with stride
13 over 24 episodes.

Validation best:

| architecture | best validation | selected step |
|---|---:|---:|
| Mean | -766.40 | 784k |
| Flat | -815.46 | 896k |
| Set mean_pool actor_detached | -1478.39 | 224k |
| Set mean_pool separate critic | -1525.51 | 672k |
| Set mean_pool shared_grad | -1511.63 | 672k |

Held-out evaluation:

| architecture | cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|
| Mean | 2.2695 | 73.5% | 702.2 m | +28.2% |
| Flat | 2.3591 | 73.0% | 729.4 m | +5.0% |
| Set mean_pool actor_detached | 4.4842 | 41.1% | 1220.4 m | -0.1% |
| Set mean_pool separate critic | 4.5496 | 40.3% | 1248.6 m | -0.1% |
| Set mean_pool shared_grad | 4.3408 | 43.2% | 1209.3 m | -0.1% |

Conclusion: the "Set is merely slower because it has more parameters" concern
has now been tested with a full 896k budget. Mean/Flat recover strong
load-bearing behavior; all three simple Set encoder paths remain far behind.
Proceed to auxiliary representation learning rather than more pure Set-MAPPO
sweeps.

## 15. 2026-06-27: HAP/UAV actor encoder sharing check

The remaining framework-level concern was whether Set fails because the HAP
actor and UAV actor share one learnable population encoder. A diagnostic switch
was added:

```text
--mec_set_actor_encoder {shared,separate}
```

`separate` gives the HAP actor and UAV actor independent population encoders
and requires `--mec_set_critic_encoder separate`, so the critic also owns its
own encoder. Tests verify disjoint optimizer ownership and role-specific
gradient routing.

112k gate:

| variant | best validation | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|
| Set mean_pool, separate HAP/UAV actor encoders, separate critic encoder | -1567.28 | 4.5959 | 39.5% | 1287.3 m | -0.0% |

This did not show an early recovery signal and should not be expanded to 896k
for now. The evidence now points away from simple HAP/UAV encoder-sharing
interference as the main cause. Continue with auxiliary representation learning
instead of more pure Set-MAPPO sweeps.

## 16. 2026-06-27: flat-MLP population encoder check

To test whether the failure was specific to self-attention / mean-pooling, a
plain ordered fixed-K MLP population encoder was added:

```text
--mec_set_encoder_type flat_mlp
```

It flattens the UAV atoms and maps them through a MAPPO-style `MLPLayer` into a
fixed-width descriptor. It is intentionally order-sensitive and supports only
pooled actor context.

The first gate used the no-sharing setting:

```text
--mec_set_encoder_type flat_mlp
--mec_set_actor_encoder separate
--mec_set_critic_encoder separate
```

400k gate:

| variant | best validation | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|
| Set flat_mlp, separate HAP/UAV actor encoders, separate critic encoder | -1470.92 | 4.1688 | 45.6% | 1166.1 m | -0.8% |

This is slightly better than mean-pool Set but still far behind Mean/Flat and
does not learn a load-bearing HAP trajectory. The failure is therefore unlikely
to be just a self-attention or mean-pool implementation bug. The suspect is now
the learnable two-stage population descriptor/readout interface under pure
MAPPO, so the next main step remains auxiliary representation learning.

## 17. 2026-06-27: Chamfer reconstruction auxiliary MVP

The user's symmetry concern is correct: `mean_pool + reconstruction` should
remain insensitive to UAV row order, so a row-wise MSE reconstruction loss is
not appropriate. A first auxiliary MVP was added:

```text
--mec_set_reconstruction_coef 0.1
```

It attaches a `PopulationReconstructionDecoder` to the shared Set actor
encoder and uses symmetric squared Chamfer set loss over normalized UAV atoms.
Tests verify target-order invariance, input-permutation invariance for
mean-pool, and that gradients reach the actor population encoder and decoder.

The first 400k gate used:

```text
--mec_policy_arch set
--mec_set_encoder_type mean_pool
--mec_set_reconstruction_coef 0.1
```

Results:

| variant | best validation | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|
| Set mean_pool + joint Chamfer auxiliary | -1511.02 | 4.6367 | 39.0% | 1250.8 m | -0.0% |

The reconstruction loss itself decreased from `0.2840` at 5.6k steps to
`0.0103` at 397.6k steps, so the auxiliary task is learnable. However, control
did not improve. Do not scale this naive joint-loss path.

Revised plan: use staged SetRec training. Collect replay states for
representation learning, pretrain `encoder + decoder` with Chamfer/Sinkhorn,
then initialize PPO from that encoder. If auxiliary updates continue during RL,
use a PPG-style phase with policy KL protection instead of mixing an
unconstrained auxiliary loss into every PPO actor update.

## 18. 2026-06-27: staged reconstruction pretraining result

The staged pipeline is implemented:

- `python -m onpolicy.scripts.train.pretrain_mec_set_reconstruction` collects
  replay UAV atoms and trains `population_encoder + reconstruction_decoder`.
- `--mec_set_pretrained_actor <actor.pt>` loads only compatible Set
  `population_encoder.*` and `reconstruction_decoder.*` tensors into PPO.
- `--mec_set_freeze_pretrained_encoder_updates N` freezes the actor population
  encoder for the first `N` PPO updates.
- TensorBoard logging now uses `add_scalar`, fixing Windows failures for tags
  containing `/`, such as `mec/uav_compute_utilization`.

Mean-pool staged pretraining:

| stage | result |
|---|---:|
| mixed replay samples | 22,400 |
| Chamfer pretrain loss | 0.0383 -> 0.0020 |
| PPO freeze-12 best validation | -1620.48 |
| PPO freeze-12 held-out cost/slot | 4.7404 |
| PPO freeze-12 HAP-freeze | -0.3% |
| PPO no-freeze 112k best validation | -1566.83 |

Latent-slot staged pretraining:

| stage | result |
|---|---:|
| mixed replay samples | 22,400 |
| Chamfer pretrain loss | 0.0441 -> 0.0017 |
| PPO no-freeze 112k best validation | -1537.72 |
| PPO no-freeze held-out cost/slot | 4.6197 |
| PPO no-freeze HAP-freeze | -0.2% |

Conclusion: pure geometry reconstruction is learnable but does not produce a
control-useful Set representation. Do not continue pure Chamfer pretraining
sweeps.

Current archive decision: pause further algorithm branches. Recent work has
not produced a substantive algorithmic improvement, and the descriptor failure
is still not explained. In particular, do not move next to Mean/Flat policy
distillation, heuristic behavior cloning, or more auxiliary-control hybrids;
those would change the problem rather than explain why the descriptor path
breaks PPO learning.

The next useful phase should be analysis of the descriptor/readout interface:
gradient scales, feature statistics, action sensitivity to individual UAV
atoms, and optimizer/loss coupling across Mean, Flat, mean_pool, flat_mlp, and
latent_slots.

## 19. 2026-06-28: public descriptor / actor-critic coupling gate

The next diagnostic stayed inside the paper's public descriptor framing.  It
did not introduce query-conditioned per-agent descriptors.  Instead, it asked
whether a public `z=f({uavs})` descriptor can support control when the
descriptor is non-learned and information-preserving, and whether Set failure
is mainly actor-side or critic-side.

New diagnostic switches:

```text
--mec_policy_arch sort_flat
--mec_critic_arch same|mean|flat|sort_flat|set
```

`sort_flat` sorts normalized UAV atoms by `(x, y, queue)` and then flattens
them.  It is a public, order-invariant, fixed-K descriptor used only as a
diagnostic.

400k seed-1 protocol: 350 slots, `mec_logstd_init=-1.2`, 403.2k environment
steps / 72 PPO updates, validation seed 1000, held-out seed 100000 with stride
13 over 24 episodes.

| variant | best validation | selected step | held-out cost/slot | accept | W1 | HAP-freeze |
|---|---:|---:|---:|---:|---:|---:|
| sort_flat actor + sort_flat critic | -1181.68 | 403.2k | 3.5106 | 55.1% | 968.9 m | +0.3% |
| Set mean_pool actor + Flat critic | -1514.10 | 302.4k | 4.6383 | 40.0% | 1230.4 m | -1.6% |
| Flat actor + Set mean_pool critic | -1433.38 | 201.6k | 4.2400 | 44.4% | 1155.3 m | -0.0% |

Interpretation:

- `sort_flat` partially recovers UAV coverage, so a public order-invariant
  descriptor is not intrinsically impossible.  But it still does not learn a
  load-bearing HAP trajectory and remains far worse than working Mean/Flat.
- `Set actor + Flat critic` remains failed, so a bad Set critic is not the
  sole explanation.  The learnable Set actor descriptor is not control-readable.
- `Flat actor + Set critic` also fails, so the Set critic path can independently
  corrupt the advantage signal even when the actor has Flat information.

Current root-cause hypothesis: public descriptor theory remains viable, but the
descriptor must be explicitly control-readable.  Pure geometry reconstruction,
vanilla Set-MAPPO, and simple actor/critic ownership changes have not created
that representation.  Next, run descriptor/readout probes and gradient/action
sensitivity audits before proposing another algorithm branch.

## 20. 2026-06-28: 1.5M rerun of the three descriptor diagnostics

The user correctly noted that the 400k/112k gates were too short to judge some
curves.  The same three diagnostics were rerun to 1.5M environment steps with
three parallel workers.  The machine handled this parallel run cleanly; all
training stderr logs remained empty.  The wrapper misclassified the run as
failed because `Start-Process` returned an empty `ExitCode` even though
checkpoint artifacts existed, so held-out evaluation was run manually from
`models\best`.  The wrapper now treats empty `ExitCode` plus complete artifacts
and empty stderr as success.

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

Important correction to section 19: the 400k conclusion that the Set critic
path independently poisons the advantage signal was premature.  `Flat actor +
Set critic` is slow early but recovers strongly by 1.5M and is the best of
these three runs.  The stable failure is specifically actor-side: `Set actor +
Flat critic` remains much worse even with enough steps.

Current root-cause hypothesis after 1.5M: the problematic path is "first learn
a shared public Set descriptor, then let each actor read fine-grained control
from that compressed descriptor plus local/env features."  Public descriptors
are still compatible with the paper goal, but the actor descriptor must be
explicitly readout-aligned/control-readable.  Later cross-attention work
confirmed this diagnosis only partially: better UAV-local readout helps, but
the result is still not strong enough to serve as the final theory-supporting
algorithm.

## 21. 2026-06-28: actor role-isolation pinpoints UAV readout

The grouped actor binding concern was addressed first with tests.  New tests
cover grouped `_features()` row binding, role-specific log-prob selection, and
hybrid actor gradient routing.  The relevant suite passes:

```text
30 passed
```

Two role-isolated actor architectures were added:

```text
--mec_policy_arch set_hap_flat_uav --mec_critic_arch flat
--mec_policy_arch flat_hap_set_uav --mec_critic_arch flat
```

`flat_hap_set_uav` also supports `--mec_set_actor_context relational`, giving
each UAV branch its equivariant token in addition to the invariant descriptor.

1.5M role-isolation results, seed 1, 350 slots, validation seed 1000,
held-out seed 100000 stride 13 over 24 episodes:

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

Conclusion: the root cause is now more specific than "Set actor descriptor is
not readable."  The failing part is the UAV actor readout from Set information.
HAP can use a Set descriptor when the UAV branch keeps Flat information.
Pooled Set-UAV remains failed at 1.5M, and relational tokens improve it but do
not solve it.

Next algorithm direction at that point: design a stronger UAV-local equivariant
readout over Set tokens/slots, while retaining the invariant/public descriptor
for global coordination.  Cross-attention was then tested and gave a real but
insufficient gain.  Do not spend more budget on pure pooled Set actor PPO
sweeps, and do not move to Flat-teacher or behavior-cloning warmups as the next
step.

## 22. 2026-06-28: Set-UAV cross-attention readout is the first working fix

Implemented `--mec_set_actor_context cross_attention` for Set actors and
`flat_hap_set_uav`.  The encoder still learns the invariant public descriptor
`z=f({uavs})` and equivariant UAV tokens; query-conditioned computation is only
in the downstream UAV readout.  This is compatible with the paper framing:
the descriptor is still a group representation, while the actor decoder reads
it through local UAV queries.

New tests cover cross-attention equivariance, invalid flat-MLP/token readout
combinations, and Set-UAV gradient routing.  Verification:

```text
33 passed
git diff --check  # CRLF warnings only
```

1.5M seed-1 comparison, all with `flat_hap_set_uav`, `mean_pool` Set encoder,
Flat critic, validation seed 1000 and held-out seed 100000:

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

Conclusion: the root-cause diagnosis is experimentally supported, but the
algorithm is not yet paper-ready.  The pooled public descriptor path is the
failure mode; UAV-local/equivariant token readout improves it, and
cross-attention improves it substantially.  However, the new branch is still
weaker than the Flat-UAV references (`Set-HAP + Flat-UAV` cost/slot 2.2900 and
`Flat actor + Set critic` 2.2201).  Archive this as a partial algorithmic
advance, not as the final SetRec architecture.

Current stop decision: do not continue with Flat-teacher warmup, heuristic
behavior cloning, or more auxiliary-control hybrids.  Those may improve a
controller, but they would no longer answer the paper's central architecture
question.  The next phase should be a written theory/architecture redesign:
define exactly what public group representation is claimed, what local decoder
queries are allowed, and what minimal trainable readout can support that claim
without importing an ordered Flat teacher.
