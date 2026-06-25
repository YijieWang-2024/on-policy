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

- 环境局部观测统一为 `[role, own(3), p(7)]`，共 11 维；
- runner 统一构造 `[p(7), s_1(3), ..., s_K(3)]` centralized state；
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
第一阶段的结构正确性已经通过测试；下一步进行 `mean/flat/set` 同预算三 seed
对照，reconstruction 只在 Set-MAPPO 稳定后加入。

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
