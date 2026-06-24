# MEC 换机训练交接摘要（2026-06-24）

> 本文是当前阶段的唯一交接入口。实际操作见
> [`docs/mec_runbook.md`](docs/mec_runbook.md)，系统模型和历史设计依据见
> [`docs/mec_env_port_spec.md`](docs/mec_env_port_spec.md)。

## 1. 当前阶段

已经完成：

- v6 continuous workload 环境、接入、队列、计算和连续 28 GHz 回传实现。
- 可复现性修复和 CPU 训练参数确认。
- `v6_continuous_workload` 的 MAPPO seed 1/2/3，各训练 1.5M steps。
- 三个模型在相同 24 个测试 episode 上的统一评估。
- HAP 承重区间扫描，以及工程约束下的新候选场景。

尚未完成：

- 新候选场景的短程参数确认训练。
- SetRec population descriptor 和置换不变 critic。
- 论文最终 5-seed、跨 K 和统计置信区间实验。

因此下一台机器的任务不是直接跑论文最终实验，而是先完成
`v6_hap_loadbearing` 的 3-seed 短程确认。

## 2. 已完成训练的结论

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
- 当前代码仍是 3 维 UAV 均值 descriptor，不是论文最终 SetRec-MAPPO。

这些结果是算法和环境诊断基线，不是论文最终结果。

## 3. HAP 承重扫描

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

## 4. 当前代码入口

| 文件 | 作用 |
|---|---|
| `onpolicy/envs/mec/scenarios/v6_hap_loadbearing.yaml` | 换机短程训练候选 |
| `onpolicy/envs/mec/scenarios/v6_continuous_workload.yaml` | 已完成 3-seed 的旧基线 |
| `onpolicy/envs/mec/finite_k_env.py` | workload、接入、回传、队列和代价 |
| `onpolicy/envs/mec/config_loader.py` | 配置推导，含显式 `link_margin_db` |
| `onpolicy/envs/mec/MEC_env.py` | 多智能体适配和系统指标 |
| `onpolicy/algorithms/mec/mec_policy.py` | 当前 mean-descriptor major/minor actor |
| `onpolicy/scripts/analysis/design_v6_sanity.py` | 场景静态/rollout probe |
| `onpolicy/scripts/analysis/scan_v6_hap_loadbearing.py` | HAP 链路预算扫描 |
| `onpolicy/scripts/eval/eval_mec.py` | policy/heuristic/hover/random 评估 |

## 5. 换机后的执行顺序

1. 安装依赖并确认 CUDA。
2. 跑 MEC 单元测试。
3. 对 `v6_hap_loadbearing` 跑训练前 probe。
4. 做 1-episode/小步数 smoke。
5. 先跑 300k-500k steps、seed 1/2/3。
6. 统一评估三个 seed。
7. 只有三类动作均能稳定学习，才进入 1.5M steps 或 SetRec 实现。

短程实验命名建议：

```text
v6_hap_lb_probe_seed1
v6_hap_lb_probe_seed2
v6_hap_lb_probe_seed3
```

完整命令见 [`docs/mec_runbook.md`](docs/mec_runbook.md)。

## 6. 短程训练通过标准

不能只看 reward。至少要求：

- cost 随训练下降，且不存在 accepted 主动下降的投机策略。
- `n_hotspot_uav` 大致保持在 5-7，背景 UAV 不消失。
- `mec/backhaul_utilization` 对 HAP 几何有响应。
- beta 对 UAV 队列正向响应，对 HAP 拥塞有抑制响应。
- 冻结 learned HAP 后 cost 明显增加，目标暂定大于 10%。
- overflow 不长期爆炸。
- 三个 seed 中不能只有一个 seed 使用 HAP。

若短程训练仍出现 HAP 失效，优先检查：

1. major 在 batch 中只有 1/(K+1) 的样本份额；
2. actor 仅使用 UAV 状态均值，存在群体构型混叠；
3. centralized critic 使用有序拼接，尚非置换不变；
4. checkpoint 目前只保留 latest，正式长训练前应增加 step checkpoint。

## 7. 文档职责

| 文档 | 职责 |
|---|---|
| `README.md` | 项目总览 |
| `HANDOFF.md` | 当前阶段状态和下一步 |
| `docs/mec_runbook.md` | 换机命令和运行步骤 |
| `docs/mec_env_port_spec.md` | 系统、参数、历史决策和技术依据 |
| `WORK_NOTES.md` | 按日期记录实际工作 |
| `docs/diagnosis_layout_not_loadbearing.md` | v2/v3 历史失败证据，仅作 archive |

本轮没有删除 Markdown。六份文件仍有不同职责，但不再把历史场景写成当前主线。
