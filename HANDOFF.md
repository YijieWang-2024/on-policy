# MEC v6 训练前交接文档(2026-06-23)

> 用途:把当前 MEC 环境探索阶段交给另一台训练机器或另一个 AI。读完本文、`docs/mec_runbook.md`
> 和 `docs/mec_env_port_spec.md` 第 12 节即可接手。
>
> 当前项目根目录:`/Users/qiaonan/Projects/on-policy`。
>
> **当前主线 = `v6_continuous_workload`。** v6 已完成训练前参数诊断,可以进入换机器训练确认阶段;
> 但它还不是论文最终参数,最终结论必须由训练后的动态指标确认。

---

## 0. 当前状态

最近一轮工作已经把 v5 的 finite-device / probability 脚手架替换为 v6 的
**normalized continuous workload field + finite-K UAV control**:

- 地面需求直接建模为 `lambda(omega, Z_t)` [bits/(m^2 slot)],不再用
  `base_probability * packet_size * device_count` 推 workload。
- 接入层不再按单设备分带宽,而是每架 UAV 拥有聚合接入频谱池。
- 回传从 60GHz + 氧吸收 + hard cutoff 改成 28GHz continuous Shannon rate。
- 计算强度从 4000 cycles/bit 改为 500 cycles/bit,避免系统被极端计算瓶颈压死。
- 新增训练前诊断脚本,用真实信道积分输出 access capacity、实际频谱效率、source loss 分解、
  hotspot/background 接纳、回传/计算利用率,不再用固定 `eta_ref` 当核心依据。

当前 v6 的 deploy-and-hold 探针显示:32 个随机热点中心下,最优热点 UAV 数落在 5-7 架:

```text
best n_hot counts over 32 centers: {5: 14, 6: 10, 7: 8}
```

这说明 v6 已经把 v5 的"16 架全挤热点"变成明显劣解,形成了训练值得验证的两层分布候选。

---

## 1. 当前代码/文件入口

核心改动文件:

| 文件 | 作用 |
|---|---|
| `onpolicy/envs/mec/scenarios/v6_continuous_workload.yaml` | 当前主场景参数 |
| `onpolicy/envs/mec/finite_k_env.py` | continuous workload field、连续接入/回传、诊断 info |
| `onpolicy/envs/mec/config_loader.py` | v6 配置解析、continuous mmWave 回传参数 |
| `onpolicy/envs/mec/MEC_env.py` | 把 v6 诊断指标透传给 runner |
| `onpolicy/runner/shared/mec_runner.py` | TensorBoard/console 记录 v6 诊断指标 |
| `onpolicy/scripts/analysis/design_v6_sanity.py` | 训练前静态/rollout 探针 |
| `requirements.txt` | 已 pin `tensorboard==2.20.0` |

文档角色:

| 文件 | 当前处理 |
|---|---|
| `HANDOFF.md` | 当前交接入口,以 v6 为准 |
| `docs/mec_runbook.md` | 换机器训练/评测/诊断手册 |
| `WORK_NOTES.md` | 历史工作日志,最新条目写 v6 |
| `README.md` | 项目总览,只放简版 MEC 说明 |
| `docs/mec_env_port_spec.md` | 长规格/历史账本,第 12 节记录 v6 |
| `docs/diagnosis_layout_not_loadbearing.md` | v2/v3 失败诊断历史,保留为反例证据 |

没有建议删除的 Markdown。`docs/mec_runbook.md` 虽然之前是 v2 runbook,但正好适合改成换机器训练手册。

---

## 2. v6 默认参数

### 区域和平台

| 参数 | 当前值 | 说明 |
|---|---:|---|
| 区域 | 6 km x 6 km | 区域级空中 MEC |
| `K` | 16 | 4x4 UAV 群体 |
| `H_U` | 300 m | 低空 UAV 接入 |
| `H_H` | 1.5 km | mobile aerial computing hub,不要称严格 HAPS |
| slot | 1 s | Mbps/Mbits 对齐 |
| episode | 200 slots | 静态热点 + 随机初始部署 |

### Continuous workload field

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `A_tot` | 150 Mbit/slot | 默认中高负载,训练主候选 |
| `zeta` | 0.70 | 70% workload 在热点,30% 在背景 |
| `sigma_h` | 700 m | 热点宽度 |
| hotspot center | U([0.3,0.7]^2) | 每 episode 静态随机 |
| swarm/hub centroid | U([0.3,0.7]^2) | 每 episode 随机初始化 |

### 接入层

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `W_ac_total` | 40 MHz | 保守 sub-6 聚合接入频谱池 |
| `W_ac_i` | 2.5 MHz | K=16 时每 UAV 聚合接入带宽 |
| `f_ac` | 2.4 GHz | sub-6 接入 |
| `p0` | 1e-8 W/Hz | low-power IoRT effective PSD |
| `gamma_ref` | 8 dB | service-attractiveness reference,不是 hard threshold |
| `tau` | 0.2 | sharp but continuous service share |

注意:当前 `eta_mean ~= 5-6 bit/s/Hz` 是 workload/service-share 加权后的 served-field 指标,
不是全空间无条件平均频谱效率。训练和论文解释时必须保留这个限定。

### 计算层

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `cycles_per_bit` | 500 cycles/bit | 默认计算强度 |
| `F_U` | 2 GHz/UAV | 每架 UAV 本地计算 |
| `F_H` | 45 GHz | Hub 计算 |
| offered compute ratio | 75/77 ~= 0.97 | 默认计算层偏紧但未压死 |

### 回传层

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `W_bh_total` | 400 MHz | mmWave 总回传带宽 |
| `W_bh_i` | 25 MHz | K=16 时每 UAV 回传带宽 |
| `f_bh` | 28 GHz | continuous mmWave backhaul |
| `P_UH` | 27 dBm | UAV 回传发射功率 |
| effective gain | 15 dB | 必须解释为 beamforming gain + fixed losses 后的 effective value |
| hard cutoff | no | 不使用 SNR 截断 |

---

## 3. 训练前诊断命令

本机使用 anaconda 的 `marl` 环境:

```bash
cd /Users/qiaonan/Projects/on-policy
PYTHONPATH=$PWD /opt/anaconda3/envs/marl/bin/python -m onpolicy.scripts.analysis.design_v6_sanity
```

期望看到的关键结果大致如下:

```text
K=16  A_tot=150.0 Mbit/slot  zeta=0.70
offered_compute_ratio=0.97
best n_hot counts over 32 centers: {5: 14, 6: 10, 7: 8}

* 5     acc ~=125.7M  src ~=24.3M(out ~=15.8M, cap ~=8.4M)  util ~=0.56
  6     acc ~=125.4M  src ~=24.6M(out ~=19.1M, cap ~=5.5M)  util ~=0.55
  7     acc ~=122.9M  src ~=27.1M(out ~=22.7M, cap ~=4.4M)  util ~=0.53
 16     acc ~=104.0M  src ~=46.0M(out ~=46.0M, cap ~=0.0M)  util ~=0.38
```

解释要点:

- `n_hot=5-7` 是稳定候选,不要死卡 5 架。
- `n_hot=16` 虽然总 capacity 更高,但背景被 outside option 吃掉,accepted 更低。
- 当前 source loss 同时来自 outside attractiveness 和 capacity,不是纯接入带宽饱和。
- access utilization 约 0.5-0.6,说明接入层没有过度富裕,但默认场景不是纯带宽饱和系统。
- hub 偏移 3km 时回传利用率上升并出现少量 overflow,说明回传几何承重但没有硬断链。

---

## 4. 当前已验证命令

```bash
cd /Users/qiaonan/Projects/on-policy

PYTHONPATH=$PWD /opt/anaconda3/envs/marl/bin/python -m pytest \
  onpolicy/envs/mec/tests/test_config_loader.py \
  onpolicy/envs/mec/tests/test_finite_k_env.py \
  onpolicy/algorithms/mec/tests/test_mec_policy.py \
  -q
# 17 passed, 1 skipped, 1 warning

PYTHONPATH=$PWD /opt/anaconda3/envs/marl/bin/python -m compileall -q \
  onpolicy/envs/mec onpolicy/scripts/analysis/design_v6_sanity.py

git diff --check
```

---

## 5. 换机器训练命令模板

在 GPU 机器上 `conda activate marl` 后运行。注意本仓库沿用 `store_false` 开关:

- **想用 GPU:不要传 `--cuda`**。
- **想用本地 TensorBoard:传 `--use_wandb`**。

```bash
PYTHONPATH=$PWD python -u -m onpolicy.scripts.train.train_mec \
  --env_name MEC --algorithm_name mappo \
  --experiment_name v6_continuous_seed1 \
  --mec_scenario v6_continuous_workload \
  --seed 1 \
  --n_rollout_threads 16 --n_training_threads 6 \
  --episode_length 200 --num_env_steps 1500000 \
  --ppo_epoch 5 --num_mini_batch 1 \
  --hidden_size 128 --layer_N 2 \
  --use_entropy_anneal --mec_logstd_init -1.9 --entropy_coef 0.003 \
  --lr 5e-4 --critic_lr 5e-4 --gamma 0.99 \
  --log_interval 5 --save_interval 25 \
  --use_wandb
```

建议先跑 seed 1。若机器资源允许,再跑 seed 2/3。当前电脑不适合训练,本阶段只做到了训练前诊断。

---

## 6. 训练后必须看的指标

不能只看 reward 和 W1。训练后验收至少看:

- `mec/n_hotspot_uav`:多数 episode 末态应在 5-7 左右。
- `mec/n_background_uav`:应保留大约 9-11 架背景覆盖力量,不应接近 0。
- `mec/source_outside` 与 `mec/source_capacity`:区分 outside option loss 与接入容量 loss。
- `mec/hotspot_accepted`, `mec/background_accepted`:背景不是只要有 UAV,还要真的接纳 workload。
- `mec/eta_p05`, `mec/eta_p50`, `mec/eta_p95`, `mec/eta_served`, `mec/eta_all`:
  防止平均频谱效率被强链路区域抬高。
- `mec/access_utilization`, `mec/backhaul_utilization`, `mec/uav_compute_utilization`,
  `mec/hub_compute_utilization`:检查三个层级是否在合理量级。
- `mec/overflow`, `mec/U_src`:不能长期爆炸。
- `mec/hub_to_hotspot`:hub 轨迹应对几何有贡献。

目标不是"刚好 5 架",而是:

```text
热点 UAV 数稳定在 5-7,
背景 UAV 数稳定在 9-11,
全员挤热点明显劣于两层分布,
source loss/overflow 不爆,
Hub 偏移或冻结会降低性能,
W1 与分布/吞吐指标共同合理。
```

如果训练后仍学成全员挤热点,优先怀疑 reward/策略表达/探索和动态训练过程,不要立刻把物理参数继续乱调。

---

## 7. 历史版本如何理解

| 场景 | 当前角色 |
|---|---|
| `v2_iort_6km_mmwave` | 锁定真源和数值对拍基准;不作为学习主场景 |
| `v3_iort_learnable` | 证明定位可学习的早期派生场景;移动热点/major 学习不稳 |
| `v4_static_demand` | 静态 inverse design 初版;固定起点导致纵带偏移 |
| `v4_fixed_demand` | 固定热点+固定起点导致探索死锁的反例 |
| `v5_static_randinit` | 随机初始化解决纵带偏移,但暴露全员挤热点 |
| `v6_continuous_workload` | 当前主线;训练前候选主场景 |

v5 以前的结论不是全删,它们解释了为什么 v6 要同时改 workload、接入、回传、计算和诊断指标。

---

## 8. 仍需谨慎的 caveats

- `W_ac_total=40MHz` 是保守 sub-6 接入池,不是 5G FR1 最大配置。
- `p0=1e-8 W/Hz` 是低功率 IoRT effective PSD;不要写成满功率 UE。
- `gamma_ref=8dB` 是 service attractiveness reference,不是硬门限。
- `effective backhaul gain=15dB` 不是普通物理天线总增益,它吸收了波束增益、固定损耗、实现损耗和链路裕量。
- 当前默认场景 source loss 主要可能来自 service attractiveness/outside option,不是纯带宽容量不足。
- v6 的静态 probe 不能替代完整 RL 训练。它只说明参数值得训练。
