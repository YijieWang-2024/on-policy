# MEC 环境移植规范 v2 — 单一真源(2026-06-12 重构)

> 把论文 `finiteK_measure_valued_mdp_reformulation.tex` 的 finite-K 分层空中 MEC 环境移植进本
> `on-policy` 库。本文是移植的**契约**:代码逐条对照实现并验收,任何偏离必须显式记录。
>
> **v2 重构说明**:v1 经多轮探索(SCI 复审 R1-R7 → Path-2 → 60GHz 回传 → 删带宽动作 → λ 拆分)
> 后参数/结论散落各节、互相覆盖。v2 以**最终锁定设计**为主体一次写清;所有被否决的中间设计统一
> 归档在 §9(决策历程),正文不再出现过期数值。**本文取代 v1 全部内容。**

来源真值:
- 论文:`Mean Field Mec/finiteK_measure_valued_mdp_reformulation.tex`(System Model 49–448 行)
- 底层仿真器:`Mean Field Mec/src/mfmec/env/finite_k_env.py`(`FiniteKHAPUAVMECEnv`)
- 配置加载:`Mean Field Mec/src/mfmec/config.py`;原场景 `configs/scenario/paper_iort_10km_hap8km.yaml`
- 数值体检:`/tmp/mfmec_h*.py`、`/tmp/mfmec_v12.py`(12+ 轮扫描;最终配方见 §5)

---

## 0. 任务与路线(一段话钉死)

**一个高对比度、持续漂移的地面 IoRT demand 场**(固定起点 + 每 episode 随机方向 + 噪声 + 边界反弹,
跨 episode 无固定最优点)**由 K 架同构弱算力 UAV(minor,~300m)+ 1 个低空移动算力中枢(major,~1.5km)
协同服务**。UAV 调整**位置**(匹配 demand 分布以接入数据)与**卸载比例 β**(本地算 vs 卸给中枢);
中枢调整**自身轨迹**(把 UAV 群罩在 mmWave 回传服务圈内,圈外卸不动)。
接入 = sub-6 2.4GHz(γ_th 限定 UAV 覆盖半径 ~861m);回传 = 60GHz mmWave 专用波束
(链路预算限定中枢服务半径 ~2.2km);设备功率 0.1W、UAV 回传功率 0.5W 均固定。
**期望涌现**:UAV 群空间分布 ∝ demand 密度(W₁ 距离评测,与论文测度值理论统一),中枢追随群体。
**代价份额(等权 λ,启发式参考;真实份额由训练策略涌现)**:覆盖(源失效)~30%、卸载相关(溢出+队列)~61%、
能耗 ~9% —— **处理/卸载主导、覆盖次要但承重,三类动作(UAV 位置 / β / 中枢轨迹)消融全部显著**。
**目标函数 = 最小化丢失的 fresh data(源失效与溢出等权,λ_ovf≤λ_src)**;不得用不等权人为凹份额(§9 陷阱)。

> **⚠️ 2026-06-15 训练实证补充(详见 §10)**:上面"期望涌现"在**本节锁定的 v2 参数下经真实 RL 训练被证伪**。
> 24 架 × 861m 覆盖半径对 6km 场地**过覆盖**(≈1.55×)、且 offered≈33 vs 容量 13.5 的 **2.4× 过载**,
> 使"UAV 定位 / demand-matching"对团队成本的影响仅 **~3.6%**(heuristic vs hover),**小于每-episode 外生噪声(~11%)**;
> MAPPO 跟着噪声漂移,学出的策略**反而劣于 hover**(W₁ 835m > random,成本 > hover)。
> **派生 v3 学习友好场景**(`v3_iort_learnable.yaml`:K=12 + 算力×2 + 队列×1.5 → 定位敏感度 +26%)+ 探索修复
> (`--mec_logstd_init -1.9`、`--use_entropy_anneal`)后,demand-matching **才真正从团队成本最小化中涌现**
> (eval W₁ 733m < 启发式 763 < hover 830 < random 1155;成本 0.509 ≈ 启发式、胜 hover 24%;中枢消融 +27.9%;详见 §10.3,含 log-prob 修复 + entropy 0.003)。
> **v2 仍作为锁定真源与数值对拍基准保留不变;v3 是面向"可学习"的派生场景。**

**卖点**(方法主导):finite-K 无损测度值重构 + Wasserstein descriptor 误差界 + SetRec-MAPPO,
testbed = 上述「可动算力中枢 + 可扩展 UAV 群」二层 MEC(查新无人凑齐,见 §7)。

---

## 1. 系统模型

### 1.1 节点与几何

| 节点 | 高度 | 速度 | 算力 | 角色 |
|---|---|---|---|---|
| 算力中枢 H ×1 | 1500 m | ≤30 m/s(轨迹=major 动作) | f_H=30GHz → C_H=7.5 Mbit/slot | 回传汇聚 + 集中计算 + 广播 descriptor |
| UAV ×K(默认 24) | 300 m | ≤40 m/s | f_U=1GHz → C_U=0.25 Mbit/slot | 接入采集 + 本地计算 + 卸载 |
| 地面设备(聚合场) | 0 | — | — | 产生 fresh data(一 slot 有效) |

区域 6km×6km;Δ=1s;episode 200 slot。**无 device→H 直连**(0.1W 设备够不到 1.5km 中枢的可用速率,
正是 UAV 中继的存在理由)。

### 1.2 链路与协议

| | 接入 device→UAV | 回传 UAV→H |
|---|---|---|
| 频段 | sub-6 **2.4GHz** | mmWave **60GHz(V 波段)** |
| 信道 | Al-Hourani LoS/NLoS(地面杂波) | 纯 LoS 空对空 + O2 吸收 κ≈10dB/km |
| 天线 | 全向(海量廉价设备) | **混合波束成形,每 UAV 一个专用波束**(双方位置已知 → 波束对准) |
| 多址 | UAV 间 FDMA(W_ac/K);小区内设备 OFDMA(每设备≤B_max) | **空分(专波束)+ 波束间正交子带 W_beam=25MHz/束**(双重隔离) |
| 干扰 | 小区内共享带宽;UAV 间正交 | **可忽略**(方向波束 + 正交子带) |
| 功率 | 设备 0.1W(固定) | UAV 27dBm=0.5W(固定),G_t+G_r=40dB |
| 范围 | γ_th=20dB → **覆盖半径 ~861m** | 链路预算 → **服务半径 ~2.2km**(SNR:0m=17dB→2km=0.3dB→2.3km 不可解调) |
| 跨层 | **两频段分离 ⇒ 零跨层干扰**(无需同频正交假设) | |

> 架构 = 5G/6G **IAB** 标准分工(low-band 广域 NLoS 接入 + mmWave LoS 方向回传,Ghasemi2024 同款)。
> 频谱:K×W_beam=24×25MHz=600MHz ≪ V 波段可用 ~14GHz,专波束+正交子带完全可行。
> **两条链路的 γ_th 各自限定服务几何**:接入门限 → UAV 的覆盖圈;回传链路预算 → 中枢的服务圈。
> 这两个「圈」就是两类位置动作的物理意义来源。

### 1.3 demand 场与运动

密度 `p(ω)=p_bg + p_hot·exp(−‖ω−c_t‖²/2σ_h²)`,clip 到 [0,5e-3];`λ(ω)=ρ_G·p(ω)·b₀`。
**运动(Markov,关键设计)**:每 episode 从**固定起点**(0.2R,0.2R)出发,方向 θ~U[0,2π) **每 episode
随机**,速度 25 m/s,每 slot 加 N(0,250m) 噪声,**边界反射**。⇒ 单 episode 内连续可追、跨 episode
路径不同 → **不存在可预调的固定最优位置**,中枢/UAV 都必须从观测学追踪。
offered ≈33 Mbit/slot(热点 ~32 + 背景 ~1);锁定配方下接纳 ~74%(~24 Mbit/slot)>
全系统算力 C_H+K·C_U=13.5 Mbit/slot ⇒ **稳态过载,处理是硬瓶颈**(灾后重载场景,合理)。

---

## 2. 决策变量(major-minor)

### 2.1 变量表(全部与 K 无关)

| 变量 | 归属 | 维度 | 约束 | 先例 |
|---|---|---|---|---|
| `v^H` 中枢速度 | **major 自有动作** | 2 | L2 ≤ V_H_max(盘投影) | Cui M3FMARL 式(1b):major 只输出自身动作 |
| `v_i` UAV 速度 | minor **共享策略** | 2/架 | L2 ≤ V_U_max | Cui 式(1a);Gu2026-position |
| `β_i` 卸载比例 | minor 共享策略 | 1/架 | [0,1],Beta 策略头 | Gu2026 λ;Bao2025 |

联合动作 = major 2 维 + 每 minor 3 维。**没有带宽分配变量**:

> **▶ 为何没有 `d_i`(带宽需求动作)— 五角度反驳(优化变量必要性)**
> 回传是 mmWave **每 UAV 专用波束**(空分)+ 正交子带(W_beam=25MHz 固定):
> ① 波束间无干扰、频谱充裕(600MHz≪14GHz),**带宽不构成跨 UAV 竞争**;
> ② 圈内速率 145→30 Mbit/slot **全程 ≥ 卸载需求**(~10–30),圈外恒 0 —— **绑定约束是几何
> (在不在服务圈),不是带宽切分** → `d_i` 没有学习信号;
> ③ 真正的共享稀缺 = **中枢算力 C_H**(由 β 经 Q_H 排队仲裁)与**服务圈名额**(由位置仲裁),
> 二者已有对应决策变量。
> **结论:删除 `d_i`(及配套双模式分配机制,归档 §9)。三个稀缺资源 ↔ 三类动作一一对应:
> 接入几何↔v_i、回传几何↔v^H、中枢算力↔β_i。系统更干净,policy 头更小。**

### 2.2 major-minor 依据(Cui M3FMARL)

系统态 =(major 态 x⁰, UAV 经验测度 μ);major 动作通过动力学影响全体 minor 即成立:
**中枢轨迹 → 全体 UAV 的回传可达性(在不在 2.2km 圈内)→ 卸载成败 → 队列/溢出**,强耦合。
major 输出自身 2 维动作(与 K 无关,Cui 式 1b);minor 共享策略(式 1a)。消融实证:固定中枢比
追踪贵 **+23%**(§5.1)——major 动作真实承重。若日后想再强化 major,可加 O(1) 广播信号
(拥塞价格,类 MFG),仍与 K 无关。

---

## 3. 转移动力学(`step()` 顺序)

1. 动作投影(速度盘 clip、β clip 到 [0,1])→ 服务位置按 `service_position` 配置(**锁定 mid_move**:
   等速 slot 中点位置服务、slot 末推进状态——§5.1 体检数值即此设置;支持 pre/mid/post 可配,R5)。
2. demand 场更新:§1.3 随机游走 + 反射。
3. **接入**(sub-6):LoS/NLoS → 带 outside-option 的平滑 logit 服务分流(温度 τ)→ 等价服务设备数
   → 单设备带宽(≤B_max)→ Shannon → 接纳 `A_i`、源失效 `U_src`。
4. **回传**(60GHz 专波束,参考实现已验证):
   ```python
   def _backhaul_rate(self, hap_xy, uav_xy):
       p = self.cfg["communication"]["backhaul"]["mmwave"]
       W = float(p["beam_bandwidth_hz"])                      # 25 MHz, 固定, 与 K 无关
       H_H = float(self.cfg["env"]["hap"]["altitude_m"]); H_U = float(self.cfg["env"]["uav"]["altitude_m"])
       horiz = np.linalg.norm(uav_xy - hap_xy[None, :], axis=1)
       d = np.sqrt(horiz**2 + (H_H - H_U)**2)
       fspl = 20*np.log10(np.maximum(d, 1.0)) + 20*np.log10(p["fc"]) + 20*np.log10(4*np.pi/2.99792458e8)
       pl = fspl + p["kappa_o2_db_per_km"] * (d / 1000.0)
       snr_db = p["Ptx_dbm"] + p["G_tot_db"] - pl - (-174 + 10*np.log10(W) + p["NF_db"])
       rate = W * np.log2(1 + 10**(snr_db / 10.0))
       return np.where(snr_db >= p["gamma_min_db"], rate, 0.0)   # 圈外卸不动
   ```
5. 队列:`S_i^U=min((1−β_i)Q_i, C_U)`;`B_i=min(β_i Q_i, Δ·R_i^bh)`;UAV 队列更新+溢出 `D_i^U`;
   中枢 `S_H=min(Q_H,C_H)`、队列更新+溢出 `D_H`。(β 只作用旧缓存,不作用同 slot 新接纳。)
6. 能耗:UAV Zeng 旋翼飞行 + DVFS 计算 `κ·c₀³/Δ²·bits³` + 回传发射;中枢 Fan-surrogate 飞行 + 计算。
7. 安全:置换不变软分离代价 `C_sep`(÷C(K,2))。
8. reward = −c(§4);`info` 带全部 raw 分项(`A_i/B_i/D_*/U_src/R_i^bh/能耗分项`)。

---

## 4. 代价 / reward

```
c = ω_Q·(ΣQ_i + Q_H)  +  ω_src·U_src  +  ω_ovf·(ΣD_i^U + D_H)  +  ω_E·(Σe_i + e_H)  +  ω_S·C_sep
ω_Q   = λ_Q   / (K·Q_U_max + Q_H_max)      # 公式派生,K-依赖(R2,禁止硬编码)
ω_src = λ_src / D_ref                       # 源端失效 = 覆盖子系统失败
ω_ovf = λ_ovf / D_ref                       # 队列溢出 = 处理/卸载子系统失败
ω_E   = λ_E   / E_ref ;  ω_S = λ_S / max(1, C(K,2))
λ = (λ_Q, λ_src, λ_ovf, λ_E, λ_S) = (1, 5, 5, 0.03, 0.5)      # 锁定;**约束 λ_ovf ≤ λ_src**
```

- **λ_src = λ_ovf(2026-06-15 关键修正,推翻早先的 8:3)**:一个 fresh bit 无论在**源端失效**还是
  **队列溢出**,都是同样丢失的数据 → 两者**等权**才是诚实目标。早先为「凑覆盖份额 20-30%」把
  λ_ovf 设成 8 > λ_src=3,结果一次 **2e6 步长训练暴露致命反向激励**:策略学会**故意少接纳数据以
  规避溢出**(`accepted 19.5M→5.0M`、`U_src 4.8M→16.9M`、ovf 份额 .59→.13),把昂贵的溢出转成廉价的
  源失效,**直接压制了 demand-matching 卖点**。启发式体检看不出(无脑覆盖),只有优化策略会钻空子。
  **铁律:绝不允许 λ_ovf > λ_src**(否则奖励「别干活」);要更强调覆盖可令 λ_src > λ_ovf,但默认等权。
  份额是**策略涌现**的结果,不用不等权去人为凹(那正是 bug 源头)。
- **两层归一化(R1)**:env 只返回加权物理代价(ω 是目标定义的一部分,ref 是量纲换算、不可删);
  RL 尺度归一化交 PopArt/ValueNorm,env 内**不**叠 running-std。评测报告原始物理量。

---

## 5. 全参数表(锁定,单一真源)

| 组 | 参数 | 锁定值 | 备注 |
|---|---|---|---|
| 场景 | 区域 / Δ / T / K | **6km×6km / 1s / 200 / 24**(K 扫 8/16/24/32) | 区域 >2× 回传半径 → 中枢必须追 |
| 中枢 | H_H / V_H_max / f_H / Q_H | **1500m / 30m/s / 30GHz / 150Mbit** | C_H=7.5 Mbit/slot |
| UAV | H_U / V_U_max / f_U / Q_U | **300m / 40m/s / 1GHz / 80Mbit** | C_U=0.25 Mbit/slot |
| 计算 | c₀ | **4000 cyc/bit** | 重计算任务(视频/CV) |
| demand | ρ_G / b₀ / p_bg / p_hot / σ_h / clip | **1e-3 /m² / 1Mbit / 2e-5 / 8e-3 / 800m / 5e-3** | offered≈33 Mbit/slot,高对比度 |
| demand 运动 | 起点 / 速度 / 噪声 / 方向 | **(0.2R,0.2R) / 25m/s / N(0,250m) / 每 episode U[0,2π)** | 边界反射;跨 episode 无固定最优点 |
| 接入 | fc / W_ac / P_dev / γ_th / B_max / τ | **2.4GHz / 80MHz(FDMA→3.3MHz/UAV) / 0.1W / 20dB / 1MHz / 0.2** | 覆盖半径 ~861m;LoS (a1,a2)=(9.61,0.16), 1/20dB |
| 回传 | fc / P_tx / G_tot / κ_O2 / NF / γ_min / W_beam | **60GHz / 27dBm / 40dB / 10dB/km / 8dB / 0dB / 25MHz/束** | 服务半径 ~2.2km;专波束+正交子带 |
| 能耗 | UAV 旋翼 / 中枢 | Zeng2019(P₀=79.86 等)/ Fan-surrogate(P_H⁰=500W 等) | 不变(R7) |
| 安全 | d_min | 30m | 软惩罚 |
| 代价 | λ / D_ref / E_ref | **(1,5,5,0.03,0.5) / 1e8 / 1e4** | §4;λ_ovf≤λ_src 铁律;ω_Q、ω_S 公式派生 |

### 5.1 体检结果(等权 λ,覆盖+卸载启发式,3 seed × 随机方向)

> 启发式(无脑覆盖,**不会钻 λ 空子**)用于验证 testbed regime + 仪表;真实份额/W₁ 由**训练策略**定。
> 份额是**策略涌现**结果,下列为启发式参考值,非凹出来的目标。

| 指标 | 值(等权 λ) | 判读 |
|---|---|---|
| 接纳率 | 82% | 覆盖基本解决 |
| 份额:源失效(覆盖) | 30% | 次要 |
| 份额:溢出+队列(卸载相关) | 48%+13% = 61% | 主导 |
| 份额:能耗 | 9% | 正则量级 |
| **中枢消融**(固定中心 vs 追踪) | **+13.5%** | major 轨迹承重 ✓ |
| **β 消融**(β≡0 vs β-aware) | **+54%** | 卸载决策承重 ✓ |
| **运动 gate**(hover vs 移动) | **+8%** | 反 hover-dominated ✓ |
| **置换不变**(env 对称) | **0.00**(精确) | 正确性 ✓ |
| **W₁ demand-match** move/hover | 529 / 577 m(+8%) | 启发式弱;训练策略应更低 |

### 5.2 调参指南

- **份额不要用不等权去凹**(那是 8:3 陷阱的来源,§9)。**铁律:λ_ovf ≤ λ_src**;要更强调覆盖/匹配
  就抬 λ_src(让「不收数据」更亏);默认等权。
- **过载度**(决定卸载/溢出有多重):f_H(30→25:溢出↑)、f_U(1→2GHz:本地算力强则卸载/溢出弱)、
  Q_U/Q_H(越小越易溢出)。
- **中枢杠杆↑**:W_beam↑(半径↓,25→100MHz: 2.2→1.5km)或 G_tot↓;须保持 区域 > 2×服务半径。
- **hover gate**:σ_h↓ 或 demand 速度↑(运动收益↑)。

---

## 6. 评测与验收

物理正确性:
- [ ] **未改动方程**(接入信道/LoS-NLoS/服务分流/DVFS/旋翼能耗)与原 env 同 seed 同动作逐步一致
      (1e-8)。**v2 改动(60GHz 回传、λ 拆分、demand 运动、新参数)不在对拍范围,改单元测试。**
- [ ] `phi_sum_error<1e-6`;派生常数与 `derive_constants` 一致;`ω_Q/ω_S` 随 K 公式重算正确。
- [ ] 60GHz 回传单元测:水平 0/1/2/2.3km → SNR 17.4/11.5/0.3/−3.2dB,2.3km 处 rate=0。

承重 / 任务(体检判据,方向性,不逐百分点):
- [ ] 代价份额:覆盖**次要**(~20-35%)、卸载相关**主导**(~55-70%)、能耗小(<15%);份额策略涌现。
- [ ] **训练策略不得退化为「少接纳」**:accepted 不随训练显著下降、U_src 不暴涨(等权 λ 的核心验收,§9 陷阱)。
- [ ] 中枢消融 ≥ +10%(60GHz);**2.4GHz 对照 ≤ ±1%**(验证机制来自服务半径)。
- [ ] β 消融 ≥ +30%;运动 gate(hover)≥ +8%。
- [ ] **demand-matching**:训练后 `W₁(μ_UAV, μ_dem)` 显著低于 Hover/Random/均匀铺开;
      UAV 经验测度 `μ_UAV=(1/K)Σδ_{x_i}`,demand 测度 `μ_dem ∝ λ(ω,Z_t)`。
- [ ] 标签置换检验:打乱 UAV 标签,团队 reward 分布不变(env 已验 0.00)。

on-policy 集成:
- [ ] 空间形状自洽(major 2 维 + K×3 维);major-minor + 连续动作跑通 shared MAPPO 一个 episode。
- [ ] truncation→bootstrap(D3);K=8/16/24/32 不改任何网络形状即可跑(可扩展性)。

---

## 7. 查新与文献定位(读 12 篇邻居,2026-06-11)

| 论文 | MF/PI | UAV 群轨迹 | 可动算力中枢(学轨迹) | 两级卸载 | 缺口 |
|---|---|---|---|---|---|
| Gu2026-position ⚠️最近 | ✅ | ✅ | ❌ 固定地面 BS | ✅ | 无空中可动中枢、无 major-minor、无测度值理论 |
| Morshed-Alam2024 | ❌ | ✅ | ❌ 固定 HAP | ✅ | 中枢不动、非 MF |
| Jia2023 / Nabi2025 / Bao2025 | ❌ | ❌ | ❌ 固定 HAP | ✅ | 不可扩展 |
| Song2025 / Emami2024 | ✅ | ✅ | ❌ | ❌ | 无两级 MEC、无 major |
| Ghasemi2024(IAB-LAP) | ❌ | — | ✅ 低空 BS,mmWave+波束成形 | 两层 | 非 RL;给链路/频段设计背书 |

**差异点**:① 空中算力中枢**可动且轨迹被学习**(major);② major-minor 用于空中 MEC(邻居全无);
③ finite-K 无损测度值 + SetRec + W₁ demand-matching 评测。**风险**:Gu2026 在「MF+位置+卸载」够近
→ 卖点落在方法+可动中枢+demand-matching,不卖「又一个 MF 卸载」。

---

## 8. on-policy 集成、算法实现与移植进度

### 8.1 算法:major-minor 共享参数 MAPPO(已实现,`onpolicy/algorithms/mec/mec_policy.py`)

**核心思想**:K 架 UAV 同质 → **共享一套 minor 策略参数**(可扩展 + 置换等变);中枢是唯一异质体 →
**单独一个 major 头**。两者共享 trunk + 条件于**置换不变的群体描述子**(mean-field)。CTDE:集中 critic。

- **MECActor**(接口与 `R_Actor` 完全一致,直插 `R_MAPPO` trainer):
  - 共享 trunk `MLPBase(obs)` → 特征;obs = 14 维(role flag + 自身态 + 中枢公共态 + demand + **UAV 群均值描述子**)。
  - **按 role flag(obs[:,0])逐行路由**两个头:
    * major 头 → `v^H`:2 维对角高斯(`μ=Linear`,`logσ` 为可学参数);
    * minor 头 → `v_i`:2 维对角高斯;`β_i`:**Beta(α,β)**,`α,β=softplus(Linear)+1`(支撑 [0,1],无边界偏置,R6)。
  - 存储统一 3 维动作 `[vx,vy,β]`;**major 的 β 槽位填 0 dummy,其 log-prob/entropy 只算 2 维速度**。
  - `logπ`:major = `logN(v^H)`(2 维求和);minor = `logN(v_i)` + `logBeta(β_i)`;按 role 合并 → `[batch,1]`。
- **MECPolicy**:`R_MAPPOPolicy` 子类,只把 actor 换成 MECActor,critic 复用 `R_Critic`(吃 `share_obs`=全局拼接)。
- **PPO 更新**(trainer 不改):batch = `n_envs×(K+1)` 展平;`ratio=exp(logπ_new−logπ_old)`,major/minor
  各用自身分布算,公式前后一致(已单元测验证 **ratio 精确=1**);**K 个 minor 行共享同一头 → minor 头拿
  K 倍梯度、major 头 1 倍**,即 major-minor 的自然样本配比。velocity 不 squash,由 env 投影到速度盘(库内 Box 惯例)。
- **当前 = mean 描述子(DeepSets 级)**;论文的 **Set Transformer + Sinkhorn 重构**是 drop-in 升级:需在
  forward 里把 batch reshape 回 `(n_env, n_agent, ·)` 做跨 UAV 注意力池化(破坏当前扁平 batch,故留作下一步)。

**有效训练要点(2026-06-15 实证更新,详见 §10)**:**前提是环境让定位承重(v3),否则再调训练也学不动**。
在 v3 上验证有效:`--use_valuenorm`(团队代价幅值小,必开);**`--mec_logstd_init -1.9`(σ≈0.15,杜绝速度满盘
乱窜——这是 v2 失败的直接机制之一);`--use_entropy_anneal --entropy_coef 0.003`(低探索任务,0.01 偏高;
熵线性退火到 0 让策略收敛)**;`--ppo_epoch 5`(过载噪声大,10 易过拟合噪声);`--n_rollout_threads 16`(降梯度方差,实测够用)。
K 个 minor 行共享同一头但**优势相同、样本高度相关**,有效 batch≈环境数(非 ×K)——这也是为何信号弱时学不动。
**per-UAV 局部奖励(吞吐项)** 是进一步增强信用分配/压低 W₁ 的可选项;v3 上不加即已学会,故暂留接口未启用。

### 8.2 集成决策(D1–D6,as-built)
- **D1 = D1a 折叠版**:单一 nn.Module 内 role 路由双头(major 头 + 共享 minor 头),buffer 仍同质(N=K+1、
  统一 3 维动作)→ trainer/buffer/runner 全部不改。(纯 D1b「padding+role one-hot+单头」已弃,因 major/minor
  共享单头会稀释学习。)
- **D2**:团队标量 reward 广播全 agent;**D3**:`terminated≡False`,horizon 用 `truncated`(bootstrap)。
- **D4**:`mec_runner._actions_to_env` Box 直通(env 内投影);β 用 Beta 头(R6)✅。
- **D5**:`--mec_scenario` 选 YAML;K 由场景定。**D6**:`num_agents=K+1` 由 probe-reset 探测。

### 8.3 文件布局(as-built)
```
onpolicy/envs/mec/{config_loader,finite_k_env,MEC_env}.py
onpolicy/envs/mec/scenarios/v2_iort_6km_mmwave.yaml          # §5 锁定参数
onpolicy/envs/mec/tests/{test_config_loader,test_finite_k_env,test_health_check}.py
onpolicy/algorithms/mec/mec_policy.py                         # MECActor + MECPolicy(本节)
onpolicy/algorithms/mec/tests/test_mec_policy.py             # ratio=1 / 梯度 / 路由 一致性
onpolicy/runner/shared/mec_runner.py                         # Box 直通 + MEC 指标日志
onpolicy/scripts/train/train_mec.py ; config.py(--env_name MEC)
```
训练命令:
```
python -m onpolicy.scripts.train.train_mec --env_name MEC --algorithm_name mappo \
  --mec_scenario v2_iort_6km_mmwave --n_rollout_threads 8 --episode_length 200 \
  --ppo_epoch 10 --hidden_size 128 --use_valuenorm   # --cuda/--use_wandb 为 store_false(传即关)

# v3 学习友好场景(实测能学出 demand-matching,§10);CPU/K=12 约 1 小时 / 1.5e6 步
python -m onpolicy.scripts.train.train_mec --env_name MEC --algorithm_name mappo \
  --mec_scenario v3_iort_learnable --seed 1 --n_rollout_threads 16 --episode_length 200 \
  --num_env_steps 1500000 --ppo_epoch 5 --hidden_size 128 --layer_N 2 \
  --use_entropy_anneal --entropy_coef 0.003 --mec_logstd_init -1.9 --use_wandb --cuda
```

### 8.4 移植进度
1. ✅ 冻结规范 v2。
2. ✅ `config_loader.py` + `v2_iort_6km_mmwave.yaml`;单元测 6/6(ω 公式派生 R2、60GHz 链路预算)。
3. ✅ `finite_k_env.py`:对拍未改动方程(1e-8)+ v2 改动(60GHz 回传、λ 拆分、随机游走、低空中枢);单元测 6/6。
4. ✅ `MEC_env.py`(N=K+1 适配器)+ `mec_runner.py`(D4)+ `train_mec.py`;shared MAPPO smoke 跑通,K=8/16/24/32 同形状可扩展。
5. ✅ **major-minor `mec_policy.py`(§8.1)**;单元测 4/4(ratio=1、梯度、路由、policy 一致);接入 base_runner。
6. ✅ §6 体检(启发式级):份额 27/49/14/10、β+44%/hover+24%、置换=0.00;中枢+6%(启发式偏弱,待训练复测)。
7. ✅ **真训练到收敛(2026-06-15,详见 §10)**:v2 锁定参数下 RL **学不出** demand-matching(过覆盖+过载致成本对
   定位仅~3.6% 敏感,策略漂移到劣于 hover);**派生 v3 学习友好场景**(K=12+算力×2+队列×1.5)后训练成功——
   eval W₁ 733m(< 启发式 763 < hover 830 < random 1155)、团队成本 0.509(≈启发式 0.500、胜 hover 0.674 约 24%)、
   中枢消融 +27.9%(修复代码 + entropy 0.003,§10.3)。(可选)Set Transformer 描述子 / per-UAV 局部奖励仍待做。


## 9. 决策历程(归档:被否决/被取代的设计,防止回头路)

| 设计 | 否决原因 | 取代为 |
|---|---|---|
| 8km HAP + 可动(R3 原案) | 8km 下水平移动对几何无关(Δslant<0.5km/8km) | 1.5km 低空中枢 |
| device→H 直连链路 | 0.1W 设备差 ~26dB 够不到;破坏 UAV 中继动机;陷入 IAB 拥挤赛道 | 仅两条链路 |
| major 输出 K 维分配头 | 输出维度随 K 变,破坏可扩展;Cui 式(1b) major 只出自身动作 | major 只出 `v^H` |
| `d_i` 带宽需求动作 + proportional/penalty 双模式 | 2.4GHz 共享回传下有意义;**mmWave 专波束后带宽不再竞争、圈内速率恒充裕 → 无学习信号** | 删除;W_beam=25MHz 固定 |
| 2.4GHz log 回传 | 高功率全域闭合(半径>30km),中枢位置无关(9 轮扫描 ≤±1%) | 60GHz mmWave 链路预算 |
| 手设 γ_bh_th=40dB 截止 | 比接入门限还高 20dB,物理荒谬(hack) | O2 吸收自然限距 |
| 确定性/圆周 demand 漂移 | 时间均值≈固定点可预调,固定中枢≈最优 | 固定起点+随机方向+噪声+反弹 |
| 单一 λ_D 损失权重 | (其实没问题)当时为凹「覆盖份额 20-30%」而嫌它压不下覆盖份额 | 一度拆成 3:8(见下,已回退) |
| **λ_ovf > λ_src(3:8 拆分)** 凹份额 | 🔴**2e6 步长训练暴露反向激励**:策略故意少接纳避溢出(accepted 19.5M→5.0M、U_src 4.8M→16.9M),压制 demand-matching | **回退到等权 λ_src=λ_ovf=5**(丢一个 bit 就是丢一个 bit);份额改由策略涌现,不再凹 |
| K=8/16、σ=1000–1500、f_U=2GHz 等中间参数 | 各档体检份额/消融不达标(过程记录在会话) | §5 锁定表 |
| **纯团队成本下 demand-matching「涌现」(v2 K=24 过覆盖+过载)** | 🔴 **RL 训练证伪**:定位对成本仅~3.6% 敏感(<外生噪声~11%),策略漂移到劣于 hover(W₁ 835>random、成本 0.795>hover 0.709)；启发式体检看不出(它无脑覆盖、不优化成本) | **派生 v3 学习友好场景**(K=12+算力×2+队列×1.5→敏感度+26%)+ 探索修复(`--mec_logstd_init -1.9`/`--use_entropy_anneal`/ppo5/16环境);demand-matching 真正涌现(§10) |

> 历史复审结论仍有效的:R1(env 只回加权物理代价,PopArt 管尺度)、R2(ω 公式派生)、
> R5(服务位置显式可配,锁定 mid_move)、R6(β 用 Beta)、R7(物理方程主体正确)。R3/R4 被 v2 设计取代。


## 10. 训练实证与 v3 学习友好场景(2026-06-15)

> 把 v2 接入 MAPPO 真训练后得到的关键结论。**v2 物理/参数(§1–§7)作为锁定真源与数值对拍基准不变**;
> 本节记录"为何 v2 学不出 demand-matching"的诊断,以及为研究该决策而派生的 v3 场景。

### 10.1 v2 训练失败与"平地面"诊断

K=24、`v2_iort_6km_mmwave`、MAPPO(8 环境、ppo10、logstd 初始 0)训练 ~6.5e5 步:

- **算法机制健康**:`ratio≡1`、critic `explained_variance≈0.95`、`value_loss` 收敛 —— **不是代码 bug**。
- **但策略学成了比 hover 还差**:eval(确定性)团队成本 **0.795 > hover 0.709 > 启发式 0.683**;
  **W₁ 835m > random 736 > hover 603**;`dist_entropy` 趋势/噪声=18.7 坚决下降(在收敛,但收敛到坏策略);
  速度方向探针显示 minor 速度**错向**(需求在东→却朝西南)、且 σ≈0.8 满速乱窜把 UAV 从栅格初始打散。
- **定量根因 = 成本对定位几乎不敏感**:启发式(匹配)vs hover 的团队成本只差 **~3.6%**,而每-episode 外生随机
  (heading U[0,2π)+噪声)给回报带来 ~11% 噪声 → **可控信号 < 噪声**,梯度跟着噪声走。
- **为何不敏感(两头夹击)**:
  1. **几何过覆盖**:24 架 × 覆盖半径 861m ≈ 55.9 km² 覆盖 vs 场地 36 km²(1.55×);栅格间距≈半径,
     **不动也已罩住热点**(hover W₁ 603 < 热点 σ 800)→ 移动 UAV 不增覆盖。
  2. **重过载**:offered≈33 vs 容量 C_H+K·C_U=13.5(2.4×)→ 绑定约束是处理,不是覆盖;且队列成本 ω_Q
     与立方计算能耗"骑"在已接纳数据上,使"接纳→排队→溢出"比"源失效"更贵,**等权 λ 仍残留少接纳激励**。
- **推论**:在 v2 这种 regime 下,"demand-matching 从纯团队成本最小化中涌现"**不可能成立**——因为匹配不降成本。

### 10.2 v3 学习友好场景(`v3_iort_learnable.yaml`,由 v2 派生)

用启发式-vs-hover 成本差作"定位敏感度"指标做参数扫描(`scan_env.py`),锁定改动:

| 改动 | v2 | v3 | 作用 |
|---|---|---|---|
| `fleet_size_k` | 24 | **12** | 解过覆盖(覆盖≈0.8×),swarm 必须集中到热点 |
| `uav.cpu_frequency_hz` | 1 GHz (C_U=0.25) | **2 GHz (C_U=0.5)** | 解过载;"更少但更强"的现代 UAV |
| `hap.cpu_frequency_hz` | 30 GHz (C_H=7.5) | **60 GHz (C_H=15)** | 容量 13.5→21,过载 2.4×→**1.6×**(仍保留处理瓶颈) |
| `uav/hap.queue_max_bits` | 80 / 150 Mbit | **120 / 225 Mbit** | ×1.5 缓冲;ω_Q 自动按总容量归一化,几乎不抬惩罚 |
| 速度 V_U / V_H | 40 / 30 | **40 / 30(不变)** | 大平台更慢、UAV 更敏捷;速度不是瓶颈 |

效果(全网格、8 episode):heuristic-vs-hover 成本差 **+25.8%**(v2 仅 +1.8~3.6%)、W₁ 763<830<1155、
成本转为 **src(覆盖)主导** → 定位成为主要可控量。

### 10.3 训练修复与 v3 结果

探索修复(算法侧,见 §8.1):`--mec_logstd_init -1.9`(σ≈0.15)、`--use_entropy_anneal`、`--entropy_coef 0.003`、
`--ppo_epoch 5`、`--n_rollout_threads 16`;**未启用** per-UAV 局部奖励(env 修复后团队信号已够)。

> **log-prob 形状修复(2026-06-15)**:连续动作下 buffer 曾把 actor 的 joint log-prob `[.,1]` 广播成 act_dim 列,
> 使 PPO actor 损失被放大 act_dim=3×(上游 on-policy 通病,非 MEC 引入;Discrete 不受影响)。已在
> `shared_buffer.py`/`separated_buffer.py` 修正(joint 存宽度 1)。该 bug 经 Adam 大部分抵消,唯一实质效应是把
> 有效 entropy_coef 压到约 1/3;修复后显式用 `entropy_coef=0.003` 复现并略超原结果。下表为**修复后**数据。

v3 训练(K=12,1.5e6 步,CPU ~1h,seed 1,修复代码 + entropy 0.003):

| | 训练策略 | 启发式 | hover | random |
|---|---|---|---|---|
| 团队成本/slot | **0.509** | 0.500 | 0.674 | — |
| **W₁ (m)** | **733** | 763 | 830 | 1155 |
| 中枢消融(冻结@中心) | **+27.9%** | — | — | — |

W₁ 733 比 hover 低 12%(eval 脚本判定 GOAL MET)、比 random 低 36%;reward 全程上升(trend/noise 4.5)。
(注:训练期 mec/w1 是带探索噪声的末步快照,偏平;确定性 eval 的 W₁=733 才是权威值。)

**结论**:demand-matching 真正从团队成本最小化中涌现,**W₁ 低于手工启发式**,中枢轨迹强承重,accepted 不崩塌。
卖点(可动中枢 + demand-matching)在 v3 + 正确 PPO 实现上由学习策略实证成立。

### 10.4 待办
- 多种子复现(论文建议 3–5 个;当前 seed 1/2)。
- (可选)per-UAV 局部奖励(吞吐项,w 可调)进一步压低 W₁ / 改信用分配。
- (可选)在 v3 算力下扫 K=8/16/24 展示可扩展性,并实证"过覆盖"是 v2 病因。
- **排列不变性尚未贯穿集中 critic**(`share_obs` 有序拼接);Set Transformer 描述子升级(§8.1)是后续工作。
