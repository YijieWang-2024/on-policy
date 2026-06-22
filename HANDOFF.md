# MEC 项目交接文档(2026-06,会话续接用)

> 用途:把当前上下文移植到新智能体继续。读完这份 + `docs/mec_env_port_spec.md` §10 即可接手。
> 项目根:`F:\置换不变性\YijieWang-2024-on-policy`;分析脚本在 `F:\置换不变性\`。
> **最新主线 = v5(静态随机热点 + 每 episode 随机初始化 + 正交接入)。** 详见 §2、§5。

---

## 0. 环境与铁律(踩过的坑,务必遵守)

- **marl 环境别用 `conda run -n marl`**——会触发交互提示卡死。直接用解释器:
  `C:/Users/Administrator/anaconda3/envs/marl/python.exe`
- torch 2.12 **CPU 版**(16 核, numpy 2.2.6)。`--cuda` 和 `--use_wandb` 都是 **store_false**(传了反而关闭);CPU 训练无所谓 cuda,wandb 传了就走本地 TensorBoard。
- **长训练命令会被自动转后台**:用 `run_in_background` + 日志写到 `F:\置换不变性\xxx.log`(工作目录),**不要**依赖 temp 里的 UUID 路径(容易写错 UUID 读不到)。
- **YAML 注释里别留裸冒号**:`rationale: ... away): far-field ...` 里的 `): ` 会被 YAML 当 mapping 分隔符报错。用 `--` 代替。
- demand 网格 `grid_shape`(v3/v4/v5 是 81×81=6561),reshape 用 `int(sqrt(size))` 别写死。
- **W₁ 会骗人**:它是"质量加权到最近 UAV 距离",背景铺开的 UAV 会把它拉低,**掩盖热点服务不足/全员挤热点**。判断好坏必须同时看:① 末态分布统计(几架在热点/几架背景)② 可视化 PNG ③ hub 消融 ④ ovf。**别只信 W₁——这是本项目最大的教训。**
- **改 env 物理/初始化后**务必先 `pytest onpolicy/envs/mec/tests -q`(应 12 passed, 1 skipped)再训练。

---

## 1. 项目主线

finite-K 分层空中 MEC:K 架同构 UAV(minor,共享策略)+ 1 个可移动算力中枢 hub(major,单独头)。MAPPO fork,major-minor 共享参数,排列不变性研究主题。
**目标**:让 **demand-matching(UAV 群按需求密度分布)成为"团队成本最小化"的涌现副产品**——不直接优化 W₁。想看到的画面:**两层分布 = ~5 架服务热点 + 其余铺背景 + hub 坐镇热点中心**,且 UAV/hub 位置、卸载 β 三类动作都"承重"(消融显著)。
**方法路线**:静态 demand 的 inverse design——先定想要的最优分布,再调参数让它成为唯一最优,最后训练验证 RL 能否涌现。

---

## 2. 场景文件现状(演进链)

| 文件 | 状态 | 说明 |
|---|---|---|
| `v2_iort_6km_mmwave.yaml` | **锁定,别动** | 原始真源 + 数值对拍基准 |
| `v3_iort_learnable.yaml` | 移动热点版(已过气) | K=12,运动重定标。能学但 hub 头在移动目标上学失败 |
| `v4_static_demand.yaml` | 静态固定起点版(已被 v5 取代) | K=16 静态随机热点,但 UAV 固定 0.85R 角起点 → 学出**纵带偏移**(见 §5.5) |
| `v4_fixed_demand.yaml` | 失败消融 | 热点+起点全固定 → **探索死锁学不动**(见 §5.6) |
| `v5_static_randinit.yaml` | **当前主线** | v4 配方 + **每 episode 随机初始化**(hub/群心/热点都随机)。消除了纵带偏移,但暴露**全员挤热点**问题(见 §5.7) |

### 配方 [C](phase1 探针验证 n*=5,v4/v5 共用)
K=16;σ=600;peak=0.008;bg=1e-5;**C_U=0.35(f_U=1.4GHz)**;**C_H=10.5(f_H=42GHz)**;队列沿用 v3(×1.5)。
**接入 = 正交带宽 W/K(每架 total/K)**,但 v4/v5 把 `total_bandwidth_hz` 设为 320MHz(=80×4)当"第一阶段脚手架"放宽接入,把瓶颈挪到处理。
phase1 验证(接入充裕下):热点最优 ~5 架(成本谷底 n=3~5,第6架回升),单架溢出→3架清零,cap≈16 够用 ovf=0。
**⚠️ 注意**:n*=5 的谷底很浅(n=3~5 cost 都≈0.12),接入越充裕越偏向"堆热点"——这是 v5 全员挤热点的根因(§5.7、§6)。

### v5 初始化设计(当前,用户拍板)
每 episode 在 `finite_k_env.reset` 内随机(用 env.rng,可复现):
- 热点中心 ~ U([0.3,0.7]²)(中心区,不贴边、不被截)
- hub = 群质心,也 ~ U([0.3,0.7]²)
- 16 架 UAV 在群心周围 1km×1km 的 4×4 网格(间距 333m ≫ d_min=30m,零碰撞)
配置在 v5 yaml 的 `env.uav.initial_deploy`(`random_centroid: true` / `centroid_frac_range` / `grid_side_m`)。

---

## 3. 训练 / 评测命令模板

```bash
C:/Users/Administrator/anaconda3/envs/marl/python.exe -u -m onpolicy.scripts.train.train_mec \
  --env_name MEC --algorithm_name mappo --experiment_name <名> \
  --mec_scenario v5_static_randinit --seed 1 \
  --n_rollout_threads 16 --n_training_threads 6 --episode_length 200 \
  --num_env_steps 1500000 --ppo_epoch 5 --num_mini_batch 1 \
  --hidden_size 128 --layer_N 2 \
  --use_entropy_anneal --mec_logstd_init -1.9 --entropy_coef 0.003 \
  --lr 5e-4 --critic_lr 5e-4 --gamma 0.99 \
  --log_interval 5 --save_interval 25 --use_wandb --cuda \
  > "F:/置换不变性/train_xxx.log" 2>&1
```
K=16 约 250 FPS,1.5e6 步 ≈ 1.7h。结果在 `onpolicy/scripts/results/MEC/<scenario>/mappo/<exp>/run1/{models,logs}`。

```bash
# 评测(controller: policy/heuristic/hover/random)
... -m onpolicy.scripts.eval.eval_mec --env_name MEC --mec_scenario v5_static_randinit \
  --mec_eval_controller policy --model_dir <run1/models 绝对路径> --mec_eval_episodes 12
```

---

## 4. 已做的代码改动(可能未全 git;用户在别处提交过)

- **log-prob 3× bug 修复**:`shared_buffer.py`/`separated_buffer.py`——连续动作 joint log-prob 曾被广播成 act_dim 列,actor loss 放大 3×。已修(joint 存宽度 1)。结论:Adam 大部分抵消,唯一实质效应是有效 entropy 压到 1/3;修复后用 `entropy_coef=0.003` 复现。
- `config.py`:新增 `--mec_logstd_init`(默认 -1.9)、`--use_entropy_anneal`、`--entropy_coef_min`。
- `mec_policy.py`:major/minor logstd 从 `args.mec_logstd_init` 初始化。
- `mpe_runner.py`:`run()` 循环加 entropy 线性退火(flag 守卫)。
- `mec_runner.py`:`[mec]` 控制台摘要行 + `w1` 日志键。
- `MEC_env.py`:逐步 W₁ 计算(`_agent_infos`)。
- `finite_k_env.py`:
  - `_initial_demand_motion` 支持 `initial_center_frac_range`(每 episode 随机热点中心)。
  - **`_initial_deployment`**(reset 内):支持 `env.uav.initial_deploy.random_centroid` → 每 episode 随机 hub/群心 + 1km 网格部署 UAV。无此配置时回退固定值(向后兼容)。
  - `_access_rate` = **正交接入**(每架 `bandwidth_per_uav_hz`=W/K,裸 SNR)。**这是当前确定的接入模型**;接入半径由 γ_th(SNR 门限)定、与带宽无关;回传 60GHz 链路预算未动。
- `config_loader.py`:UAV `cluster` generator + hub `swarm_centroid`(=群质心)。

测试:`... -m pytest onpolicy/envs/mec/tests onpolicy/algorithms/mec/tests -q` → 应 26 passed, 1 skipped。

---

## 5. 关键发现与经验(按时间线,**这是交接的核心**)

1. **v2 "平地面"失败**:K=24 过覆盖 → 定位对成本只 ~3.6% 敏感 < 外生噪声 ~11% → demand-matching 学不出来。**教训:定位必须"承重"(heuristic vs hover 成本差要够大)才可能学出。**
2. **v3 移动热点**:成本跨种子稳健,但 W₁ 只有 best-seed 胜 hover。诊断 `corr(W₁,cost)=+0.71`(**目标无错位**),不稳是部分种子收敛到"少动"局部最优。
3. **major(hub)头在移动目标上学失败**:中枢平均落后群质心 1528m,对"偏离群质心"响应方向混乱。根因:major 占展平 batch 仅 1/K,梯度被 K 个 minor 淹没。**静态目标能治好它**(v4 hub 消融 +71%)。
4. **追踪动力学太难** → pivot 到**静态 demand**:删掉"追移动目标"(难),保留"demand→部署映射"(有价值)。每 episode 随机热点位置,保证策略学的是映射而非记坐标。
5. **v4(静态随机热点,固定 0.85R 角起点)**:
   - 学习曲线迄今最强(W₁ 2220→505,ovf 全程 0,eval 胜最强基线 49% "GOAL MET",hub 消融 +71%)。
   - **但动图揭穿标量**:群+中枢**系统性偏向热点一侧**(群成偏左下竖直带、hub 停热点右侧没坐中心)。分布统计:热点内仅 ~2.9 架、方差极大(0~8),不是稳定两层。**W₁ 好看是被背景 11 架拉低掩盖了**(再次印证 §0 铁律)。
   - 根因:**固定起点 → 策略学成"往左下走固定向量"的方向惯性**,而非相对热点转向。
6. **v4 固定热点消融(热点+起点全固定)= 学不动**(accept≈0,群没出发)。**诊断:稀疏奖励 + 弱探索(σ=0.15)死锁**——群要走 ~3.8km 才有第一个正信号,之前梯度全 0。**反直觉:随机化反而帮助学习**(热点偶尔出现在近处→拿到信号→共享策略泛化,课程式探索引导)。用户已接受,不再验证。
7. **v5(每 episode 随机 hub/群心/热点)= 最新结果**:
   - ✅ **纵带偏移彻底消失**:群以热点为中心对称聚拢,hub 基本坐热点旁(hub 到热点 1185m→555m)。**证实固定起点是 v4 偏差的根因,用户的随机化方案成功。**
   - ✅ 训练更好:W₁ 末段 360m、U_src 仅 1.27(96% 接纳)、accept 17.2。
   - ❌ **新问题:全员挤热点**——16 架几乎全聚热点(<1.5σ 13.8 架、背景仅 0.4 架),两层塌成单层。
   - **根因**:配方[C] 的 n*=5 是在"接入×4 充裕"下算的,谷底浅;接入充裕时堆热点的边际接入速率不衰减 → 最优滑向"全堆热点"。

---

## 6. 下一步(开放问题 + 候选方向)

**当前卡点**:v5 已解决空间偏差(画面对称),但**全员挤热点、背景被放弃**。需要让"两层分布"成为真正的物理最优。

**注意约束**(用户明确):**接入保持正交带宽(各 UAV 之间不复用、不引干扰)**。不要再走"复用+SINR 干扰"那条路(试过,干扰过大无法工作,已撤回正交)。

**候选方向**(未定,需与用户确认):
- **A. 把接入×4 脚手架调回物理值**(320→80MHz):正交接入下,热点拥挤区每架分到的 W/K 带宽被 n_srv 摊薄(+B_max 截顶),堆第 6/7 架到热点的边际接入速率下降 → 背景独占带宽划算 → 两层成为最优。最直接。先用 `phase1_design.py` 在 80MHz 下重算 n*,确认谷底够深(背景边际 > 热点第6架边际)再训。
- **B. 降单架处理/带宽容量**:让热点"一架服务不过来"更刚性,逼出分摊。
- **C. 调热点/背景质量比**:背景总量调大,让"去背景"的绝对收益更高。
- 验收(任何方向):重训后必看 **`v5_viz.py` 分布图** + `v5_dist_check.py` 逐 episode 计数(目标:热点 ~5、背景 ~11、hub 到热点≈0、**跨 episode 一致**),不能只看 W₁。

---

## 7. 分析脚本清单(都在 `F:\置换不变性\`)

| 脚本 | 用途 |
|---|---|
| `trend_analysis.py <logdir>` | 每指标 OLS 趋势 + trend/noise(判断是否真在学) |
| `tb_dump.py` | 解析 TensorBoard 日志(被上面 import) |
| `compare_curves.py` | 多 run 训练曲线头对头 |
| `diagnose_w1.py` | corr(W₁, 各成本分量),判断目标是否错位 |
| `phase1_design.py` | **静态 n* 设计探针**(验证配方[C];改接入带宽重算 n* 用这个) |
| `static_design.py` | deploy-and-hold 分布打分 |
| `scan_env / noise_scan / motion_scan / gap_scan / sigma_scan / sigma_peak_scan` | 各维度环境敏感度扫描 |
| `major_diag.py` | major 头失败三假设诊断 |
| `v4_dist_check.py` / `v5_dist_check.py` | 末态 UAV 热点/背景计数(改 MD 路径+场景名复用) |
| `v4_viz.py` / `v5_viz.py` | 末态分布渲染 PNG(关键验收工具) |
| `make_v3.py / make_v4.py` | 从上一版派生场景 yaml(派生 v5 见 git 历史 / 直接改 yaml) |

**注意**:`v4_dist_check.py`/`v4_viz.py` 里硬编码了 model_dir 路径和 `mec_scenario`,复用时改这两处(v5 版已生成好)。

---

## 8. 文档位置

- `docs/mec_env_port_spec.md` §10 / §10.3b:v3 诊断 + 场景重定标设计。**v4/v5 的内容尚未写入 spec**,待补。
- `docs/diagnosis_layout_not_loadbearing.md`:布局非承重诊断(本会话产出)。
- `WORK_NOTES.md`:历史变更记录。
- 用户 auto-memory:`MEMORY.md` → `mec-project-overview.md`、`mec-overload-equal-lambda-misalignment.md`。
- **本文件 `F:\置换不变性\HANDOFF.md` 是最新、最全的交接入口。**
