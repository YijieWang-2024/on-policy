# MEC 训练与结构实验手册

本手册用于 `v6_hap_loadbearing` 的结构实验。当前阶段不直接启动论文
最终实验，先比较 mean、flat 和 set 三类输入架构。

当前状态见 [`../HANDOFF.md`](../HANDOFF.md)，网络与训练契约见
[`setrec_architecture.md`](setrec_architecture.md)。

## 1. 环境安装

推荐 Python 3.10。PyTorch 应根据训练机器的 CUDA 和驱动单独安装。

```bash
cd ~/Projects/YijieWang-2024-on-policy

conda env create -f environment.yaml
conda activate marl

# environment.yaml 会读取仓库内已经验证过的 requirements.txt。
# 若安装到 CPU wheel 或 CUDA 不匹配，再用 pytorch.org 针对本机生成的命令
# 重新安装 requirements.txt 中相同版本的 PyTorch。

export PYTHONPATH=$PWD:$PYTHONPATH

python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

若 `marl` 环境已经存在，使用：

```bash
conda env update -n marl -f environment.yaml --prune
conda activate marl
```

本仓库有两个反向开关：

- 想使用 GPU：不要传 `--cuda`。
- 想关闭 wandb、写本地 TensorBoard：传 `--use_wandb`。

## 2. 代码自检

```bash
PYTHONPATH=$PWD python -m pytest \
  tests \
  onpolicy/envs/mec/tests \
  onpolicy/algorithms/mec/tests \
  -q

PYTHONPATH=$PWD python -m compileall -q \
  onpolicy/envs/mec \
  onpolicy/algorithms/mec \
  onpolicy/scripts/analysis \
  onpolicy/scripts/eval
```

若 parity test 因缺少原始 `Mean Field Mec` 仓库而 skip，属于预期行为。

## 3. 场景确认

当前短程训练候选：

```text
onpolicy/envs/mec/scenarios/v6_hap_loadbearing.yaml
```

它与旧 `v6_continuous_workload` 的唯一区域性差别是 28 GHz 回传链路预算：

```text
W_bh_total = 400 MHz
P_tx = 23 dBm
combined antenna gain = 20 dB
link margin = 7 dB
```

接入、workload、计算、队列和代价保持不变。

先跑 probe：

```bash
PYTHONPATH=$PWD python -m onpolicy.scripts.analysis.design_v6_sanity \
  --scenario v6_hap_loadbearing
```

应确认：

- `K=16`，`A_tot=150 Mbit/slot`。
- `W_ac_i=2.5 MHz`，`W_bh_i=25 MHz`。
- offered compute ratio 约 0.97。
- 最优热点 UAV 数仍主要落在 5-7。
- 近距离回传充足，2-3 km 回传明显下降，但没有 hard cutoff。

## 4. 最小 smoke

先确认训练入口、日志和模型保存路径：

```bash
PYTHONPATH=$PWD python -u -m onpolicy.scripts.train.train_mec \
  --env_name MEC --algorithm_name mappo \
  --experiment_name v6_hap_lb_smoke \
  --mec_scenario v6_hap_loadbearing \
  --mec_policy_arch set \
  --seed 1 \
  --n_rollout_threads 2 --n_training_threads 2 \
  --episode_length 200 --num_env_steps 6400 \
  --ppo_epoch 2 --num_mini_batch 1 \
  --hidden_size 128 --layer_N 2 \
  --use_entropy_anneal --mec_logstd_init -1.9 --entropy_coef 0.003 \
  --lr 5e-4 --critic_lr 5e-4 --gamma 0.99 \
  --log_interval 1 --save_interval 1 \
  --use_wandb
```

架构开关：

```text
--mec_policy_arch legacy_mean  # 仅用于读取旧 14 维 mean checkpoints
--mec_policy_arch mean         # 对齐基线：HAP p+mean，UAV s_i+p+mean，critic p+mean
--mec_policy_arch flat         # 对齐基线：有序完整 UAV 集合，固定 K、标签敏感
--mec_policy_arch set          # 对齐方法：公共不变 population encoder
```

新的 `mean/flat/set` 都使用完整 team group 和同一套 role-specific
actor/team critic；只替换 population representation。环境局部观测为
`[role, own(3), p(7)]`，runner 生成 centralized state
`[p(7), s_1(3), ..., s_K(3)]`。旧模型若 config 中没有
`mec_policy_arch`，加载脚本会自动选择 `legacy_mean`。

## 5. 3-seed 结构对照训练

旧 mean-descriptor MAPPO 已完成，但不属于严格对齐的表示消融。新的
`mean/flat/set` 应使用相同预算、role-wise loss、step checkpoint 和固定验证集：

```bash
SCENARIO=v6_hap_loadbearing
STEPS=512000
SEED=1
ARCH=set
EXP=v6_hap_lb_${ARCH}_seed${SEED}

PYTHONPATH=$PWD python -u -m onpolicy.scripts.train.train_mec \
  --env_name MEC --algorithm_name mappo \
  --experiment_name $EXP \
  --mec_scenario $SCENARIO \
  --mec_policy_arch $ARCH \
  --seed $SEED \
  --n_rollout_threads 16 --n_training_threads 2 \
  --n_eval_rollout_threads 8 \
  --episode_length 200 --num_env_steps $STEPS \
  --ppo_epoch 5 --num_mini_batch 1 \
  --hidden_size 128 --layer_N 2 \
  --use_entropy_anneal --mec_logstd_init -1.9 --entropy_coef 0.003 \
  --lr 5e-4 --critic_lr 5e-4 --gamma 0.99 \
  --mec_rolewise_loss \
  --log_interval 5 --save_interval 20 --save_step_checkpoints \
  --use_eval --eval_interval 20 --eval_episodes 24 --eval_seed 1000 \
  --use_wandb \
  2>&1 | tee ${EXP}.log
```

依次运行 seed 1/2/3。机器资源允许时可以并行，但先确认单个 run 的显存和 CPU 占用。

结果目录：

```text
onpolicy/scripts/results/MEC/v6_hap_loadbearing/mappo/<experiment>/runN/
```

保存内容：

```text
models/actor.pt, critic.pt, trainer_state.pt        # latest
models/checkpoints/step_<env_steps>/                # numbered
models/best/                                        # fixed-validation best
```

## 6. TensorBoard 与实时检查

```bash
tensorboard \
  --logdir onpolicy/scripts/results/MEC/v6_hap_loadbearing/mappo \
  --port 6006
```

远程训练可使用：

```bash
ssh -N -L 6006:localhost:6006 user@gpu-host
```

训练中至少查看：

| 指标 | 判断 |
|---|---|
| `average_episode_rewards` | 总趋势改善，不要求单调 |
| `mec/accepted` | 不应靠少接纳换低成本 |
| `mec/U_src` | 不应持续恶化 |
| `mec/overflow` | 不应长期爆炸 |
| `mec/n_hotspot_uav` | 多数阶段约 5-7 |
| `mec/n_background_uav` | 不应接近 0 |
| `mec/source_outside` | 判断背景 coverage |
| `mec/source_capacity` | 判断接入容量 |
| `mec/backhaul_utilization` | 应随 HAP/UAV 几何变化 |
| `mec/uav_compute_utilization` | 检查本地计算压力 |
| `mec/hub_compute_utilization` | 目标约 0.7-0.8，不应长期贴满 |
| `mec/hub_to_hotspot` | 仅作几何辅助，不单独判优 |
| `mec/w1` | 必须与 cost/accepted/queue 联合解释 |

## 7. 统一评估

三个 seed 必须使用相同测试 seed 和 episode 数。

```bash
RUN=onpolicy/scripts/results/MEC/v6_hap_loadbearing/mappo/v6_hap_lb_probe_seed1/run1

PYTHONPATH=$PWD python -m onpolicy.scripts.eval.eval_mec \
  --env_name MEC \
  --mec_eval_controller policy \
  --model_dir $RUN/models \
  --seed 1000 \
  --mec_eval_episodes 24
```

启发式基线：

```bash
PYTHONPATH=$PWD python -m onpolicy.scripts.eval.eval_mec \
  --env_name MEC \
  --mec_scenario v6_hap_loadbearing \
  --mec_eval_controller heuristic \
  --seed 1000 \
  --mec_eval_episodes 24
```

policy 评估会同时输出冻结 HAP 消融。不要用单个 best episode 替代均值评估。

## 8. 短程通过标准

进入长训练前，3 个 seed 应同时满足：

- reward/cost 已出现明确改善趋势。
- demand matching 优于 hover/random。
- accepted 不发生系统性下降。
- queue cost 相比旧 v6 有改善迹象。
- learned HAP freeze ablation 多数 seed 大于 10%。
- beta、UAV 轨迹和 HAP 轨迹不是只在单一 seed 承重。
- 无持续 overflow。

若 Set-MAPPO 仍只在个别 seed 承重，不继续堆训练步数。先检查 permutation
测试、descriptor 使用情况、critic 拟合和 role-wise KL，再决定是否进入
reconstruction phase。decoder 与 auxiliary phase 的命令将在 phase 2
实现后补充。
