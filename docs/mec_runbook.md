# MEC 换机器训练手册

本手册用于在新机器上确认 `v6_hap_loadbearing` 参数。当前阶段只建议先跑
300k-500k steps 的 3-seed 诊断，不直接启动论文最终实验。

当前状态和实验依据见 [`../HANDOFF.md`](../HANDOFF.md)。

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

## 5. 3-seed 短程训练

建议从 300k steps 开始。若三组仍在稳定改善，可延长至 500k。

```bash
SCENARIO=v6_hap_loadbearing
STEPS=300000
SEED=1
EXP=v6_hap_lb_probe_seed${SEED}

PYTHONPATH=$PWD python -u -m onpolicy.scripts.train.train_mec \
  --env_name MEC --algorithm_name mappo \
  --experiment_name $EXP \
  --mec_scenario $SCENARIO \
  --seed $SEED \
  --n_rollout_threads 16 --n_training_threads 6 \
  --episode_length 200 --num_env_steps $STEPS \
  --ppo_epoch 5 --num_mini_batch 1 \
  --hidden_size 128 --layer_N 2 \
  --use_entropy_anneal --mec_logstd_init -1.9 --entropy_coef 0.003 \
  --lr 5e-4 --critic_lr 5e-4 --gamma 0.99 \
  --log_interval 5 --save_interval 25 \
  --use_wandb \
  2>&1 | tee ${EXP}.log
```

依次运行 seed 1/2/3。机器资源允许时可以并行，但先确认单个 run 的显存和 CPU 占用。

结果目录：

```text
onpolicy/scripts/results/MEC/v6_hap_loadbearing/mappo/<experiment>/runN/
```

注意：当前 `save_interval` 会更新 latest `actor.pt/critic.pt`，不会保留所有历史 step。
短程参数确认可接受；正式长训练前应实现 step checkpoint 和 validation checkpoint selection。

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

若 HAP 仍只在个别 seed 承重，不继续堆训练步数。下一步应转向：

- major/minor 独立 encoder 或 optimizer；
- SetRec population descriptor；
- 置换不变 centralized critic；
- role-wise PPO loss normalization。

这些修改不改变论文的联合动作变量。
