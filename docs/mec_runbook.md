# MEC v6 换机器训练手册

本手册面向"把当前 `on-policy` 仓库搬到更强的机器上训练 `v6_continuous_workload`"。
当前本机只完成到训练前诊断阶段,下一步是在训练机器上跑 MAPPO 并用诊断指标确认参数。

> 两个反直觉开关:
>
> - `--cuda` 是 `store_false`: **想用 GPU 就不要传 `--cuda`**;传了反而强制 CPU。
> - `--use_wandb` 也是 `store_false`: **想用本地 TensorBoard 就传 `--use_wandb`**;不传会走 wandb。

---

## 1. 环境搭建

```bash
cd ~/Projects/on-policy

conda create -n marl python=3.10 -y
conda activate marl

# 先安装匹配训练机器 CUDA/驱动的 PyTorch,示例:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH

python -c "import torch; print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

如果已经有 `marl` 环境,只需:

```bash
conda activate marl
cd ~/Projects/on-policy
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

`requirements.txt` 已 pin `tensorboard==2.20.0`。

---

## 2. 代码自检

先跑轻量测试,确认 v6 配置解析、环境 step、MEC policy 都正常:

```bash
PYTHONPATH=$PWD python -m pytest \
  onpolicy/envs/mec/tests/test_config_loader.py \
  onpolicy/envs/mec/tests/test_finite_k_env.py \
  onpolicy/algorithms/mec/tests/test_mec_policy.py \
  -q
```

本机最近结果:

```text
17 passed, 1 skipped, 1 warning
```

再跑编译检查:

```bash
PYTHONPATH=$PWD python -m compileall -q \
  onpolicy/envs/mec onpolicy/scripts/analysis/design_v6_sanity.py
```

---

## 3. 训练前 probe

训练前先跑真实积分诊断,不要用固定 `eta_ref` 判断参数:

```bash
PYTHONPATH=$PWD python -m onpolicy.scripts.analysis.design_v6_sanity
```

当前期望结果的关键形态:

```text
K=16  A_tot=150.0 Mbit/slot  zeta=0.70
W_ac_i=2.500 MHz
W_bh_i=25.000 MHz
offered_compute_ratio=0.97
best n_hot counts over 32 centers: {5: 14, 6: 10, 7: 8}
```

重点检查:

- `n_hot=5-7` 应该比 `n_hot=16` 接纳更多 workload。
- `source_outside` 和 `source_capacity` 都要看,不能只看总 source loss。
- `eta_p05/p50/p95` 是真实信道积分得到的分位数;`eta_mean` 不是全空间无条件平均。
- hub 偏移 3km 时应出现更高 backhaul utilization 和少量 overflow,说明回传几何承重。

如果 probe 结果大幅偏离,先不要训练,优先检查依赖、代码版本和 YAML 是否同步。

---

## 4. 启动训练

建议用 `tmux` 或 `screen`:

```bash
tmux new -s mec-v6
conda activate marl
cd ~/Projects/on-policy
export PYTHONPATH=$PWD:$PYTHONPATH
```

训练 seed 1:

```bash
python -u -m onpolicy.scripts.train.train_mec \
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
  --use_wandb \
  2>&1 | tee train_v6_seed1.log
```

注意:上面没有 `--cuda`,因此默认使用 GPU。如果只想在 CPU 上 smoke test,才传 `--cuda`。

如果资源允许,继续跑:

```bash
# 只改 experiment_name 和 seed
--experiment_name v6_continuous_seed2 --seed 2
--experiment_name v6_continuous_seed3 --seed 3
```

结果默认在:

```text
onpolicy/scripts/results/MEC/v6_continuous_workload/mappo/<experiment>/runN/
```

---

## 5. TensorBoard

```bash
tensorboard --logdir onpolicy/scripts/results/MEC/v6_continuous_workload/mappo --port 6006
```

远程机器推荐 SSH 端口转发:

```bash
ssh -N -L 6006:localhost:6006 user@gpubox
```

本地浏览器打开 `http://localhost:6006`。

---

## 6. 训练中重点看什么

不要只看 `average_episode_rewards` 和 `mec/w1`。v6 的核心验收是系统闭环是否合理。

| 指标 | 期望 |
|---|---|
| `mec/n_hotspot_uav` | 多数 episode 末态约 5-7 |
| `mec/n_background_uav` | 不应接近 0,理想约 9-11 |
| `mec/accepted` | 稳定或上升,不能靠少接纳偷懒 |
| `mec/U_src` | 不长期爆炸 |
| `mec/source_outside` | 用于判断 outside option 是否吃掉太多背景 |
| `mec/source_capacity` | 用于判断接入容量是否成为瓶颈 |
| `mec/hotspot_accepted` | 热点 workload 应被充分接纳 |
| `mec/background_accepted` | 背景 workload 也应有实质接纳 |
| `mec/overflow` | 不应长期爆炸 |
| `mec/access_utilization` | 不应长期过低或长期贴满 |
| `mec/backhaul_utilization` | hub 几何变化时应有响应 |
| `mec/uav_compute_utilization`, `mec/hub_compute_utilization` | 默认场景应有明显计算压力 |
| `mec/eta_p05/p50/p95` | 检查真实频谱效率分布,防止均值误导 |
| `mec/hub_to_hotspot` | hub 应学习到有意义的几何位置 |

如果训练后 `n_hotspot_uav` 又接近 16,先不要立即调物理参数。需要同时检查 reward 权重、
探索、策略表达、动态训练轨迹和背景 accepted workload。

---

## 7. 评测

训练完成后先用 policy 评测:

```bash
RUN=onpolicy/scripts/results/MEC/v6_continuous_workload/mappo/v6_continuous_seed1/run1

PYTHONPATH=$PWD python -m onpolicy.scripts.eval.eval_mec \
  --env_name MEC \
  --mec_eval_controller policy \
  --model_dir $RUN/models \
  --mec_eval_episodes 12
```

再跑基线:

```bash
PYTHONPATH=$PWD python -m onpolicy.scripts.eval.eval_mec \
  --env_name MEC --mec_scenario v6_continuous_workload \
  --mec_eval_controller heuristic --mec_eval_episodes 12

PYTHONPATH=$PWD python -m onpolicy.scripts.eval.eval_mec \
  --env_name MEC --mec_scenario v6_continuous_workload \
  --mec_eval_controller hover --mec_eval_episodes 12

PYTHONPATH=$PWD python -m onpolicy.scripts.eval.eval_mec \
  --env_name MEC --mec_scenario v6_continuous_workload \
  --mec_eval_controller random --mec_eval_episodes 12
```

评测时同样不要只看 W1。要把 accepted/source/overflow、hotspot/background split 和
hub geometry 一起看。

---

## 8. 当前 v6 解释口径

论文或对外说明建议这样表述:

- `gamma_ref=8dB` 是 service-attractiveness normalization reference,不是 hard access threshold。
- `p0=1e-8 W/Hz` 是 low-power IoRT effective PSD。
- `W_ac_total=40MHz` 是 conservative sub-6 access bandwidth pool。
- `G_eff=15dB` 是 effective backhaul gain after beamforming gains and fixed implementation losses。
- 默认场景是 access-constrained / compute-aware / backhaul-geometry-sensitive,但当前 source loss
  主要可能来自 service attractiveness/outside option,不是纯带宽饱和。

