# BTWM 完整实验 Prompt v2

将此 prompt 交给 GPU 服务器上的 Codex，实现 BTWM 论文的全部实验。

> **实现状态（2026-06-13）：** 当前代码只实现了 DreamerV3 baseline 和
> “DreamerV3 + inverse-action loss”的 Full BTWM。Policy masking、独立
> inverse trunk、return relabeling 和 random-label control 仍是实验规划，
> 尚无可运行实现，不得据此声称相关 ablation 已完成。所有新实验统一通过
> `code/run_experiment.sh` 启动；`results_archive/` 中的旧结果来自修复前
> 实现，不可用于论文比较。

---

## 2026-06-13 实现与运行记录

### 已完成的正确性修复

- inverse head 的离散动作类别数改为读取 `space.classes`，不再错误使用
  one-hot 张量末维或默认值。
- 连续动作按每个 action dimension 分箱，使用离散交叉熵训练；类别数由
  `agent.inv_num_bins` 控制。旧实现对连续动作返回零损失，因而旧 DMC
  BTWM 结果无效。
- inverse transition 对齐为 `(z_t, z_{t+1}) -> a_t`，训练目标使用
  `prevact[:, 1:]`，不再错位预测上一转移的动作。
- 使用 `reset[:, 1:]` 屏蔽跨 episode transition，并按有效 transition
  数量归一化，避免 padding/reset 样本污染 loss。
- 新增原始 inverse CE、accuracy、valid fraction 指标：
  `train/inv/ce`、`train/acc/inv`、`train/inv/valid_frac`。其中
  `train/loss/inv` 是乘过 loss scale 后的优化项。
- `code/test_btwm_head.py` 的 7 个单元测试全部通过；混合离散/连续动作的
  CPU debug smoke test 通过。

### 统一运行约定

- 正式实验只通过 `code/run_experiment.sh` 启动，输出必须位于本项目的
  `runs/` 和 `logs/`，不得写到根目录或 `/home/ligq/logdir`。
- 每个模型、domain、task、seed 使用独立 run directory；修复前 checkpoint
  不允许 resume。
- 当前共享 A100 上采用 `--batch_size 8 --batch_length 32
  --report_length 32`，启动前至少要求 30 GB 空闲显存。
- 长时间任务放入 tmux；`runs/pids/*.pid` 和 `runs/pids/*.tmux` 分别记录
  Python PID 与会话名。

### 当前 Atari 验证运行

| 模型 | 任务 | GPU | PID | tmux | 记录时 step | run directory |
|------|------|:---:|----:|------|------------:|---------------|
| DreamerV3 | Atari100k Pong seed 0 | 1 | 1365602 | `btwm_dv3_pong_s0` | 18120 | `runs/dreamerv3_atari_pong_seed0` |
| BTWM | Atari100k Pong seed 0 | 4 | 1365596 | `btwm_full_pong_s0` | 28520 | `runs/btwm_atari_pong_seed0` |

两者目标均为 100,000 环境步。BTWM 已产生非零 inverse 指标；记录时
`train/inv/ce=2.38e-4`、`train/acc/inv=1.0`、
`train/inv/valid_frac=0.9988`。这只用于确认实现正在训练，不能作为最终
性能结论。

### 当前 DMC 验证运行

2026-06-13 12:02:35 启动 `walker_walk seed 0` 对照：

| 模型 | GPU | PID | tmux | 记录时 step | run directory |
|------|:---:|----:|------|------------:|---------------|
| DreamerV3 | 6 | 1402254 | `btwm_dv3_dmc_walker_s0` | 6752 | `runs/dreamerv3_dmc_walker_walk_seed0` |
| BTWM | 7 | 1402249 | `btwm_full_dmc_walker_s0` | 5648 | `runs/btwm_dmc_walker_walk_seed0` |

两者目标均为 100,000 环境步，配置为 `dmc_vision`、replay 300,000、
batch size 8、batch length 32、report length 32。首轮 JIT 后两个
`metrics.jsonl` 均持续增长，GPU 6/7 各占用约 36.7 GB。BTWM 记录时
`train/inv/ce=0.0332`、`train/acc/inv=0.9942`、
`train/inv/valid_frac=1.0`、`train/loss/inv=0.00332`，确认连续动作分箱
分支在实际训练而非返回零损失。早期 accuracy 受初始数据分布影响，不能
作为最终可预测性结论。

服务器其他 GPU 虽有约 34 GB 空闲显存，但每张均已有约 6 GB 的既有进程；
当前先只新增这一组 DMC 对照。`/data2` 记录时剩余约 749 GB，扩展任务前
必须继续检查磁盘和共享 GPU 占用。

### BTWM v2 调整与 1M 公平对照

100k seed 0 结果中，旧 BTWM 在 Pong 和 Walker 均未优于 baseline。审计
发现旧 inverse head 有两个主要风险：

- head 直接读取包含上一动作输入的 RSSM deterministic state，inverse
  accuracy 很快接近 100%，可能主要利用动作泄漏而不是学习视觉动力学。
- BTWM 比 baseline 多约 11M 参数（DMC 为 176.98M vs 165.95M），不满足
  参数量公平对照。

2026-06-13 实施 v2：

- inverse 输入改为相邻 encoder tokens，去除直接受动作驱动的 deterministic
  state；辅助梯度仍会更新共享视觉 encoder。
- head 缩小为 2 层、hidden 256；连续动作分箱从 5 增加到 21。
- inverse CE 除以随机猜测熵 `log(num_classes)`，使未训练 objective 约为
  1；loss weight 从 0.1 降为 0.05。
- 参数匹配 baseline 也实例化并前向同一 inverse head，但
  `inv_loss_weight=0`。因此两边结构、参数量、forward compute 和训练步数
  一致，仅辅助梯度不同。
- 默认正式训练改为 1,000,000 steps，旧 checkpoint 不续跑，run directory
  使用 `v2_1m` 后缀。
- logger 默认同时输出 JSONL、Scope 和标准 TensorBoard event 文件。项目
  使用轻量 `tensorboard` 依赖，不要求安装完整 TensorFlow。

当前运行：

| Domain/任务 | 模型 | GPU | PID | 参数量 | tmux | run directory |
|-------------|------|:---:|----:|-------:|------|---------------|
| DMC Walker | matched baseline | 1 | 2161276 | 168,144,780 | `btwm_v2_dv3_dmc_walker_1m_s0` | `runs/dreamerv3_dmc_walker_walk_seed0_v2_1m` |
| DMC Walker | BTWM v2 | 4 | 2161282 | 168,144,780 | `btwm_v2_full_dmc_walker_1m_s0` | `runs/btwm_dmc_walker_walk_seed0_v2_1m` |
| Atari Pong | matched baseline | 6 | 2211661 | 168,107,790 | `btwm_v2_dv3_pong_1m_s0` | `runs/dreamerv3_atari_pong_seed0_v2_1m` |
| Atari Pong | BTWM v2 | 7 | 2211656 | 168,107,790 | `btwm_v2_full_pong_1m_s0` | `runs/btwm_atari_pong_seed0_v2_1m` |

DMC 和 Atari 两组均已完成 JIT、保存首个 checkpoint、写出 metrics 和
TensorBoard event。记录时 DMC BTWM objective 约 0.51、accuracy 约 0.52；
Atari BTWM objective 约 0.040、accuracy 约 0.979。baseline 的同一 head
权重为零，objective 保持约 1，用于验证参数匹配而不参与优化。当前只是
训练健康度检查，最终性能必须等 1M 完成并扩展到多 seeds 后判断。

### BTWM v3：confidence-gated bidirectional control

v2 的 1M seed 0 结果：

- Pong 最后 50 局：baseline 13.78，BTWM v2 19.52，提升明显但仍需多 seed。
- DMC Walker BTWM 最后 50 局为 962.52；matched baseline 尚未完成时，
  共同 720k step 的最后 50 局为 baseline 950.05、BTWM 942.10，未显示
  稳定优势。

诊断表明 v2 只在 replay observation encoder 上施加 inverse loss，对 actor
的 imagined rollout 没有直接约束；同时统一强度的 representation gradient
会把不可逆或视觉歧义 transition 与高置信 transition 同等处理。

2026-06-15 实施 v3：

1. **双分支 inverse 优化**
   - detached-feature 分支用 `inv_head_loss_weight=0.02` 先稳定训练 head。
   - shared-feature 分支继续更新 encoder，但其权重乘以 inverse head 对真实
     动作的 confidence。head 尚未可靠时，避免高噪声辅助梯度破坏表征。
2. **选择性 actor weighting**
   - 将 replay start state 的 inverse confidence 转为 bounded、unit-mean
     权重。
   - 对该 start state 发起的 imagined policy loss 重权重，优先优化动作后果
     更可辨识、更可控的状态，同时保持平均 actor loss scale 不变。
3. **公平性**
   - 不新增网络参数。matched baseline 仍实例化同一 inverse head，但
     representation/head/policy 三个 BTWM 权重均为 0。
   - 同一 domain 的 baseline 与 BTWM 参数量、batch、train ratio、训练步数
     和 TensorBoard 输出完全一致。
4. **新增监控**
   - `train/inv/confidence`
   - `train/inv/effective_rep_weight`
   - `train/inv/head_objective`
   - `train/inv/policy_weight_{mean,std,min,max}`

9 个单元测试和完整 CPU debug training smoke 均通过。

当前 100k 开发筛选使用 seed 1；这只用于选择 v3 权重，不作为论文主结果：

| Domain | 变体 | GPU | PID | actor weight | run directory |
|--------|------|:---:|----:|-------------:|---------------|
| Pong | matched baseline | 4 | 793785 | 0.0 | `runs/dreamerv3_atari_pong_seed1_v3_screen_100k` |
| Pong | BTWM v3 medium | 6 | 793775 | 0.5 | `runs/btwm_atari_pong_seed1_v3_pw05_100k` |
| Pong | BTWM v3 strong | 7 | 793790 | 1.0 | `runs/btwm_atari_pong_seed1_v3_pw10_100k` |
| DMC Walker | BTWM v3 medium | 2 | 793797 | 0.5 | `runs/btwm_dmc_walker_walk_seed1_v3_pw05_100k` |
| DMC Walker | BTWM v3 strong | 3 | 793802 | 1.0 | `runs/btwm_dmc_walker_walk_seed1_v3_pw10_100k` |

所有筛选已完成 JIT 并写入 TensorBoard。早期梯度健康度上 medium 比 strong
更稳定：两个 domain 都有更低 inverse objective、更高 confidence/accuracy，
actor weight std 约 0.19--0.21，而 strong 约 0.40--0.47。必须等待 100k
实际回报后再选正式 1M 配置，不得把早期辅助指标当作性能提升。

2026-06-16 复核 v3 100k 筛选结论：

- 用户指出“v3 虽有提升，但 Pong 仍是负分，没有意义”。该判断成立。
  Pong v3 100k 只能说明相对早期学习速度，不能作为有效性能提升或论文主
  结论。最后一局分数仍为 baseline -8、v3 strong -7；即便相对更高，也
  没有进入可用策略区间。
- 当前可作为有效正分证据的是 v2 1M seed 0：Pong 最后 50 局 baseline
  13.78、BTWM v2 19.52，最后一局 baseline 14、BTWM v2 20。该结果仍需
  多 seed 验证，但比 v3 100k 负分更有解释意义。
- 代码层面已将正式默认回退为 v2：`inv_head_loss_weight=0.0`、
  `inv_confidence_gating=False`、`inv_policy_weight=0.0`。v3 的
  detached-head、confidence gating 和 actor weighting 只在显式设置
  `BTWM_VARIANT=v3` 时启用。
- `code/run_experiment.sh` 新增 `BTWM_VARIANT=v2|v3`，默认 `v2`。matched
  baseline 仍实例化相同 inverse head，并将 representation/head/policy
  权重置零，保持参数量、forward compute、训练步数和 TensorBoard 输出
  与 BTWM 公平对齐。
- 2026-06-16 验证：`code/test_btwm_head.py` 9 个单元测试通过；
  `rssm.py`/`agent.py` 语法编译通过；runner dry-run 确认 v2 命令为
  `head_weight=0.0 confidence_gating=False policy_weight=0.0`，v3 命令为
  `head_weight=0.02 confidence_gating=True policy_weight=1.0`。

---

## 论文定位

BTWM (Bidirectional Transition World Model) = DreamerV3 + 逆行动作预测头 `p(a_t | z_t, z_{t+1})`。
与 AAAI 2026 配套论文对齐。

## 实验矩阵

### Block 1 — Atari 100k 主结果 [P0]

10 款游戏，5 个类别，每类 2 款：

| # | 游戏 | 类别 | 动作数 |
|---|------|------|:---:|
| 1 | Montezuma's Revenge | 探索 | 18 |
| 2 | Private Eye | 探索 | 18 |
| 3 | Breakout | 精控 | 4 |
| 4 | Pong | 精控 | 6 |
| 5 | Alien | 生存 | 18 |
| 6 | Amidar | 生存 | 10 |
| 7 | Qbert | 解谜 | 6 |
| 8 | Ms Pacman | 解谜 | 9 |
| 9 | Boxing | 对战 | 18 |
| 10 | Seaquest | 对战 | 18 |

每种游戏跑：
- DreamerV3 基线
- BTWM（DreamerV3 + inverse head）
- 每游戏每模型 5 seeds
- 1,000,000 环境步/游戏

**小计：10 games × 2 models × 5 seeds = 100 runs**

### Block 2 — DMC 连续控制

4 个任务：walker_walk, cheetah_run, reacher_easy, finger_spin

每种任务跑：
- DreamerV3 基线
- BTWM
- 每任务每模型 5 seeds
- 1,000,000 环境步/任务

**小计：4 tasks × 2 models × 5 seeds = 40 runs**

### Block 3 — Crafter 探索

1 个任务：crafter_reward

- DreamerV3 基线
- BTWM
- 每模型 5 seeds

**小计：1 task × 2 models × 5 seeds = 10 runs**

### Block 4 — Ablation（仅 Atari Pong + Breakout）

6 个变体，确认每个组件的贡献：

| # | 变体 | 说明 |
|---|------|------|
| 1 | Full BTWM | 完整版 |
| 2 | No inverse loss | 去掉逆行动作损失，仅 policy masking |
| 3 | No policy masking | 去掉 policy masking，仅逆行动作损失 |
| 4 | No shared trunk | 逆行动作头用独立 encoder |
| 5 | No return relabeling | 去掉 return-conditioned relabeling |
| 6 | Forward-only | 纯 DreamerV3（等价 Block 1 基线） |

- 2 游戏 × 6 变体 × 5 seeds = 60 runs
- 每个 1,000,000 步

### Block 5 — Random Head 对照实验（仅 Pong）

- BTWM（真实逆行动作头）
- Random-aux：相同结构但输出随机标签
- 各 5 seeds

**小计：2 × 5 = 10 runs**

### 总计
| Block | Runs |
|-------|:---:|
| Atari 100k | 100 |
| DMC | 40 |
| Crafter | 10 |
| Ablation | 60 |
| Random head | 10 |
| **Total** | **220** |

使用 4-6 张 A100 并行跑，预计 4-7 天。

**优先级：Block 1 > Block 2 > Block 3 > Block 4 > Block 5**

---

## 服务器环境

```bash
cd /data2/liguoqi/habmm/hab_z/btwm
source venv_dv3/bin/activate
```

GPU: 8× A100 40GB

## 代码实现

### 核心修改（最小化，全部在 DreamerV3 代码中）

**1. 新增文件：`code/btwm_head.py`**

逆行动作预测头。**关键 bug 防范：离散/连续动作判断必须用 `act_space[k].discrete`，不能用 ndim。**

```python
import jax, jax.numpy as jnp, ninjax as nj, embodied.jax.nets as nn, numpy as np

class InverseActionHead(nj.Module):
    hidden_dim: int = 512
    depth: int = 3
    
    def __init__(self, act_space, **kw):
        self.act_space = act_space
    
    def __call__(self, z_t, z_next):
        x_t = jnp.concatenate([z_t['deter'], z_t['stoch'].reshape((*z_t['stoch'].shape[:-2], -1))], -1)
        x_n = jnp.concatenate([z_next['deter'], z_next['stoch'].reshape((*z_next['stoch'].shape[:-2], -1))], -1)
        x = jnp.concatenate([x_t, x_n], -1)
        for i in range(self.depth):
            x = self.sub(f'lin{i}', nn.Linear, self.hidden_dim)(x)
            x = nn.act('gelu')(self.sub(f'norm{i}', nn.Norm, 'rms')(x))
        logits = {}
        for k, space in self.act_space.items():
            num_actions = int(np.asarray(space.high).item())
            logits[k] = self.sub(f'out_{k}', nn.Linear, num_actions)(x)
        return logits

def inverse_loss(inv_logits, actions, act_space):
    """== BUG 防范：用 discrete 属性判断，不用 ndim == """
    total, n = None, 0
    for k in act_space.keys():
        logits, labels = inv_logits[k], actions[k]
        if act_space[k].discrete:
            ce = -jnp.sum(jax.nn.one_hot(labels, logits.shape[-1]) * jax.nn.log_softmax(logits), axis=-1)
        else:
            ce = -jnp.sum(jax.nn.one_hot(labels, logits.shape[-1]) * jax.nn.log_softmax(logits), axis=(-2, -1))
        total = ce if total is None else total + ce
        n += 1
    return total / max(n, 1)

def inverse_accuracy(inv_logits, actions, act_space):
    acc, n = 0.0, 0
    for k in act_space.keys():
        acc += jnp.mean(jnp.argmax(inv_logits[k], axis=-1) == actions[k])
        n += 1
    return acc / max(n, 1)
```

**2. 修改 `dreamerv3/dreamerv3/rssm.py`**

- `__init__` 新增参数 `use_btwm: bool = False, inv_loss_weight: float = 0.1`
- `__init__` 中如果 `use_btwm`，初始化 `InverseActionHead`
- `loss()` 方法加逆行动作损失（dyn/rep loss 之后）
- `loss_scales` 加 `inv: 1.0`

pad 安全处理：
```python
loss_val = inv_loss_val * self.inv_loss_weight
if loss_val.ndim >= 2:
    losses['inv'] = jnp.pad(loss_val, ((0,0),(0,1)))
else:
    losses['inv'] = jnp.zeros_like(dyn)
```

metrics 记录 `train/loss/inv` 和 `train/acc/inv`

**3. 修改 `dreamerv3/dreamerv3/configs.yaml`**

```yaml
agent:
  use_btwm: False
  inv_loss_weight: 0.05
  inv_num_bins: 21
  inv_hidden_dim: 256
  inv_depth: 2
  inv_feature_source: tokens
  inv_normalize_loss: True
loss_scales:
  inv: 1.0
replay:
  size: 500000  # 5e5，防止磁盘满
```

**4. 修改 `dreamerv3/dreamerv3/agent.py`**

将 `use_btwm` 和 `inv_loss_weight` 从 config 注入到 world model 的 dyn_cfg。

---

## 实验脚本模板

统一使用 `code/run_experiment.sh`。现有 `code/run_*.sh` 均为该入口的
兼容包装。

```bash
MODEL=btwm DOMAIN=atari TASK=pong GPU=0 SEED=0 RUN_STEPS=1000000 \
  bash code/run_experiment.sh

MODEL=dreamerv3 DOMAIN=dmc TASK=walker_walk GPU=1 SEED=0 RUN_STEPS=1000000 \
  bash code/run_experiment.sh
```

Atari 会自动使用 `atari100k` 配置；DMC 会自动设置 `MUJOCO_GL=egl`。
其他 DreamerV3 参数可追加在命令末尾。未实现的 ablation 不提供 flag。

### 批量启动

默认每张 A100 同时只跑 1 个任务；统一启动器会在空闲显存低于 30 GB
时拒绝启动。下面按 seed 顺序运行，每个 seed 的 baseline 和 BTWM 并行：

```bash
for seed in 0 1 2 3 4; do
  MODEL=dreamerv3 DOMAIN=atari TASK=pong GPU=0 SEED=$seed \
    bash code/run_experiment.sh &
  baseline_pid=$!
  MODEL=btwm DOMAIN=atari TASK=pong GPU=1 SEED=$seed \
    bash code/run_experiment.sh &
  btwm_pid=$!
  wait "$baseline_pid" "$btwm_pid"
done
```

---

## 依赖版本

```bash
pip install "jax[cuda12]==0.4.33" "jaxlib==0.4.33" "numpy<2"
pip install dm-control==1.0.9 mujoco==2.3.7
pip install crafter
```

## 日志和 TensorBoard

- 文本日志：`logs/<name>_seed<N>_YYYY-MM-DD_HHMMSS.log`
- TensorBoard：每个 `runs/<name>/events.out.tfevents.*`
- 汇总查看：`tensorboard --logdir runs/ --port 6006`

## 监控

```bash
ps aux | grep main.py | wc -l    # 运行数
nvidia-smi                        # GPU 占用
tail -f logs/btwm_pong_*.log      # 实时日志
grep -h "episode/score" logs/*.log | tail -20   # 最新分数
tensorboard --logdir runs/ --port 6006
df -h /data2                       # 磁盘（需保持 >50GB）
```

## 提取结果

所有实验完成后，对每个任务的 5 seeds 取最后 10 个 episode 分数的均值：
```bash
for f in logs/dreamerv3_pong_seed*.log; do
  echo "$f: $(grep 'episode/score' $f | tail -10 | grep -oP 'score \K[0-9.-]+' | awk '{s+=$1;n++}END{print s/n}')"
done
```

## 期望验证的 Claims

| Claim | 验证方式 |
|-------|---------|
| BTWM > DreamerV3 | Atari IQM + DMC mean return |
| 增益来自逆行动作损失 | Ablation: No-inv-loss vs Full |
| 不是来自额外容量 | Random head 对照 |
| 不是任意信号 | 逆行动作准确率应显著 > 随机 |
| 在连续控制也有效 | DMC walker/cheetah/reacher |
| 在探索型环境增益最大 | Montezuma's Revenge, Private Eye 的 Δ 应最大 |
