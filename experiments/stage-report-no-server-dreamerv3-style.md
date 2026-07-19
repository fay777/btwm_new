# BTWM 阶段汇报（未上服务器版本）

## 1. 当前定位

本次汇报按“尚未进入服务器正式大规模运行”这一前提组织，目标不是汇报最终结果，而是汇报：

1. BTWM 相对 DreamerV3 的方法改动是否已经落到代码中。
2. 本地是否已经完成最基本的正确性验证与短程训练验证。
3. 后续正式实验应如何参照 DreamerV3 的思路来设计对比、控制变量与汇报方式。

当前结论应限定为：**项目已经从方法设计阶段进入到可运行实现阶段，但还没有形成可用于论文主结论的正式 benchmark 结果。**

## 2. 方法目标与核心思路

BTWM 可以看作在 DreamerV3 world model 框架上的一项定向增强。核心出发点是：如果一个 latent transition 真实保留了决策相关的因果信息，那么模型不应只会从当前状态向前预测未来状态，也应能从相邻状态变化中反推导致该变化的动作。

因此，BTWM 的核心改动不是简单“加一个辅助 loss”，而是在 DreamerV3 的表征学习过程中加入 inverse-action 监督，让 world model 学到更强的动作可辨识表征，再把这种表征收益传递给后续策略学习。

当前仓库中的主线做法是：

1. 保留 DreamerV3 的主体训练框架。
2. 在 world model 中接入 inverse-action head。
3. 让 DreamerV3 matched baseline 保留同样的 head 结构，但把 BTWM 相关损失权重置零。
4. 用这种方式保证 BTWM 与 baseline 在参数量、前向计算路径和训练步数上尽量公平。

这套思路与 DreamerV3 论文的实验哲学是一致的：**统一配置、统一训练范式、跨任务尽量少调参、对比时强调公平和可复现。**

## 3. 目前已经完成的工作

### 3.1 方法与代码实现

结合仓库中的 `method-v1.md`、`memory.md` 和当前代码结构，BTWM 的关键实现已经进入 DreamerV3 fork 主路径，主要包括：

1. `code/btwm_head.py` 中的 inverse-action head 与相关损失计算。
2. `dreamerv3/dreamerv3/rssm.py` 中接入 inverse loss。
3. `dreamerv3/dreamerv3/agent.py` 中注入 BTWM 配置与训练诊断项。
4. `dreamerv3/dreamerv3/configs.yaml` 中加入 BTWM 相关超参数。
5. `code/run_experiment.sh` 作为统一实验入口，用于约束日志、目录命名和实验配置。

从项目文档记录来看，团队已经明确区分：

1. 正式实验入口必须统一。
2. 旧的 legacy 结果不能再用于论文比较。
3. baseline 必须采用 matched setting，而不是简单删除 head。

### 3.2 本地验证状态

在“未上服务器”的汇报口径下，目前最重要的不是分数，而是本地链路是否打通。现有材料可以支持以下判断：

1. 单元测试与语法检查路径已经准备好，`code/test_btwm_head.py` 被反复作为 smoke-test gate 的核心检查项。
2. 仓库里已有多组本地日志，说明最小训练链路已经跑通，BTWM 指标能够被正确记录。
3. `results_archive/README.md` 已明确声明 2026-06-13 correctness fixes 之前的结果无效，说明当前项目已经完成了一轮关键的正确性清理。

换句话说，当前最可靠的进展不是“性能已经显著提升”，而是：**实现路径、诊断指标、对比协议已经基本成型。**

## 4. 本地阶段可作为证据的结果

按照“未上服务器”这一前提，汇报里不能把零散日志包装成正式结论，但可以把它们作为“实现有效、训练正常”的证据。

### 4.1 Atari Pong 本地信号

从现有日志中可读到：

1. `BTWM v2, Pong, 1M, seed0` 的最后分数约为 `20`，最后 10 次平均约为 `19.9`。
2. `DreamerV3 matched baseline, Pong, 1M, seed0` 的最后分数约为 `14`，最后 10 次平均约为 `13.3`。
3. `BTWM v3` 的 100k seed1 筛选结果波动较大，`pw05` 最后约 `-9`，`pw10` 最后约 `-7`，baseline 约 `-8`。

这说明：

1. BTWM 至少在单个本地 Pong 日志上出现过正向信号。
2. 但 100k 短程结果方差很大，不能据此形成主结论。
3. v3 更适合当作变体探索，而不是当前阶段的正式主线结果。

### 4.2 DMC Walker Walk 本地信号

从现有日志中可读到：

1. `BTWM, 100k, seed0` 最后 return 约为 `846.05`。
2. `DreamerV3, 100k, seed0` 最后 return 约为 `767.99`。
3. `BTWM v2, 1M, seed0` 最后 return 约为 `977.32`。
4. `DreamerV3 matched baseline, 1M, seed0` 最后 return 约为 `985.98`。

这说明：

1. BTWM 在短程 100k 本地验证里出现过正向提升。
2. 在 1M 单 seed 对比里，BTWM 并没有稳定领先 baseline。
3. 连续控制场景下，BTWM 目前更像“有潜力但尚未收敛出稳定结论”。

### 4.3 诊断指标的意义

从 1M 日志里还能看到一个重要现象：

1. Pong 1M 中，BTWM 的 `train/acc/inv` 末期约 `0.94`，baseline 约 `0.11`。
2. Walker 1M 中，BTWM 的 `train/acc/inv` 末期约 `0.48`，baseline 约 `0.04`。

这表明 inverse 分支不是空转的，BTWM 的确在学习动作相关表征；但这仍然只是机制层信号，不等于最终性能结论。

## 5. 为什么汇报要参考 DreamerV3 的思路

`DreamerV3.pdf` 给我们的启发不在于具体数值，而在于实验组织方式：

1. **统一超参数**：DreamerV3 强调同一套配置跨任务使用，减少针对单任务反复调参。
2. **标准 benchmark 协议**：按 Atari、Control Suite 等标准协议汇报，而不是只挑少数有利样例。
3. **聚合指标 + 单任务曲线并行**：主文给 aggregate score，补充材料给单任务曲线。
4. **公平基线**：比较时明确协议、预算、模型规模与训练步数，避免“提升来自额外算力”这种质疑。
5. **主结论与支持结论分层**：先证明核心 claim，再补泛化和机制分析。

因此，本项目后续正式实验也应按这个逻辑来汇报，而不是直接堆若干零散分数。

## 6. 建议采用的正式实验与对比方式

结合 `experiments/protocol-v0.1.md` 和已有设计文档，后续正式汇报建议采用下面的结构。

### 6.1 主比较对象

同框架主比较：

1. `DreamerV3 matched baseline`
2. `BTWM v2`
3. `BTWM v3` 仅作为扩展变体或附录 ablation，不作为当前默认主结果

跨框架补充比较：

1. `LeWM`

但 LeWM 由于训练范式不同，应单独放在“cross-framework comparison”表格中，不与 DreamerV3/BTWM 主表直接混写。

### 6.2 主 benchmark

优先级建议保持为：

1. `DMC 4-task suite`
   - `walker_walk`
   - `cheetah_run`
   - `reacher_easy`
   - `finger_spin`
2. `Crafter`
   - `crafter_reward`
3. 后续再扩展 LeWM 对照任务

### 6.3 统一配置

建议正式汇报中明确写出：

1. DMC 和 Crafter 正式训练统一采用 `1,000,000` steps。
2. 种子统一采用 `0,1,2,3,4`。
3. replay、batch size、batch length、report length 固定。
4. baseline 使用同构 head，但将 `inv_loss_weight / inv_head_loss_weight / inv_policy_weight` 置零。

### 6.4 汇报指标

参考 DreamerV3 的写法，建议主汇报使用：

1. 每个任务最终 return 的 `mean ± std`。
2. 学习曲线图。
3. 如任务数足够，再增加 aggregate 指标，例如跨任务平均或 IQM。
4. BTWM 诊断指标单独成图，不与主性能表混合。

### 6.5 必做对照

为了让结论更像 DreamerV3 风格下的“可信比较”，至少要有：

1. matched DreamerV3 baseline。
2. `inv_loss_weight` sweep。
3. `inv_feature_source` sweep。
4. 若保留 v3，则需单独说明它是扩展变体，不与 v2 混为一个主结论。

## 7. 当前阶段最合理的结论

如果老师问“现在做到哪一步”，当前最稳妥的回答应是：

1. BTWM 相对 DreamerV3 的方法已经完成主路径实现。
2. 本地 smoke test、单任务短程训练和部分 1M 单 seed 日志说明训练链路是通的，inverse 监督也确实在起作用。
3. 但当前结果仍然以单任务、单 seed、短程验证为主，不能直接当作正式论文结论。
4. 下一步重点不是继续改叙述，而是按 DreamerV3 的实验哲学，把统一协议下的多任务、多 seed、公平对照跑完整。

## 8. 下一步计划

建议后续执行顺序为：

1. 先锁定主线变体为 `BTWM v2`。
2. 先完成 `DMC 4-task suite` 上的 matched baseline vs BTWM。
3. 再补 `Crafter` 以验证长时程像素任务上的迁移性。
4. 最后再把 LeWM 比较整理为单独的 cross-framework 表。

如果正式进入服务器阶段，汇报写法也应保持不变：**先给统一协议，再给主结果表，再给诊断和 ablation，最后给跨框架补充比较。**
