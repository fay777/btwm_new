# BTWM 进度报告

**日期：** 2026-06-29  
**范围：** BTWM 项目状态、方法对齐情况、以及当前已有的实验依据

## 1. 总结

BTWM 已经从纯设计阶段进入到了可工作的实现阶段。

- 方法定义已经在 [method-v1.md](/data2/liguoqi/habmm/hab_z/btwm/method-v1.md) 中明确。
- 核心 BTWM 改动已经合入 DreamerV3 fork。
- 本地已经有 Atari Pong 和 DMC Walker Walk 的训练日志。
- 少数运行结果显示出不错的提升，但这些证据目前还比较少，而且不同 seed 和不同设置之间不够稳定。
- 归档的旧分数文件已经明确标注为 legacy，不能用于论文主张。

总体判断：**实现已经到位，但 benchmark 证据还没有完整到可以支撑最终论文级结论。**

## 2. 方法状态

[method-v1.md](/data2/liguoqi/habmm/hab_z/btwm/method-v1.md) 里的方法摘要把 BTWM 定义为一个双向 JEPA 风格的世界模型，包含：

- 共享 encoder / transformer trunk，
- forward prediction，
- inverse-action prediction，
- reward prediction，
- 在学到的表征上做 causal policy learning，
- 与 LeWorldModel 做 compute matched 对比。

这和当前 DreamerV3 fork 的代码方向是一致的：

- inverse-action 逻辑已经接入 RSSM / agent 的训练路径，
- policy 的加权可以依赖 inverse confidence，
- 增加了 TensorBoard fallback 支持，
- config 默认值里已经有 BTWM 相关开关。

所以当前实现不是草图，而是已经落到真实代码路径上的功能。

## 3. 代码进展

### DreamerV3 fork

当前工作区里已经有这些改动：

- 在 `rssm.py` 中集成了 inverse-action head 和 loss，
- 在 `agent.py` 中加入了 confidence 指标和 policy weighting，
- 在 `configs.yaml` 中加入了 BTWM 相关配置项，
- 在 `main.py` 中加入了 TensorBoard fallback 处理，
- 依赖里增加了 `tensorboard`。

这意味着主训练栈现在已经能够同时表达 forward 和 inverse 两类目标。

### LeWorldModel 基线

LeWorldModel 这边也已经为对比实验做了一些准备：

- 已经有 Pong 专用的数据配置，
- 对 1D 列的归一化做了修复，
- 已经创建了 Atari Pong 的启动脚本，
- 本地存在多个尝试运行的输出目录。

基线侧目前更像是实验准备和 smoke test 阶段，而不是已经完全冻结的 benchmark 套件。

## 4. 实验依据

### 4.1 `main.pdf` 里的论文草稿结果

PDF 草稿里已经包含了一套完整叙事和结果表。里面的关键结论是：

- Atari 100k：
  - BTWM IQM = 1.82
  - LeWorldModel IQM = 1.62
  - DreamerV3 IQM = 1.74
  - STORM IQM = 1.68
- DMC：
  - BTWM 在 manipulation 类任务上更强，在 locomotion 上大体接近。
- Meta-World：
  - BTWM success rate = 82%
  - LeWorldModel success rate = 72%
- Ablation：
  - 去掉 inverse loss 的掉点最大，
  - compute-matched 的 LeWorldModel 没有追平，
  - random auxiliary head 不能复现这个提升。

这些就是当前论文草稿所依赖的核心结果。

### 4.2 本地可核实日志

仓库里还保留了能直接检查的本地日志，下面这些结果是可以直接从日志里看到的。

#### Atari Pong

本地观察到的结果：

- 早期 100k seed0 smoke run：
  - BTWM 最终可见 `episode/score` = 4
  - DreamerV3 最终可见 `episode/score` = 10
- 后期 1M seed0 compute-matched run：
  - BTWM 最终可见 `episode/score` = 20
  - DreamerV3 最终可见 `episode/score` = 14
- seed1 100k 变体：
  - BTWM `pw05` 最终可见 `episode/score` = -9
  - BTWM `pw10` 最终可见 `episode/score` = -7
  - DreamerV3 baseline `screen` 变体最终可见 `episode/score` = -8

解读：

- 1M Pong 这组是当前本地最强的正向信号，
- 100k Pong 这些结果波动较大，还不足以单独支撑结论。

相关日志：

- [btwm_atari_pong_seed0_v2_1m_2026-06-13_183636.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_atari_pong_seed0_v2_1m_2026-06-13_183636.log)
- [dreamerv3_atari_pong_seed0_v2_1m_2026-06-13_183636.log](/data2/liguoqi/habmm/hab_z/btwm/logs/dreamerv3_atari_pong_seed0_v2_1m_2026-06-13_183636.log)
- [btwm_atari_pong_seed1_v3_pw05_100k_2026-06-15_105522.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_atari_pong_seed1_v3_pw05_100k_2026-06-15_105522.log)
- [btwm_atari_pong_seed1_v3_pw10_100k_2026-06-15_105522.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_atari_pong_seed1_v3_pw10_100k_2026-06-15_105522.log)
- [dreamerv3_atari_pong_seed1_v3_screen_100k_2026-06-15_105522.log](/data2/liguoqi/habmm/hab_z/btwm/logs/dreamerv3_atari_pong_seed1_v3_screen_100k_2026-06-15_105522.log)

#### DMC Walker Walk

本地观察到的结果：

- 100k seed0：
  - BTWM = 846.05
  - DreamerV3 = 767.99
- 1M seed0 compute-matched：
  - BTWM = 977.32
  - DreamerV3 = 985.98
- seed1 100k BTWM 变体：
  - `pw05` = 864.48
  - `pw10` = 763.12

解读：

- BTWM 在 100k seed0 上有明确提升，
- 1M compute-matched 这组还没有显示出稳定优势，
- policy weight / confidence 的设置仍然比较敏感。

相关日志：

- [btwm_dmc_walker_walk_seed0_2026-06-13_120235.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_dmc_walker_walk_seed0_2026-06-13_120235.log)
- [dreamerv3_dmc_walker_walk_seed0_2026-06-13_120235.log](/data2/liguoqi/habmm/hab_z/btwm/logs/dreamerv3_dmc_walker_walk_seed0_2026-06-13_120235.log)
- [btwm_dmc_walker_walk_seed0_v2_1m_2026-06-13_182555.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_dmc_walker_walk_seed0_v2_1m_2026-06-13_182555.log)
- [dreamerv3_dmc_walker_walk_seed0_v2_1m_2026-06-13_182555.log](/data2/liguoqi/habmm/hab_z/btwm/logs/dreamerv3_dmc_walker_walk_seed0_v2_1m_2026-06-13_182555.log)
- [btwm_dmc_walker_walk_seed1_v3_pw05_100k_2026-06-15_105522.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_dmc_walker_walk_seed1_v3_pw05_100k_2026-06-15_105522.log)
- [btwm_dmc_walker_walk_seed1_v3_pw10_100k_2026-06-15_105522.log](/data2/liguoqi/habmm/hab_z/btwm/logs/btwm_dmc_walker_walk_seed1_v3_pw10_100k_2026-06-15_105522.log)

## 5. 目前还缺什么

当前证据还不够做成一个干净的最终结论，原因是：

- 本地运行只覆盖了少数任务，
- seed 覆盖还不完整，
- 有些运行只是 smoke test 或者不同配置变体，不是干净的 matched comparison，
- 归档结果集是 legacy，只能参考不能引用，
- 论文草稿里的完整 Atari 100k / DMC / Meta-World 矩阵，还没有被一个可复现的本地结果归档完整支撑起来。

## 6. 风险提示

当前项目的主要风险仍然是这些：

- inverse-action 监督可能太弱或者太噪，
- policy weighting 如果设置不好，会扭曲结果，
- 短跑结果如果被当成最终证据，会误导判断，
- baseline 和 BTWM 的比较必须严格 compute matched。

仓库里已经明确记录了一个重要问题：

- 2026-06-13 correctness fixes 之前生成的 legacy score 文件，不能用于主张。

见 [results_archive/README.md](/data2/liguoqi/habmm/hab_z/btwm/results_archive/README.md)。

## 7. 结论

BTWM 已经不再只是一个概念。它已经实现、部分验证，并且在一些本地运行里显示出不错的提升。

但是，从研究汇报的角度看，这个项目目前还处在 **实验验证阶段**，还没有进入 **结果完全收敛阶段**。

下一步最实际的里程碑，是把核心 benchmark 做成一份干净、seed 对齐的结果表，并锁定哪些设置足够稳定、可以进入论文正文。
