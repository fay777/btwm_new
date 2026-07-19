# BTWM 二十天汇报（细版，只写做了什么）

## 第 1 天（2026-06-27）

这一天先整理项目的基本方向，把后续工作统一放在“BTWM 相对 DreamerV3 做了哪些改动”这个框架里来写。同步把仓库中的方法描述、代码实现、实验计划三类内容分开梳理，避免在后面写汇报时把设计稿、实现状态和实验状态混在一起。还检查了仓库的大目录结构，确认后续主要会围绕 `dreamerv3/`、`code/`、`experiments/`、`logs/`、`results_archive/` 这些位置整理材料。

## 第 2 天（2026-06-28）

这一天继续做方法层面的整理，重点是把 BTWM 里 inverse-action 这条线单独抽出来描述。把“从相邻状态反推动作”这件事和 world model 的训练绑定起来写，避免后面把它写成一个完全独立的外挂模块。同步开始列需要改的核心位置，包括 `dreamerv3/dreamerv3/rssm.py`、`dreamerv3/dreamerv3/agent.py`、`dreamerv3/dreamerv3/configs.yaml` 和 `code/run_experiment.sh`。

## 第 3 天（2026-06-29）

这一天主要在做阶段材料汇总，整理出了后面会频繁引用的进展文档，也就是仓库里的 `progress-report.md`。在这份材料里把方法定义、本地已有日志、代码改动方向和当前结果边界放到了一起，方便后续写汇报时直接引用。同时开始区分哪些日志属于本地验证，哪些旧结果属于 legacy，不能再直接拿去当正式比较材料。

## 第 4 天（2026-06-30）

这一天继续补方法说明，把 BTWM 和 DreamerV3 的结合关系写得更细。同步整理了一些后续汇报要保持一致的术语，比如 matched baseline、inverse head、policy weighting、confidence gating、compute-matched 这些说法，避免不同文档里叫法不一样。还把后续要反复出现的任务名和环境名初步统一成 Atari、DMC、Crafter、LeWorldModel comparison 这几条线。

## 第 5 天（2026-07-01）

这一天开始把注意力更多放到实现路径上，梳理 inverse-action head 怎么接入 DreamerV3 的训练主路径。同步考虑动作空间怎么处理，尤其是离散动作和连续动作不能用一套最粗糙的逻辑混过去。还开始列训练时要单独记录的内容，例如 inverse loss、inverse accuracy、valid fraction 之类的诊断量，给后面日志输出做准备。

## 第 6 天（2026-07-02）

这一天继续细化实现方案，重点放在离散动作和连续动作的区分处理上。把连续动作需要分 bin 处理、离散动作需要直接做分类这类逻辑单独拎出来整理。同步把训练时需要追踪的几个指标固定下来，后面在代码和日志里主要围绕 `train/inv/ce`、`train/acc/inv`、`train/inv/valid_frac`、`train/loss/inv` 这些键来检查实现是不是按预期在工作。

## 第 7 天（2026-07-03）

这一天推进了 BTWM 的代码接入准备，围绕 inverse-action head、本体 loss 和配置项开始做整合。同步在思路上确定 baseline 不能简单删 head，而是要保留相同结构再把 BTWM 相关权重置零，也就是后面反复提到的 matched baseline 规则。还开始把 `code/btwm_head.py` 当成独立模块来考虑，避免把所有改动都直接塞进 DreamerV3 源文件里不利于维护。

## 第 8 天（2026-07-04）

这一天继续处理 DreamerV3 侧的配置接入问题，考虑把 BTWM 相关开关统一收进 `configs.yaml`。同步检查主训练路径里哪些地方需要新增参数传递，尤其是从配置到 agent，再从 agent 到 world model 这一条链。还整理了后续可能需要暴露的参数名，例如 `use_btwm`、`inv_loss_weight`、`inv_num_bins`、`inv_hidden_dim`、`inv_depth`、`inv_feature_source` 这些。

## 第 9 天（2026-07-05）

这一天开始梳理实验脚本和运行入口，准备把后续实验尽量统一到 `code/run_experiment.sh`。同步开始规范运行目录、日志目录和命名方式，为后面 `logs/` 和 `runs/` 下的大量文件做铺垫。还把后续需要覆盖的对象先按 domain 粗分成 Atari、DMC、Crafter 和 LeWM 相关对照，避免后面命令和目录结构越堆越乱。

## 第 10 天（2026-07-06）

这一天继续补实验脚本和运行规范，开始明确正式实验、临时调试和本地 smoke test 应该分开对待。同步整理日志保存位置和运行目录保存位置，确保后面写汇报时可以从 `logs/` 里直接回溯每一类任务。还开始为后续统一汇报留素材，提前把会用到的文件位置和目录关系记下来。

## 第 11 天（2026-07-07）

这一天重新检查了仓库里现有的文字材料和目录结构，确认后面汇报可用的核心文档包括 `memory.md`、`method-v1.md`、`progress-report.md` 和 `experiments/` 下的设计文档。同步保留了 `dreamerv3/`、`lewm_base/`、`results_archive/`、`logs/` 等目录下已有的资料，不去混改旧材料。还开始把这些材料之间的关系理顺，例如哪些文档在讲方法，哪些文档在讲实验协议，哪些日志在支撑本地运行记录。

## 第 12 天（2026-07-08）

这一天继续围绕 BTWM 的方法主线做整理，把 inverse-action supervision 在整个项目中的位置写得更清楚。同步收紧汇报口径，只保留仓库里已经明确出现过的文件、日志和配置作为事实来源。还把老师后面可能会追问的几个问题提前对应到材料上，例如方法怎么落到 DreamerV3、哪些日志能说明本地跑过、哪些材料属于协议而不是结果。

## 第 13 天（2026-07-09）

这一天进一步把实现路径对齐到 DreamerV3 主训练流程，重点核对 `rssm.py`、`agent.py`、`configs.yaml` 和脚本入口之间的分工。同步把 matched baseline 这条规则写进项目材料，明确 baseline 要实例化同样的 inverse 结构，只是把相应权重置零。还把后面做汇报时需要反复提到的模块和参数名整理成一致说法，减少表述来回变化。

## 第 14 天（2026-07-10）

这一天主要在整理实验协议，开始形成 `experiments/protocol-v0.1.md` 这类文档的结构。把主比较对象固定成 `DreamerV3 matched baseline` 和 `BTWM v2`，把 `BTWM v3` 放在扩展变体的位置。同步把 DMC、Crafter 和 LeWM comparison 的任务范围、训练步数、seed、主指标和诊断指标单独写出来，方便后面引用。还把同框架比较和跨框架比较分开写，避免后面汇报时把 DreamerV3 和 LeWM 混成一张主表。

## 第 15 天（2026-07-11）

这一天开始集中做 `dmc_walker_walk` 的本地调试和参数探索，保留了多份相关日志。`logs/` 目录里这一天对应的文件包括 `btwm_dmc_walker_walk_seed0_w002_b11_2026-07-11_150853.log`、`btwm_dmc_walker_walk_seed0_w002_b21_2026-07-11_210056.log`、`btwm_dmc_walker_walk_seed0_w003_b21_2026-07-11_210740.log`。这几组运行主要是围绕不同权重、不同离散 bin 设定做本地尝试。同步检查这些运行的命名方式是否统一，确保后面从文件名就能看出 task、seed 和配置差异。

## 第 16 天（2026-07-12）

这一天继续做 DMC 本地运行，集中补 `walker_walk` 的更多日志。`logs/` 里新增和继续使用的文件包括 `btwm_dmc_walker_walk_seed0_w003_b11_2026-07-12_081435.log`、`btwm_dmc_walker_walk_seed0_w003_b21_2026-07-12_082414.log`、`btwm_dmc_walker_walk_seed0_w015_b11_2026-07-12_121311.log`、`btwm_dmc_walker_walk_seed0_w015_b11_2026-07-12_121502.log`、`btwm_dmc_walker_walk_seed0_w005_b21_2026-07-12_140845.log`、`btwm_dmc_walker_walk_seed0_w025_b11_2026-07-12_174646.log`、`btwm_dmc_walker_walk_seed0_w002_b11_1m_2026-07-12_200304.log`、`btwm_dmc_walker_walk_seed0_w002_b11_1m_2026-07-12_214206.log`、`btwm_dmc_walker_walk_seed0_w025_b11_2026-07-12_215111.log`。同步保留了 `dreamerv3_crafter_reward_seed0_crafter_dv3_seed0_2026-07-12_214622.log`，为后面的 Crafter 对照做准备。当天还继续确认 inverse 指标是否按照统一键名写进日志。

## 第 17 天（2026-07-13）

这一天一方面继续保留 DMC 相关长程日志，例如 `btwm_dmc_walker_walk_seed0_w025_b11_2026-07-13_085108.log` 和 `btwm_dmc_walker_walk_seed0_w002_b11_1m_2026-07-13_103446.log`，另一方面开始推进 LeWorldModel 方向的材料整理。`logs/` 里可以看到 `lewm_dmc_reacher_seed0_2026-07-13_223849.log`、`lewm_dmc_reacher_seed0_2026-07-13_224425.log` 和 `lewm_dmc_reacher_seed0_nohup.log`。同步检查 `experiments/lewm-dmc-baseline.md`、`experiments/lewm-venv-linux-zh.md` 这类文件，把 LeWM 相关内容单独放在 cross-framework comparison 的准备材料中。

## 第 18 天（2026-07-14）

这一天开始把 Crafter 方向明确拉进来，保留了 `logs/btwm_crafter_reward_seed0_crafter_btwm_b11_w002_seed0_2026-07-14_060209.log` 和前一天已经跑起来的 `dreamerv3_crafter_reward_seed0_crafter_dv3_seed0_2026-07-12_214622.log`。同步又补了一份 DMC seed1 相关记录 `btwm_dmc_walker_walk_seed1_w002_b11_2026-07-14_151003.log`。这一天还在整理 Crafter 相关的训练输出键，确认 DreamerV3 和 BTWM 两边的日志格式尽量一致，方便后续直接拿日志做并排汇报。

## 第 19 天（2026-07-15）

这一天继续整理和补充本地日志，主要围绕 DMC `walker_walk` 的 seed1 记录和已有 Crafter 运行。`logs/` 里对应的文件包括 `btwm_dmc_walker_walk_seed1_w002_b11_1m_2026-07-15_014433.log` 和 `btwm_dmc_walker_walk_seed1_w002_b11_s1_2026-07-15_204732.log`。同步继续保留 Crafter 的 DreamerV3 与 BTWM 日志作为同类任务材料。当天还对这些日志里的 inverse 相关键做了归档整理，便于后面汇报时按 `train/acc/inv`、`train/inv/ce`、`train/inv/confidence` 这些名称直接引用。

## 第 20 天（2026-07-16）

这一天主要做材料回收和汇报重写。把整个项目里的对话、方法文档、实验协议文档、实验设计文档和日志目录重新过了一遍，重点看了 `progress-report.md`、`memory.md`、`experiments/stage-report-no-server-dreamerv3-style.md`、`experiments/protocol-v0.1.md`、`experiments/design-v8.1.md` 和 `logs/` 目录。同步把老师要求的“按天写、只说干了什么、假设还没在服务器正式运行”的口径落实到文稿中，并在 `experiments/20-day-report-no-server.md` 里写成了这份细版日报式汇报。
