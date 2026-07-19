# BTWM Method Summary

## 1. 目标

BTWM（Bidirectional Transition World Model）的核心目标，不是把 inverse-action loss 当作一个孤立正则项，而是把它纳入一个完整的、双向约束的世界模型训练框架中。

它的出发点是：如果一个 latent transition 真正保留了可用于决策的因果信息，那么模型不仅应当能“向前预测”下一步状态和奖励，也应当能“反向解释”这一步状态变化对应的动作。

因此，BTWM 的设计不是“LeWorldModel + 一个辅助 loss”这么简单，而是：

- 在共享的 trajectory transformer 上，同时训练 forward / inverse / reward 三类预测任务；
- 在 world model 训练和 policy 训练阶段使用不同的 mask 机制；
- 用 inverse-action 监督增强表示学习，再把这种表示收益传递给后续 policy learning；
- 保持与 baseline 的参数量、训练轮数和计算预算公平对齐。

## 2. 方法定义

BTWM 基于 LeWorldModel 的 JEPA-style masked trajectory learning 架构，输入是一段 tokenized trajectory，形式上可以写成：

`[z_t, a_t, z_{t+1}, r_t, ...]`

其中：

- `z_t` 是状态 / latent token；
- `a_t` 是动作 token；
- `z_{t+1}` 是下一时刻状态 token；
- `r_t` 是奖励 token。

模型主体由三部分组成：

1. 共享 encoder / transformer trunk
2. forward head
3. inverse-action head
4. reward head

与 baseline 的区别，不在于把网络改成另一种完全不同的结构，而在于训练目标被改成了双向一致的 masked objective。

## 3. 三种 mask

BTWM 的训练关键是“同一段轨迹，使用不同 mask，学习不同预测任务”。

### 3.1 Forward mask `M_fwd`

forward mask 遮住部分未来状态 token `z_{t+1}`，模型要基于上下文去重建 / 预测未来 latent。

对应 LeWorldModel 的原始 forward dynamics 目标。

### 3.2 Inverse-action mask `M_inv`

inverse-action mask 遮住动作 token `a_t`，但保留 `z_t` 和 `z_{t+1}` 可见。

模型需要根据“前后状态差异”去预测动作：

`p(a_t | z_t, z_{t+1})`

这是 BTWM 的主要新增项。它不是 exploration bonus，也不是单独挂在旁边的正则，而是对 latent transition 可逆性的显式约束。

### 3.3 Reward mask `M_rew`

reward mask 遮住奖励 token `r_t`，模型根据周围上下文预测奖励。

这部分继承自 LeWorldModel，用于保持奖励建模能力。

## 4. 训练目标

BTWM 的 world model 训练损失由三部分组成：

`L_BTWM = L_fwd + L_inv + L_rew`

其中：

- `L_fwd`：forward latent prediction loss，通常使用 cosine / embedding similarity；
- `L_inv`：inverse-action classification loss，离散动作用交叉熵，连续动作按分箱离散化后再做分类；
- `L_rew`：reward regression loss，通常是 MSE。

实现上，三种 mask 可以在同一个 batch 内同时采样、同时计算，然后一起反传。

这点很重要：BTWM 的效果不应被理解成“额外加了一个 loss，所以更强”，而是“用反向监督改变了共享表征的学习方式”。

## 5. policy learning

BTWM 不只训练 world model，也要训练 policy。

policy 训练使用 causal mask `M_pol`，即：

- 只允许看见当前位置及其之前的 token；
- 不能访问未来 token；
- 避免信息泄漏。

policy 仍然通过 imagined rollouts 来优化，目标是最大化 `lambda-return` 或其对应的 actor-critic objective。

这里的关键设计是：

- world model 阶段使用 bidirectional / masked learning；
- policy 阶段使用 left-to-right causal learning；
- 两者共享表示，但约束方式不同。

这样做的理由是：world model 需要“看前看后”去学更好的 transition 表征，而 policy 必须在真实决策时“只能看过去”。

## 6. Return relabeling

在论文设计里，return-conditioned relabeling 也是方法的一部分。

其作用不是单独创造新算法，而是把历史轨迹中的回报信号重新整理成更适合 policy 学习的监督形式，让过去的轨迹能够作为更稳定的训练样本。

它和 inverse-action loss 的关系是：

- inverse loss 提升 transition representation 的可辨识性；
- relabeling 提升 policy 训练的样本利用效率；
- 两者分别作用在 representation 和 policy 的不同环节。

## 7. 为什么这不是“正则项”

把 BTWM 只写成 regularizer 是不够的，原因有三点：

1. 训练对象不同  
   它不是只约束一个 embedding norm 或 smoothness，而是直接预测动作 token。

2. 作用路径不同  
   inverse-action supervision 通过 shared trunk 改变 latent 表征，再影响后续 policy，而不是只在最后加一点辅助项。

3. 方法叙事不同  
   论文的创新点应该表述为“bidirectional world model learning”，不是“额外加了一个 loss”。

## 8. 和 baseline 的公平比较

为了让实验成立，BTWM 和 baseline 必须保持以下公平条件：

- 参数量尽量一致；
- 训练轮数一致；
- batch size、rollout length、优化器设置一致；
- same compute budget；
- 记录相同的 TensorBoard 指标；
- matched baseline 可以实例化同样的 head，但把 BTWM 相关损失权重置零，以保证参数和前向开销一致。

这比“简单删掉 head”更严谨，因为可以避免参数量差异成为性能来源。

## 9. 当前项目里的实现原则

当前项目中，BTWM 的实现应该围绕以下主线展开：

1. 保留 LeWorldModel 的稳定 backbone；
2. 在共享 trunk 上加入 inverse-action 监督；
3. 让 inverse 监督真正参与 representation learning，而不是只作为一个边缘指标；
4. 在 policy 阶段维持 causal 约束；
5. 所有实验都用 1M 轮级别的训练做主比较，短跑只用于健康检查；
6. 保存 TensorBoard，方便后续可视化与论文作图。

## 10. 当前风险点

如果实现偏离上面的主线，BTWM 很容易退化成下面几种弱形式：

- 只是在 embedding 上挂一个 inverse classifier；
- inverse head 过强，学到动作泄漏而不是 transition 语义；
- 只做短轮次训练，结论被 early learning noise 污染；
- baseline 和 BTWM 训练预算不一致，导致比较不公平；
- 没有区分 world model loss 和 policy loss，方法边界变模糊。

这些都应该避免。

## 11. 一句话定义

BTWM = 在 LeWorldModel 的 JEPA 世界模型上，引入 inverse-action 的双向 masked 监督，并把这种双向表征能力传递给 causal policy learning 的方法。

