# E.4.3 PPO、RLHF 与 DPO 中的信息论

## 第 07 讲：PPO 和 RLHF 中的 KL 约束

假设旧策略对某个 token 的概率是 $0.01$，新策略突然变成 $0.20$。概率比是：

$$
\frac{0.20}{0.01}=20.
$$

这说明新策略对这个 token 的偏好突然放大了 20 倍。即使这个 token 在当前样本上带来了高奖励，也可能是过拟合或奖励黑客。

KL 惩罚的思想是：

$$
\text{优化目标} = \text{奖励} - \beta D_{KL}(\pi_{new}\|\pi_{old}).
$$

如果新策略离旧策略太远，第二项会变大，从而抵消奖励收益。这样可以让策略更新更稳。

在 RLHF 中，KL 约束还承担另一个作用：让对齐后的模型不要偏离初始语言模型太远，避免语言能力或通用行为突然退化。

## 第 08 讲：交叉熵、熵和 KL 的关系

前面分别看了熵、交叉熵和 KL。它们之间其实有一个非常漂亮的关系：

$$
D_{KL}(P\|Q)=H(P,Q)-H(P).
$$

先看定义：

$$
H(P,Q)=-\sum_x P(x)\log Q(x),
$$

$$
H(P)=-\sum_x P(x)\log P(x),
$$

两者相减：

$$
H(P,Q)-H(P)
=
-\sum_xP(x)\log Q(x)
+\sum_xP(x)\log P(x).
$$

合并成：

$$
\sum_xP(x)\log\frac{P(x)}{Q(x)}
=D_{KL}(P\|Q).
$$

这说明：**KL 散度就是用错误分布 $Q$ 描述真实分布 $P$ 时，多出来的编码成本**。

用分类任务理解也很自然：

- 如果模型分布 $Q$ 很接近真实分布 $P$，交叉熵接近真实熵，KL 小。
- 如果 $Q$ 偏离 $P$，交叉熵变大，多出来的部分就是 KL。

所以在机器学习里，最小化交叉熵经常等价于让模型分布靠近真实分布。

## 第 09 讲：DPO 公式里的对数概率比

DPO 的损失函数看起来比较吓人：

$$
\mathcal{L}_{DPO}
=
-\mathbb{E}\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right)
\right].
$$

先不要急着看整体。只看一个核心部件：

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

它比较的是：当前模型相对于参考模型，有多偏好回答 $y$。

看一个数字例子。对某个回答 $y$：

| 模型                           | 概率   |
| ------------------------------ | ------ |
| 当前模型 $\pi_\theta(y\mid x)$ | $0.20$ |
| 参考模型 $\pi_{ref}(y\mid x)$  | $0.05$ |

概率比是：

$$
\frac{0.20}{0.05}=4.
$$

对数概率比是：

$$
\log 4.
$$

这说明当前模型比参考模型更偏好这个回答。DPO 比较 winner 和 loser 的对数概率比：

$$
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}.
$$

如果 winner 的相对概率提高，loser 的相对概率降低，这个差值就变大，损失就变小。

DPO 和 KL 的关系在于：参考模型 $\pi_{ref}$ 不只是被动比较对象，它还像一根“锚”。当前模型不能只追求偏好数据，还会通过概率比受到参考模型约束。直觉上，这和 RLHF 中的：

$$
J(\pi)=\mathbb{E}_\pi[r(x,y)]-\beta D_{KL}(\pi_\theta\|\pi_{ref})
$$

是同一类思想：**提高偏好回答的概率，但不要离参考模型太远**。
