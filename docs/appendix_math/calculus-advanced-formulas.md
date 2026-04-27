# E.3.5 PG、DQN、GAE、PPO、GRPO 完整公式

## 进阶公式：策略梯度定理

前面我们已经看过单样本形式：

$$
\nabla_\theta J(\theta) \approx G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

完整的策略梯度定理写作：

$$
\nabla_\theta J(\theta)
=\sum_s d^\pi(s)\sum_a q_\pi(s,a)\nabla_\theta\pi_\theta(a\mid s).
$$

这里：

- $d^\pi(s)$ 是策略 $\pi$ 下状态 $s$ 被访问的频率。
- $q_\pi(s,a)$ 是动作价值，表示动作 $a$ 后未来平均回报。
- $\nabla_\theta\pi_\theta(a\mid s)$ 表示参数变化时，动作概率如何变化。

这个公式还可以变成更常用的 log 形式。因为：

$$
\nabla_\theta\pi_\theta(a\mid s)
=\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s),
$$

所以：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^\pi,a\sim\pi}
\left[q_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

如果用采样回报 $G_t$ 估计 $q_\pi(s_t,a_t)$，就得到 REINFORCE：

$$
\nabla_\theta J(\theta)
\approx
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

如果用优势函数替代动作价值，就得到更稳定的形式：

$$
\nabla_\theta J(\theta)
=\mathbb{E}
\left[A_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

这条推导说明：复杂公式并不是凭空出现的，它只是把“好动作概率上升、坏动作概率下降”写成了对所有状态和动作的平均。

## 进阶公式：价值函数近似的损失

Critic 或 DQN 需要学习一个价值估计。最常见目标是让预测值接近目标值。

给定样本 $(s_t,a_t,r_{t+1},s_{t+1})$，DQN 的 TD 目标是：

$$
y_t=r_{t+1}+\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a').
$$

其中 $\theta^-$ 表示目标网络参数。损失函数是：

$$
L(\theta)=\frac{1}{2}\left(Q_\theta(s_t,a_t)-y_t\right)^2.
$$

求梯度：

$$
\nabla_\theta L(\theta)
=\left(Q_\theta(s_t,a_t)-y_t\right)
\nabla_\theta Q_\theta(s_t,a_t).
$$

这个式子里，前一项是 TD 误差：

$$
\delta_t=y_t-Q_\theta(s_t,a_t).
$$

后一项告诉参数如何改变预测值。DQN 的训练就是不断减小这种预测误差。

## 进阶公式：GAE 从 TD 误差累积而来

优势函数可以用 TD 误差估计。一步 TD 误差是：

$$
\delta_t=R_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

如果 $\delta_t>0$，说明实际结果比 Critic 预期更好；如果 $\delta_t<0$，说明实际结果比预期更差。

GAE（Generalized Advantage Estimation）把多个时间步的 TD 误差折扣累加：

$$
\hat{A}^{GAE}_t
=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

其中 $\lambda\in[0,1]$ 控制偏差和方差的权衡：

- $\lambda$ 小：更依赖短期 TD 误差，方差较低，但偏差可能较大。
- $\lambda$ 大：更接近完整回报，偏差较低，但方差可能较大。

PPO 中常用 GAE，是因为它能在稳定性和准确性之间取得较好平衡。

## 进阶公式：PPO 裁剪目标

前面我们已经看过概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

PPO 的裁剪目标是：

$$
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

这个公式看起来复杂，但可以分两种情况理解。

如果 $\hat{A}_t>0$，说明动作比平均好。我们希望提高它的概率，但最多提高到 $1+\epsilon$ 倍。

如果 $\hat{A}_t<0$，说明动作比平均差。我们希望降低它的概率，但最多降低到 $1-\epsilon$ 倍。

因此 `min` 和 `clip` 的组合实现了一件事：**让策略朝正确方向更新，但不让它一次走太远**。

## 进阶公式：GRPO 的组归一化优势

GRPO 常用于一组回答的相对比较。假设同一个问题生成 $n$ 个回答，奖励分别为：

$$
r_1,r_2,\ldots,r_n.
$$

先计算均值：

$$
\mu=\frac{1}{n}\sum_{i=1}^n r_i.
$$

再计算标准差：

$$
\sigma=\sqrt{\frac{1}{n}\sum_{i=1}^n(r_i-\mu)^2}.
$$

每个回答的标准化优势是：

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma+\epsilon}.
$$

例如奖励是 $[2,4,10]$，均值是 $5.33$。第三个回答明显高于平均，优势为正；第一个回答低于平均，优势为负。这样模型不需要一个单独的 Critic，也可以利用组内相对好坏来更新策略。
