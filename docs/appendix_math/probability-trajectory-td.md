# E.2.4 轨迹概率、Baseline 与 GAE

## 第 09 讲：一条轨迹的概率怎么写

前面我们分别看了策略概率 $\pi(a\mid s)$ 和环境转移概率 $p(s'\mid s,a)$。现在把它们串起来。

假设一条轨迹是：

$$
\tau=(s_0,a_0,s_1,a_1,s_2).
$$

它的发生概率由三类东西相乘得到：

1. 初始状态出现的概率 $p(s_0)$。
2. 每一步策略选择动作的概率 $\pi(a_t\mid s_t)$。
3. 每一步环境转移的概率 $p(s_{t+1}\mid s_t,a_t)$。

所以：

$$
p(\tau\mid\pi)=p(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
$$

看一个两步数字例子：

| 项                                  | 数值   |
| ----------------------------------- | ------ |
| 初始状态概率 $p(s_0)$               | $0.6$  |
| 第一步动作概率 $\pi(a_0\mid s_0)$   | $0.5$  |
| 第一步转移概率 $p(s_1\mid s_0,a_0)$ | $0.8$  |
| 第二步动作概率 $\pi(a_1\mid s_1)$   | $0.25$ |
| 第二步转移概率 $p(s_2\mid s_1,a_1)$ | $0.4$  |

那么这条轨迹的概率是：

$$
0.6\times0.5\times0.8\times0.25\times0.4=0.024.
$$

策略梯度的推导之所以离不开概率，就是因为目标函数本质上是在所有可能轨迹上求平均：

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[G(\tau)].
$$

这句话的展开含义是：

$$
J(\theta)=\sum_{\tau}p_\theta(\tau)G(\tau).
$$

也就是说，每条轨迹的回报 $G(\tau)$ 都按它在当前策略下出现的概率加权。

## 第 10 讲：baseline 为什么不改变期望

策略梯度中常把回报 $G_t$ 换成 $G_t-b(s_t)$。直觉上，这是把“绝对分数”改成“比平均好多少”。但为什么这样不改变梯度方向的期望？

关键是下面这个项为零：

$$
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)b(s)\right]=0.
$$

先把与动作无关的 $b(s)$ 提出来：

$$
b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

利用对数导数技巧：

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}.
$$

代入后：

$$
b(s)\sum_a\pi_\theta(a\mid s)
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s).
$$

把梯度移到求和外面：

$$
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s).
$$

而动作概率之和永远等于 $1$：

$$
\sum_a\pi_\theta(a\mid s)=1.
$$

所以：

$$
b(s)\nabla_\theta 1=0.
$$

这说明减去 baseline 不改变梯度的期望，只改变方差。于是策略梯度可以写成：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
(G_t-b(s_t))
\right].
$$

当 $b(s_t)=V^\pi(s_t)$ 时，括号里的项就是优势估计：

$$
A^\pi(s_t,a_t)=G_t-V^\pi(s_t).
$$

## 第 11 讲：从 TD error 到 GAE

时序差分误差 TD error 是：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

先用数字看它。假设：

$$
r_t=2,\qquad \gamma=0.9,\qquad V(s_{t+1})=5,\qquad V(s_t)=6.
$$

那么：

$$
\delta_t=2+0.9\times5-6=0.5.
$$

这表示：这一步实际看到的“奖励 + 下一状态价值”比当前估计高 $0.5$，所以当前状态价值可能低估了。

一步 TD error 方差低，但可能偏差大；完整蒙特卡洛回报偏差低，但方差大。GAE 用一串 TD error 做加权平均：

$$
\hat{A}_t^{GAE(\gamma,\lambda)}
=\sum_{k=0}^{T-t-1}(\gamma\lambda)^k\delta_{t+k}.
$$

如果 $\lambda=0$，只保留第一项：

$$
\hat{A}_t=\delta_t.
$$

如果 $\lambda$ 接近 $1$，后面很多步的 TD error 都会参与，结果更接近蒙特卡洛优势。于是 $\lambda$ 成了一个“偏差-方差旋钮”：

| $\lambda$ | 更像什么 | 特点             |
| --------- | -------- | ---------------- |
| $0$       | 一步 TD  | 方差低，偏差较高 |
| $1$       | 蒙特卡洛 | 偏差低，方差较高 |
| $0.95$    | 折中     | PPO 中常用       |

## 第 12 讲：PPO 裁剪目标里的概率思想

PPO 的核心目标是：

$$
L^{CLIP}(\theta)
=
\mathbb{E}\left[
\min\left(
r_t(\theta)\hat{A}_t,\,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right],
$$

其中：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}.
$$

这个概率比本质上就是重要性采样权重。旧策略收集了样本，但我们想评估新策略，所以要用“新策略概率 / 旧策略概率”修正。

看一个数字例子。旧策略下某动作概率是 $0.2$，新策略下变成 $0.3$：

$$
r_t=\frac{0.3}{0.2}=1.5.
$$

如果优势 $\hat{A}_t=4$，未裁剪项是：

$$
r_t\hat{A}_t=1.5\times4=6.
$$

若 $\epsilon=0.2$，允许范围是 $[0.8,1.2]$，裁剪后：

$$
\mathrm{clip}(1.5,0.8,1.2)\times4=4.8.
$$

取两者较小值：

$$
\min(6,4.8)=4.8.
$$

所以 PPO 的公式虽然看起来复杂，其实结合了三层意思：

1. 用概率比做异策略修正。
2. 用优势函数决定动作该加强还是削弱。
3. 用裁剪限制策略变化幅度。
