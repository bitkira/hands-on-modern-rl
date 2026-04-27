# E.3.4 策略梯度、Taylor 与 GRPO 推导

## 第 09 讲：对数导数技巧如何推出策略梯度

策略梯度最常见的形式是：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi[
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
].
$$

这个公式的关键是“对数导数技巧”。我们从一个简单恒等式开始：

$$
\nabla_\theta \log \pi_\theta(a\mid s)
=
\frac{\nabla_\theta \pi_\theta(a\mid s)}
{\pi_\theta(a\mid s)}.
$$

两边乘以 $\pi_\theta(a\mid s)$：

$$
\nabla_\theta \pi_\theta(a\mid s)
=
\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

现在看一个离散动作下的目标：

$$
J(\theta)=\sum_a \pi_\theta(a\mid s)Q^\pi(s,a).
$$

对参数求梯度：

$$
\nabla_\theta J(\theta)
=\sum_a \nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a).
$$

代入对数导数技巧：

$$
\nabla_\theta J(\theta)
=\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
Q^\pi(s,a).
$$

这就变成了一个期望：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
[
\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)
].
$$

更完整地考虑所有状态，就得到：

$$
\nabla_\theta J(\theta)
=
\sum_s d^\pi(s)\sum_a
\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a),
$$

或者等价写成采样形式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)Q^\pi(s_t,a_t)
].
$$

实际算法中 $Q^\pi(s_t,a_t)$ 不容易精确知道，所以常用 $G_t$ 或优势估计 $\hat{A}_t$ 替代：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t
].
$$

这就是 REINFORCE、Actor-Critic 和 PPO 共同使用的核心梯度结构。

## 第 10 讲：Taylor 展开、Hessian 与 PPO 的二阶直觉

一阶 Taylor 展开说：

$$
f(x+h)\approx f(x)+f'(x)h.
$$

看一个数字例子。令：

$$
f(x)=x^2,\qquad x=3,\qquad h=0.1.
$$

真实值是：

$$
f(3.1)=9.61.
$$

一阶近似是：

$$
f(3)+f'(3)h=9+6\times0.1=9.6.
$$

已经很接近。二阶 Taylor 展开再加上曲率项：

$$
f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2.
$$

对 $f(x)=x^2$，$f''(x)=2$，所以：

$$
9+6\times0.1+\frac{1}{2}\times2\times0.1^2=9.61.
$$

多变量时，二阶项中的 $f''$ 变成 Hessian 矩阵 $H$：

$$
f(\theta+\Delta\theta)
\approx
f(\theta)
\nabla f(\theta)^\top\Delta\theta
\frac{1}{2}\Delta\theta^\top H\Delta\theta.
$$

PPO 和 TRPO 背后的信任域思想，正是担心参数更新过大时，一阶近似不再可靠，二阶曲率项开始变得重要。

对 PPO 的概率比：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)},
$$

在 $\theta_{old}$ 附近展开：

$$
r_t(\theta)
\approx
1
+\nabla_\theta r_t^\top(\theta-\theta_{old})
+\frac{1}{2}(\theta-\theta_{old})^\top
\nabla_\theta^2 r_t
(\theta-\theta_{old}).
$$

这里三项分别是：

| 项     | 含义                           |
| ------ | ------------------------------ |
| $1$    | 新旧策略相同时，概率比为 $1$   |
| 一阶项 | 小步更新带来的线性变化         |
| 二阶项 | 步子变大后，曲率带来的额外变化 |

PPO 的裁剪不是显式计算 Hessian，但它通过限制 $r_t(\theta)$ 的范围，间接避免高阶项失控。

## 第 11 讲：GRPO 的组归一化

GRPO 中常见的优势估计来自同一问题的一组回答。假设同一个 prompt 采样 4 个回答，奖励分别是：

$$
r=[2,4,6,8].
$$

均值是：

$$
\mu=\frac{2+4+6+8}{4}=5.
$$

标准差是：

$$
\sigma=
\sqrt{
\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}
}
=\sqrt{5}.
$$

第 4 个回答的标准化优势是：

$$
\hat{A}_4=\frac{8-5}{\sqrt{5}}\approx1.34.
$$

一般形式是：

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma}.
$$

这个公式的含义非常直接：

1. 减去均值：判断这个回答比组内平均好还是差。
2. 除以标准差：把不同题目的奖励尺度统一起来。

GRPO 之所以能省掉传统 PPO 里的 Critic，一个关键原因就是它用组内相对比较构造 baseline。它不问“这个回答绝对多少分”，而问“这个回答在同组里排得怎么样”。
