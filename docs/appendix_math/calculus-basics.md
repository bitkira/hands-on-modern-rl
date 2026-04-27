# E.3.1 微积分基础：导数、梯度与链式法则

## 第 00 讲：函数、目标和变化率

微积分关注的是“一个量变化时，另一个量如何变化”。在强化学习中，我们最关心的是：参数 $\theta$ 变化时，目标函数 $J(\theta)$ 会如何变化。

可以先把目标函数理解成一个输入输出关系：

$$
\theta \longmapsto J(\theta).
$$

如果 $J(\theta)$ 表示平均回报，那么训练就是寻找让 $J(\theta)$ 更大的参数。导数和梯度回答的问题是：**现在站在这个参数位置，往哪个方向走，目标会上升最快？**

所以微积分不是孤立的公式，而是优化算法的语言。

## 第 01 讲：从一维函数看导数

先看一个简单目标函数：

$$
J(\theta)=-(\theta-0.8)^2+1.
$$

可以把 $\theta$ 理解成一个非常简化的策略参数。例如 $\theta=0.2$ 表示选择 right 的概率较小，$\theta=0.8$ 表示选择 right 的概率较大。

这个函数在 $\theta=0.8$ 时最大。我们先不看图，只看几个数字：

| $\theta$ | $J(\theta)$ |
| -------- | ----------- |
| $0.2$    | $0.64$      |
| $0.5$    | $0.91$      |
| $0.8$    | $1.00$      |
| $1.0$    | $0.96$      |

从 $0.2$ 到 $0.5$，目标变大；从 $0.8$ 到 $1.0$，目标变小。导数就是描述“当前位置附近，函数往哪个方向变化更快”的工具。

对这个函数求导：

$$
J'(\theta)=-2(\theta-0.8).
$$

在 $\theta=0.2$ 时：

$$
J'(0.2)=1.2.
$$

导数为正，说明往右调会让目标上升。在 $\theta=1.0$ 时：

$$
J'(1.0)=-0.4.
$$

导数为负，说明往左调会让目标上升。

## 第 02 讲：梯度上升与梯度下降

如果目标是最大化回报，就用梯度上升：

$$
\theta \leftarrow \theta + \alpha J'(\theta).
$$

从 $\theta=0.2$ 开始，学习率取 $\alpha=0.1$：

$$
\theta \leftarrow 0.2 + 0.1\times1.2 = 0.32.
$$

再算一次导数：

$$
J'(0.32)=-2(0.32-0.8)=0.96.
$$

继续更新：

$$
\theta \leftarrow 0.32 + 0.1\times0.96 = 0.416.
$$

参数一步步靠近 $0.8$。

如果目标是最小化损失函数 $L(\theta)$，就用梯度下降：

$$
\theta \leftarrow \theta - \alpha L'(\theta).
$$

深度学习里常说的“反向传播 + 优化器”，核心就是计算梯度并更新参数。

## 第 03 讲：从导数到梯度

真实模型通常有很多参数。假设目标函数有两个参数：

$$
J(\theta_1,\theta_2)=-(\theta_1-1)^2-(\theta_2-2)^2+5.
$$

它在 $(1,2)$ 处最大。梯度是对每个参数分别求导后组成的向量：

$$
\nabla J(\theta_1,\theta_2)=
\begin{bmatrix}
-2(\theta_1-1) \\
-2(\theta_2-2)
\end{bmatrix}.
$$

如果当前参数是 $(0,0)$，梯度是：

$$
\nabla J(0,0)=
\begin{bmatrix}
2 \\
4
\end{bmatrix}.
$$

这表示：$\theta_1$ 应该增加，$\theta_2$ 也应该增加，而且 $\theta_2$ 增加得更快。学习率取 $0.1$，一次梯度上升后：

$$
\begin{bmatrix}
\theta_1 \\
\theta_2
\end{bmatrix}
\leftarrow
\begin{bmatrix}
0 \\
0
\end{bmatrix}
+0.1
\begin{bmatrix}
2 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\
0.4
\end{bmatrix}.
$$

神经网络有成千上万个参数，梯度就是一个同样维度的向量，告诉每个参数应该怎么动。

## 第 04 讲：链式法则为什么重要

神经网络是很多函数一层层复合起来的。链式法则告诉我们：外层变化如何通过中间变量传到内层参数。

看一个具体例子：

$$
y = 3\theta, \qquad L=(y-6)^2.
$$

如果 $\theta=1$，那么 $y=3$，损失是：

$$
L=(3-6)^2=9.
$$

我们想知道 $\theta$ 变一点，$L$ 怎么变。直接展开：

$$
L=(3\theta-6)^2.
$$

求导得到：

$$
\frac{dL}{d\theta}=2(3\theta-6)\times3.
$$

在 $\theta=1$ 时：

$$
\frac{dL}{d\theta}=2(3-6)\times3=-18.
$$

这里的 $2(3\theta-6)$ 是损失对 $y$ 的导数，$3$ 是 $y$ 对 $\theta$ 的导数。链式法则把它们乘起来：

$$
\frac{dL}{d\theta}=\frac{dL}{dy}\frac{dy}{d\theta}.
$$

反向传播就是在大规模神经网络里高效应用链式法则。

## 第 05 讲：偏导数与多参数模型

真实策略网络不只有一个参数，而是有很多参数。设目标函数是：

$$
J(\theta_1,\theta_2)=-(\theta_1-1)^2-(\theta_2-2)^2+5.
$$

它有两个参数。我们可以分别问：固定 $\theta_2$ 时，$\theta_1$ 怎么影响目标？固定 $\theta_1$ 时，$\theta_2$ 怎么影响目标？这就是偏导数。

对 $\theta_1$ 的偏导是：

$$
\frac{\partial J}{\partial \theta_1}=-2(\theta_1-1).
$$

对 $\theta_2$ 的偏导是：

$$
\frac{\partial J}{\partial \theta_2}=-2(\theta_2-2).
$$

把所有偏导数排成向量，就得到梯度：

$$
\nabla_\theta J=
\begin{bmatrix}
\frac{\partial J}{\partial \theta_1} \\
\frac{\partial J}{\partial \theta_2}
\end{bmatrix}.
$$

在 $(\theta_1,
\theta_2)=(0,0)$ 处：

$$
\nabla_\theta J(0,0)=
\begin{bmatrix}
2 \\
4
\end{bmatrix}.
$$

这表示两个参数都应该增大，而且第二个参数的上升方向更强。神经网络的梯度也是同样道理，只是参数数量从 2 个变成了几百万甚至更多。

## 第 06 讲：学习率决定“走多远”

梯度只告诉我们方向，不告诉我们应该走多远。学习率 $\alpha$ 决定每一步的长度。

如果当前参数是

$$
\theta=
\begin{bmatrix}
0 \\
0
\end{bmatrix},
\qquad
\nabla J=
\begin{bmatrix}
2 \\
4
\end{bmatrix},
$$

学习率取 $\alpha=0.1$，一次梯度上升是：

$$
\theta\leftarrow
\begin{bmatrix}
0 \\
0
\end{bmatrix}
+0.1
\begin{bmatrix}
2 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\
0.4
\end{bmatrix}.
$$

如果学习率过小，训练很慢；如果学习率过大，参数可能越过最优点甚至发散。强化学习的梯度本来就噪声较大，所以学习率、梯度裁剪、Adam、PPO 裁剪都会影响稳定性。

## 第 07 讲：从数学梯度到代码里的反向传播

在代码中，我们通常不会手写每个参数的偏导数，而是让自动微分系统计算。自动微分依赖的正是链式法则。

例如一个简单计算图：

$$
\theta \to y=3\theta \to L=(y-6)^2.
$$

反向传播从最后的损失开始，先算 $\frac{dL}{dy}$，再乘上 $\frac{dy}{d\theta}$，得到 $\frac{dL}{d\theta}$。

这和深度网络一样：每一层只需要知道本层的局部导数，整条链路的梯度就能通过链式法则传回去。策略网络、价值网络和奖励模型的训练都依赖这套机制。
