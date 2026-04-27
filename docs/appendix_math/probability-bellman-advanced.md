# E.2.5 贝尔曼期望方程与动作价值

## 进阶公式：贝尔曼期望方程的完整展开

前面我们把价值函数解释成：

$$
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
$$

现在把 $G_t$ 拆成“一步奖励 + 未来回报”：

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

代入价值函数：

$$
v_\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
$$

利用期望的线性性：

$$
v_\pi(s)=\mathbb{E}_\pi[R_{t+1}\mid S_t=s]+\gamma\mathbb{E}_\pi[G_{t+1}\mid S_t=s].
$$

第一项是即时奖励的平均值。完整展开为：

$$
\mathbb{E}_\pi[R_{t+1}\mid S_t=s]
=\sum_a \pi(a\mid s)\sum_r p(r\mid s,a)r.
$$

第二项是下一状态价值的平均值：

$$
\mathbb{E}_\pi[G_{t+1}\mid S_t=s]
=\sum_a \pi(a\mid s)\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

合在一起得到贝尔曼期望方程：

$$
v_\pi(s)=\sum_a\pi(a\mid s)
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
\right].
$$

这看起来复杂，但每一层都来自前面的简单例子：

- $\sum_a\pi(a\mid s)$：对策略可能选的动作求平均。
- $\sum_r p(r\mid s,a)r$：对可能奖励求平均。
- $\sum_{s'}p(s'\mid s,a)v_\pi(s')$：对可能下一状态的价值求平均。

## 进阶公式：动作价值与优势函数

动作价值函数定义为：

$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

它回答的是：已经在状态 $s$ 选择了动作 $a$，之后继续按策略 $\pi$ 行动，平均能拿多少回报。

它的贝尔曼形式是：

$$
q_\pi(s,a)=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

状态价值可以由动作价值加权得到：

$$
v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a).
$$

优势函数定义为：

$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s).
$$

它衡量动作 $a$ 比状态 $s$ 下的平均动作好多少。前面“回报 10、平均 8，所以优势是 2”的例子，就是这个公式的数值版本。

## 进阶公式：重要性采样的轨迹形式

单步重要性权重是：

$$
\rho_t=\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

如果一整条轨迹由行为策略 $b$ 采样，但我们想估计目标策略 $\pi$ 的回报，就要把每一步的概率比乘起来：

$$
\rho_{0:T}=\prod_{t=0}^{T}\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

于是异策略蒙特卡洛估计可以写成：

$$
\hat{v}_\pi(s)=\frac{1}{N}\sum_{i=1}^N \rho^{(i)}G^{(i)}.
$$

这个公式很强，但也有风险：如果很多概率比相乘，权重可能非常大，导致方差爆炸。因此实际算法常使用截断、加权重要性采样或其他更稳定的异策略方法。

## 进阶公式：协方差与策略梯度方差

两个随机变量一起变化的趋势可以用协方差表示：

$$
\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
$$

相关系数进一步把协方差标准化到 $[-1,1]$：

$$
\rho_{X,Y}=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

在强化学习中，梯度估计常常是随机变量。例如策略梯度样本：

$$
g_t=\hat{A}_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

如果 $\hat{A}_t$ 波动很大，$g_t$ 的方差也会变大。使用 baseline、advantage 标准化、GAE，本质上都是在控制这个随机梯度的方差。
