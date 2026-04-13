# E.2 概率与统计

> 相关章节：[第3章 MDP 形式化](/chapter03_mdp/formalism)、[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)、[第6章 GAE](/chapter06_ppo/gae-reward-model)

概率论是强化学习的地基。MDP 本质上是一个概率框架——状态转移是随机的（$P(s' \mid s, a)$），策略输出的是概率分布（$\pi(a \mid s)$），回报是随机变量的期望。策略梯度定理、GAE、重要性采样——这些核心算法全是概率推导的产物。

## 条件概率

条件概率回答的问题是："已知 B 发生了，A 发生的概率是多少？"

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

这不是一个需要死记的公式。它的直觉很简单：把样本空间从"所有可能"缩小到"B 已经发生"的范围，然后在缩小后的范围里看 A 占多少。

**RL 中的条件概率无处不在**。状态转移概率 $P(s' \mid s, a)$ 是"已知当前状态是 $s$、采取了动作 $a$，下一状态是 $s'$ 的概率"。策略 $\pi(a \mid s)$ 是"已知当前状态是 $s$，选择动作 $a$ 的概率"。整个 MDP 就是条件概率的链条。

## 贝叶斯定理

贝叶斯定理把条件概率"反过来"：

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

直觉：你想知道 $P(A \mid B)$（已知 B 发生，A 的概率），但你手上有的是 $P(B \mid A)$（已知 A 发生，B 的概率）。贝叶斯告诉你怎么翻过来。

各部分的命名：$P(A)$ 是**先验**——在看到证据之前你对 A 的信念。$P(B \mid A)$ 是**似然**——如果 A 为真，看到 B 的概率有多大。$P(A \mid B)$ 是**后验**——看到证据之后你对 A 的更新信念。

**RL 中的贝叶斯**。贝叶斯强化学习用贝叶斯定理维护对环境（转移概率、奖励函数）的后验分布。每收集一条新数据，就更新后验。偏好学习中也用到贝叶斯思想——Bradley-Terry 模型通过观察到的偏好 $(y_w \succ y_l)$ 反推奖励函数的分布。

## 随机变量与期望

随机变量是一个"值不确定"的量。掷骰子的结果、股票明天的价格、RL 中智能体在某个状态能获得的回报——都是随机变量。

**期望**（Expected Value）是随机变量的"加权平均"：

$$\mathbb{E}[X] = \sum_x x \cdot P(x)$$

对连续随机变量，求和变成积分：

$$\mathbb{E}[X] = \int x \cdot p(x) \, dx$$

期望的核心性质（线性性）：

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

这个性质在策略梯度推导中反复出现。策略梯度定理 $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a \mid s) \cdot G_t]$ 的推导，本质上就是把期望拆开、利用线性性重组的过程。

**RL 中的期望**。价值函数 $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_0 = s]$ 就是回报的期望。策略梯度是 $\nabla \log \pi \cdot A$ 的期望。几乎所有 RL 目标函数都是期望的形式——因为环境的随机性，我们只能优化"平均表现"。

## 方差与标准差

方差度量随机变量的"波动程度"：

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

方差的平方根是标准差 $\sigma = \sqrt{\text{Var}(X)}$。

直觉：方差告诉你"实际值偏离平均值的程度有多大"。方差大，意味着结果不稳定——可能很好也可能很差。方差小，意味着结果比较确定。

**方差与策略梯度**。策略梯度方法的最大痛点就是高方差。$\nabla_\theta J = \mathbb{E}[\nabla \log \pi(a \mid s) \cdot G_t]$ 中的 $G_t$ 是一个方差很大的随机变量（因为它是多步回报的累积）。引入 baseline（通常取 $V(s)$）不改变期望，但能显著降低方差：

$$\nabla_\theta J = \mathbb{E}[\nabla \log \pi(a \mid s) \cdot (G_t - b(s))]$$

这是 REINFORCE → Actor-Critic 演化的核心动机。

## 常用概率分布

### 伯努利分布

最简单的分布，只有两种结果（成功/失败）：

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

期望 $\mathbb{E}[X] = p$，方差 $\text{Var}(X) = p(1-p)$。

RL 中，二值奖励（任务完成/未完成）就是伯努利分布。

### 分类分布（Categorical Distribution）

伯努利分布的推广——不是两种结果，而是 $K$ 种：

$$P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1$$

**这是离散动作空间 RL 中最核心的分布**。策略网络 $\pi_\theta(a \mid s)$ 输出的就是一个分类分布——每个动作的概率 $p_k$。采样就是按概率随机选一个动作。训练时用 $\log \pi_\theta(a \mid s)$ 计算策略梯度。

### 高斯分布（正态分布）

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

钟形曲线，由均值 $\mu$ 和方差 $\sigma^2$ 决定。

**这是连续动作空间 RL 中最核心的分布**。连续策略通常参数化为 $\pi_\theta(a \mid s) = \mathcal{N}(a \mid \mu_\theta(s), \sigma_\theta(s)^2)$——网络输出均值和标准差，动作从这个高斯分布中采样。SAC、PPO 在连续控制中都用高斯策略。

高维情形（多元高斯）：

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\Sigma}$ 是协方差矩阵。对角协方差（各维度独立）是最常见的简化。

## 大数定律与蒙特卡洛估计

大数定律说的是：当样本量 $N \to \infty$ 时，样本均值趋近期望：

$$\frac{1}{N}\sum_{i=1}^{N} x_i \xrightarrow{N \to \infty} \mathbb{E}[X]$$

这不是一个抽象的定理——它是蒙特卡洛方法的全部理论基础。

**RL 中的蒙特卡洛**。MC 回报估计 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 就是对价值函数的蒙特卡洛估计。跑一条完整轨迹，把沿途奖励加起来，作为 $V(s)$ 的估计。跑的轨迹越多（$N$ 越大），估计越准。但每条轨迹都是高方差的——这正是 TD 学习和 GAE 要解决的问题。

## 重要性采样

重要性采样让你在"用一个分布采样的样本"来估计"另一个分布下的期望"：

$$\mathbb{E}_{x \sim P}[f(x)] = \mathbb{E}_{x \sim Q}\left[f(x) \cdot \frac{P(x)}{Q(x)}\right]$$

其中 $\frac{P(x)}{Q(x)}$ 叫做**重要性权重**（importance weight）。

直觉：你想知道"在策略 $P$ 下表现如何"，但你手上只有"在策略 $Q$ 下收集的数据"。重要性采样给每条数据乘一个权重——如果 $P$ 比 $Q$ 更倾向选这个动作，权重就大；反之权重就小。

**RL 中的重要性采样**。PPO 和 GRPO 的概率比 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ 就是重要性权重。它让你用旧策略 $\pi_{\theta_{old}}$ 收集的数据来更新新策略 $\pi_\theta$，不需要每次更新都重新采样。

**重要性采样的风险**。当 $P$ 和 $Q$ 差异很大时，重要性权重会变得极大（某些样本的权重是其他样本的几千倍），导致估计方差爆炸。PPO 的裁剪机制 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 就是为了防止这个问题——当新旧策略差异太大时，直接截断权重。

## 协方差与相关系数

协方差度量两个随机变量的线性关系：

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

标准化后得到相关系数 $\rho \in [-1, 1]$：

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

在分析 reward model 的评分与人类偏好的一致性时，相关系数是常用的评估指标。

## 公式速查

| 概念       | 公式                                                  | RL 中的角色                 |
| ---------- | ----------------------------------------------------- | --------------------------- |
| 条件概率   | $P(A \mid B) = P(A \cap B) / P(B)$                    | MDP 状态转移、策略分布      |
| 贝叶斯定理 | $P(A \mid B) = P(B \mid A) P(A) / P(B)$               | 贝叶斯 RL、偏好学习         |
| 期望       | $\mathbb{E}[X] = \sum x \cdot P(x)$                   | 价值函数、策略梯度          |
| 方差       | $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | baseline 的动机、GAE        |
| 分类分布   | $P(X=k) = p_k$                                        | 离散动作策略                |
| 高斯分布   | $\mathcal{N}(\mu, \sigma^2)$                          | 连续动作策略                |
| 重要性采样 | $\mathbb{E}_P[f] = \mathbb{E}_Q[f \cdot P/Q]$         | PPO/GRPO 的 off-policy 修正 |
