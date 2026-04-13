# E.3 微积分与优化

> 相关章节：[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)、[第6章 PPO 数学推导](/chapter06_ppo/ppo-math)、[第8章 GRPO](/chapter08_grpo_rlvr/grpo-mechanism)

微积分是优化的语言，优化是 RL 的核心操作。训练一个 RL 模型就是不断调整参数 $\theta$，让某个目标函数（期望回报、DPO 偏好损失等）变大或变小。微积分告诉你"往哪个方向调"和"调多少"。

## 导数的几何意义

函数 $f(x)$ 在点 $x$ 处的导数 $f'(x)$ 是该点切线的斜率：

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

**直觉**：导数回答的是"如果我往右挪一点点，函数值会怎么变？"如果导数为正，函数在增长；导数为负，函数在下降；导数为零，函数在这个点是平的（可能是极值点）。

为什么这很重要？因为优化就是"找到函数下降最快的方向"。导数告诉你局部变化率——沿着导数反方向走，函数值就会下降。这就是梯度下降的全部思想。

## 链式法则

链式法则处理复合函数的导数。如果 $y = f(g(x))$，那么：

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**直觉**：变化是"逐层传递"的。$x$ 变了一点 → $g$ 变了一点 → $f$ 变了一点。总变化率是每一层变化率的乘积。

链式法则是反向传播（backpropagation）的数学基础。神经网络是嵌套的复合函数：$\hat{y} = f_L(f_{L-1}(\cdots f_1(\mathbf{x})\cdots))$。反向传播就是从输出层开始，逐层应用链式法则，计算每个参数的梯度。

**RL 中的链式法则**。策略梯度 $\nabla_\theta \log \pi_\theta(a \mid s)$ 的计算就是链式法则的应用。$\pi_\theta$ 是一个神经网络，$\log \pi$ 对 $\theta$ 的梯度通过反向传播逐层反传。PPO 损失函数 $\nabla_\theta L^{CLIP}$ 更复杂——它涉及概率比 $r_t(\theta) = \pi_\theta / \pi_{\theta_{old}}$ 对 $\theta$ 的梯度，同样是链式法则的展开。

## 梯度：多变量的导数

当函数有多个变量时，导数升级为**梯度**。函数 $f(\mathbf{x})$ 在点 $\mathbf{x}$ 处的梯度是一个向量，每个分量是 $f$ 对对应变量的偏导数：

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**梯度的核心性质**：梯度指向函数增长最快的方向。因此，**负梯度指向函数下降最快的方向**。

这不是一个显然的事实。直觉上的解释：梯度的每个分量 $\frac{\partial f}{\partial x_i}$ 告诉你 $f$ 沿 $x_i$ 方向的变化率。梯度把这些信息组合成一个向量，指向"综合增长最快"的方向。可以证明（通过 Cauchy-Schwarz 不等式），沿梯度方向的"爬升速度"大于任何其他方向。

这就是为什么梯度下降 $\theta \leftarrow \theta - \alpha \nabla f(\theta)$ 有效——它沿着最陡的下降方向走。

## 多变量微积分扩展

### Jacobian 矩阵

当输出也是多维的时（$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$），偏导数组成 Jacobian 矩阵：

$$\mathbf{J} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

Jacobian 描述了多输入多输出函数的局部线性近似。梯度是 Jacobian 的特例（$m = 1$ 时的转置）。

### Hessian 矩阵

二阶偏导数组成 Hessian 矩阵：

$$\mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

Hessian 描述了函数的**曲率**——梯度的变化率。在优化中：

- Hessian 正定（所有特征值 $> 0$）→ 局部极小值
- Hessian 负定（所有特征值 $< 0$）→ 局部极大值
- 特征值混合 → 鞍点

二阶优化方法（如牛顿法）用 Hessian 的逆来调整步长。在 RL 中，TRPO 的 KL 约束可以用 Fisher 信息矩阵（Hessian 的期望版本）来近似，从而实现更稳定的策略更新。

## 梯度下降的演化

### 基础梯度下降

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

每一步都沿负梯度方向走 $\alpha$ 步。$\alpha$ 是学习率——太大会震荡发散，太小会收敛极慢。

### 随机梯度下降（SGD）

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta; \mathbf{x}_i)$$

用单个样本（或小批量）估计梯度，而不是全量数据。代价是梯度估计有噪声，但好处是：(1) 每步计算量小；(2) 噪声有隐式正则化效果，帮助逃离局部极小值。

RL 天然就是"随机"的——每条轨迹只是真实梯度的一个噪声估计。REINFORCE 就是用一条轨迹的样本来估计策略梯度。

### 动量法（Momentum）

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta)$$

$$\theta \leftarrow \theta - \alpha \mathbf{v}_t$$

动量法累积历史梯度，让更新方向更稳定——像球滚下山坡，有惯性。在 RL 中不太常用，因为 RL 的梯度噪声太大，动量反而会累积误差。

### Adam

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta \leftarrow \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam 结合了动量（一阶矩 $m_t$）和自适应学习率（二阶矩 $v_t$）。$\hat{m}_t$ 和 $\hat{v}_t$ 是偏差校正后的估计值。

**为什么 Adam 在 RL 中最常用？** RL 的梯度噪声极大（因为环境的随机性 + 策略的探索），而且不同参数的梯度尺度差异很大（网络前几层和后几层的梯度量级可能差几个数量级）。Adam 的自适应学习率为每个参数单独调整步长，自动适应这种差异。

默认超参数 $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ 在大多数 RL 任务中无需调整。

## 凸优化基础

凸函数有一个非常好的性质：**局部最优 = 全局最优**。

形式化地说，函数 $f$ 是凸函数，当且仅当对任意 $\mathbf{x}, \mathbf{y}$ 和 $\lambda \in [0, 1]$：

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$$

直觉：凸函数像一只碗——从任何位置出发，梯度下降都能到达碗底（全局最优）。

**RL 是凸优化吗？不是。** RL 的目标函数几乎都是非凸的——策略参数 $\theta$ 和目标函数 $J(\theta)$ 之间的关系极其复杂（多层神经网络嵌套），存在大量局部极小值和鞍点。这正是 RL 训练困难的根本原因之一。

但凸优化的概念在 RL 中仍然有用：

- **正则化项**是凸的。权重衰减 $\lambda \|\theta\|_2^2$ 是凸函数，加到非凸目标上可以让"优化景观"更平滑。
- **奖励模型训练**中的交叉熵损失是凸的（对最后一层线性参数而言），所以 reward model 的训练通常比策略优化更稳定。
- **DPO 损失**虽然整体非凸，但对 $\beta$ 的选择等超参数的分析可以借鉴凸优化的结论。

## Taylor 展开与局部近似

Taylor 展开用多项式近似一个函数。一阶 Taylor 展开：

$$f(x + \delta) \approx f(x) + f'(x) \cdot \delta$$

在多维情形下：

$$f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\delta}$$

这就是梯度下降的理论依据——在 $\mathbf{x}$ 附近，函数可以被线性近似。要让 $f$ 下降最快，就选让 $\nabla f(\mathbf{x})^\top \boldsymbol{\delta}$ 最小的 $\boldsymbol{\delta}$，即 $\boldsymbol{\delta} = -\alpha \nabla f(\mathbf{x})$。

PPO 的裁剪目标也有 Taylor 展开的影子。概率比 $r_t(\theta) = \pi_\theta / \pi_{\theta_{old}}$ 在 $\theta = \theta_{old}$ 附近的一阶展开是一个常数（$r_t = 1$），二阶项才体现出变化。KL 约束本质上就是限制二阶项的大小。

## 公式速查

| 概念        | 公式                                                                                          | RL 中的角色             |
| ----------- | --------------------------------------------------------------------------------------------- | ----------------------- |
| 导数        | $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$                                              | 优化基础                |
| 链式法则    | $\frac{d}{dx}f(g(x)) = f'(g) \cdot g'(x)$                                                     | 反向传播、策略梯度计算  |
| 梯度        | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^\top$                  | 参数更新方向            |
| 梯度下降    | $\theta \leftarrow \theta - \alpha \nabla \mathcal{L}$                                        | 所有 RL 算法的优化基础  |
| Adam        | $\theta \leftarrow \theta - \alpha \hat{m}/(\sqrt{\hat{v}} + \epsilon)$                       | RL 最常用的优化器       |
| 一阶 Taylor | $f(\mathbf{x}+\boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f^\top \boldsymbol{\delta}$ | PPO/GRPO 的局部近似分析 |
