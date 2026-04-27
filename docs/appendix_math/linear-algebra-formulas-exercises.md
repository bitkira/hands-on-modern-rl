# E.1.5 线性代数公式速查与练习

## 本书中你会遇到的线性代数公式

| 概念           | 公式                                                                                                               | 强化学习含义                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| 价值向量       | $\boldsymbol{v} = [v(s_1), \ldots, v(s_n)]^\top$                                                                   | 把所有状态价值放在一起               |
| 状态转移矩阵   | $P_{ij}=p(s_j \mid s_i)$                                                                                           | 第 $i$ 个状态转到第 $j$ 个状态的概率 |
| 贝尔曼矩阵形式 | $\boldsymbol{v}=\boldsymbol{r}+\gamma P\boldsymbol{v}$                                                             | 当前价值等于即时奖励加未来价值       |
| 线性方程解     | $\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}$                                                                   | 已知模型时求策略价值                 |
| Bellman 压缩   | $\|\mathcal{T}\boldsymbol{u}-\mathcal{T}\boldsymbol{v}\|_\infty\leq\gamma\|\boldsymbol{u}-\boldsymbol{v}\|_\infty$ | 价值迭代收敛保证                     |
| 点积           | $\boldsymbol{w}^\top\boldsymbol{x}=\sum_i w_ix_i$                                                                  | 线性价值函数、神经网络层             |
| 线性 Q 近似    | $Q(s,a)=\boldsymbol{w}^\top\phi(s,a)$                                                                              | 用特征向量估计动作价值               |
| 特征值         | $A\boldsymbol{u}=\lambda\boldsymbol{u}$                                                                            | 分析矩阵在不同方向上的放缩           |
| L2 范数        | $\|\boldsymbol{x}\|_2=\sqrt{\sum_i x_i^2}$                                                                         | 梯度裁剪、参数正则化                 |
| 信任域约束     | $(\theta-\theta_{old})^\top F(\theta-\theta_{old})\leq\delta$                                                      | TRPO/PPO 背后的安全更新思想          |

## 阶段小结：先掌握这些

到这里，你已经能把单个状态的价值推广到价值向量，把单步转移推广到转移矩阵，把一条贝尔曼方程推广到整个方程组。下面进入更完整的公式层：收敛性、Q 值近似和信任域约束。

## 进阶公式：从矩阵贝尔曼方程到收敛性

前面我们已经看过两状态方程：

$$
\boldsymbol{v}=\boldsymbol{r}+\gamma P\boldsymbol{v}.
$$

现在把它推广到任意有限状态空间。设共有 $n$ 个状态，固定策略 $\pi$ 后：

- $\boldsymbol{v}_\pi \in \mathbb{R}^n$ 是状态价值向量。
- $\boldsymbol{r}_\pi \in \mathbb{R}^n$ 是每个状态在策略 $\pi$ 下的期望即时奖励。
- $P_\pi \in \mathbb{R}^{n\times n}$ 是策略诱导的状态转移矩阵。

完整的策略评估方程是：

$$
\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma P_\pi \boldsymbol{v}_\pi.
$$

移项后得到：

$$
(I-\gamma P_\pi)\boldsymbol{v}_\pi=\boldsymbol{r}_\pi.
$$

如果 $I-\gamma P_\pi$ 可逆，就有：

$$
\boldsymbol{v}_\pi=(I-\gamma P_\pi)^{-1}\boldsymbol{r}_\pi.
$$

这个式子是动态规划里“已知模型、已知策略时求价值”的封闭形式。前面的两状态例子只是它的 $n=2$ 特例。

为什么 $\gamma<1$ 时通常可以放心？因为 $P_\pi$ 是转移概率矩阵，每行概率和为 $1$。它的谱半径满足：

$$
\rho(P_\pi) \le 1.
$$

乘上折扣因子后：

$$
\rho(\gamma P_\pi) \le \gamma < 1.
$$

这意味着不断应用贝尔曼更新：

$$
\boldsymbol{v}_{k+1}=\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_k
$$

会逐步收敛到唯一的不动点 $\boldsymbol{v}_\pi$。这就是“价值迭代为什么能稳定下来”的线性代数解释。

## 进阶公式：Q 值线性近似

在大状态空间中，不能给每个状态动作对都存一张表。一个简单办法是用特征向量近似：

$$
Q(s,a)=\boldsymbol{w}^\top\phi(s,a).
$$

这和前面的点积例子完全一样：

- $\phi(s,a)$ 是状态动作的特征向量。
- $\boldsymbol{w}$ 是需要学习的权重。
- 点积结果就是当前对 $Q(s,a)$ 的估计。

例如：

$$
\phi(s,a)=
\begin{bmatrix}
1 \\
0.5 \\
2
\end{bmatrix},
\qquad
\boldsymbol{w}=\begin{bmatrix}
0.2 \\
1.0 \\
-0.1
\end{bmatrix},
$$

则：

$$
Q(s,a)=0.2\times1+1.0\times0.5-0.1\times2=0.5.
$$

深度 Q 网络只是把这个线性近似推广成神经网络：

$$
Q(s,a;\theta) \approx Q^*(s,a).
$$

线性代数仍然在内部发挥作用，因为神经网络每一层都包含矩阵乘法。

## 进阶公式：信任域里的二次型

PPO 的前身 TRPO 用一个二次型限制参数变化：

$$
(\theta-\theta_{old})^\top F(\theta-\theta_{old})\le \delta.
$$

先把它看成二维例子。令：

$$
F=
\begin{bmatrix}
4 & 0 \\
0 & 1
\end{bmatrix},
\qquad
\theta-\theta_{old}=
\begin{bmatrix}
0.2 \\
0.3
\end{bmatrix}.
$$

则：

$$
(\theta-\theta_{old})^\top F(\theta-\theta_{old})
=4\times0.2^2+1\times0.3^2=0.25.
$$

如果 $\delta=0.3$，这次更新在安全范围内；如果 $\delta=0.1$，这次更新太大。

这里的 $F$ 可以理解成一种“带权距离”。普通 L2 距离只看参数变了多少，$F$ 加权后的距离还考虑“哪些方向更敏感”。这就是自然梯度和信任域方法背后的几何直觉。

## 最终小结

本页的层次是：先从价值向量、转移矩阵和两状态贝尔曼方程开始，再推广到 $\boldsymbol{v}_\pi=(I-\gamma P_\pi)^{-1}\boldsymbol{r}_\pi$、$Q(s,a)=\boldsymbol{w}^\top\phi(s,a)$ 和 TRPO 的二次型约束。读复杂公式时，先找出其中的向量、矩阵和点积，再把它们还原成“很多状态一起算”或“参数空间里的方向和距离”。

## 常见误区

1. **把矩阵当成抽象符号。** 在贝尔曼方程里，转移矩阵 $P$ 的每一行都是“从当前状态出发，下一步去哪里”的概率表。
2. **以为求逆就是实际算法。** $\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}$ 是理论闭式解，真实大规模问题通常用迭代或函数近似。
3. **忽略向量方向。** 梯度、优势更新、信任域约束都不仅关心“数值大小”，还关心参数空间中的方向。

## 小练习

1. 若 $\gamma=0.9$，$v_1=1+0.9v_2$，$v_2=2+0.9v_1$，试手算 $v_1,v_2$。
2. 写出上题对应的 $\boldsymbol{r}$ 和 $P$。
3. 若梯度为 $[6,8]^\top$，最大范数设为 $5$，裁剪后的梯度是多少？
