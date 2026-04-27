# E.1.4 收敛性、谱半径与信任域

## 第 07 讲：特征值与价值迭代为什么会收敛

现在进入稍微抽象一点、但非常关键的一步：为什么贝尔曼更新反复做下去会稳定？

先看一个纯数字变换：

$$
x_{k+1}=0.5x_k.
$$

如果 $x_0=8$，那么：

$$
8 \rightarrow 4 \rightarrow 2 \rightarrow 1 \rightarrow 0.5 \rightarrow \cdots
$$

因为每一步都乘以 $0.5$，误差会越来越小。换成向量时，矩阵也会在某些方向上“放大”或“缩小”误差。特征值就是描述这种放缩比例的工具。

如果存在非零向量 $\boldsymbol{u}$ 和标量 $\lambda$，使得：

$$
A\boldsymbol{u}=\lambda \boldsymbol{u},
$$

那么 $\boldsymbol{u}$ 是矩阵 $A$ 的特征向量，$\lambda$ 是对应特征值。意思是：矩阵 $A$ 作用在 $\boldsymbol{u}$ 上时，没有改变方向，只把长度缩放了 $\lambda$ 倍。

回到贝尔曼更新：

$$
\boldsymbol{v}_{k+1}=\boldsymbol{r}+\gamma P\boldsymbol{v}_k.
$$

假设真实解是 $\boldsymbol{v}^\*$，它满足：

$$
\boldsymbol{v}^\*=\boldsymbol{r}+\gamma P\boldsymbol{v}^\*.
$$

两式相减，得到误差递推：

$$
\boldsymbol{v}_{k+1}-\boldsymbol{v}^\*
=\gamma P(\boldsymbol{v}_k-\boldsymbol{v}^\*).
$$

令误差 $\boldsymbol{e}_k=\boldsymbol{v}_k-\boldsymbol{v}^\*$，则：

$$
\boldsymbol{e}_{k+1}=\gamma P\boldsymbol{e}_k.
$$

这和一开始的一维例子 $x_{k+1}=0.5x_k$ 是同一种结构。只不过这里的“乘以 $0.5$”变成了“乘以矩阵 $\gamma P$”。当 $0<\gamma<1$ 且 $P$ 是转移概率矩阵时，这个变换会压缩误差，所以价值迭代会收敛。

更正式地说，贝尔曼算子：

$$
\mathcal{T}\boldsymbol{v}=\boldsymbol{r}+\gamma P\boldsymbol{v}
$$

是一个压缩映射。对任意两个价值向量 $\boldsymbol{u},\boldsymbol{v}$，有：

$$
\|\mathcal{T}\boldsymbol{u}-\mathcal{T}\boldsymbol{v}\|_\infty
\leq \gamma \|\boldsymbol{u}-\boldsymbol{v}\|_\infty.
$$

这条公式看起来复杂，但它只是在说：**做一次贝尔曼更新后，两个价值估计之间的差距最多变成原来的 $\gamma$ 倍**。由于 $\gamma<1$，差距会越来越小。

## 第 08 讲：从普通长度到信任域椭球

前面说的 L2 范数是普通长度：

$$
\|\boldsymbol{x}\|_2^2=\boldsymbol{x}^\top \boldsymbol{x}.
$$

例如：

$$
\boldsymbol{x}=
\begin{bmatrix}
3 \\
4
\end{bmatrix},
\qquad
\boldsymbol{x}^\top\boldsymbol{x}=3^2+4^2=25.
$$

但有时不同方向的“风险”不一样。比如参数空间里，向右走一点可能让策略变化很小，向上走一点却让策略变化很大。这时我们不用普通长度，而用一个矩阵 $F$ 定义加权长度：

$$
\|\boldsymbol{x}\|_F^2=\boldsymbol{x}^\top F\boldsymbol{x}.
$$

看一个具体数字：

$$
F=
\begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
\qquad
\boldsymbol{x}=
\begin{bmatrix}
1 \\
1
\end{bmatrix}.
$$

那么：

$$
\boldsymbol{x}^\top F\boldsymbol{x}
=
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
=5.
$$

第二个方向被权重 $4$ 放大了，所以同样走 $1$ 步，它更“贵”。

TRPO 的信任域约束就长这样：

$$
(\theta-\theta_{old})^\top F(\theta-\theta_{old})\leq \delta.
$$

其中 $F$ 通常是 Fisher 信息矩阵。直觉是：**参数更新不能只看欧氏距离，还要看这一步会让策略分布变化多大**。如果某个方向会让策略急剧变化，$F$ 会把这个方向的步长压小。

所以从普通范数到信任域约束，其实是一条自然升级路径：

$$
\|\Delta\theta\|_2^2
=\Delta\theta^\top \Delta\theta
\quad\Longrightarrow\quad
\Delta\theta^\top F\Delta\theta\leq\delta.
$$
