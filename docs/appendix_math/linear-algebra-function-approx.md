# E.1.3 函数近似、点积与范数

## 第 05 讲：点积为什么重要

假设一个状态有三个特征：

$$
\boldsymbol{x}(s) =
\begin{bmatrix}
1 \\
0.5 \\
2
\end{bmatrix}.
$$

一个线性价值函数用三个权重表示：

$$
\boldsymbol{w} =
\begin{bmatrix}
0.2 \\
1.0 \\
-0.1
\end{bmatrix}.
$$

那么状态价值可以用点积计算：

$$
\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)
= 0.2 \times 1 + 1.0 \times 0.5 + (-0.1) \times 2 = 0.5.
$$

点积的直觉是：**每个特征贡献多少，由对应权重决定，最后加总**。神经网络的线性层也是这个思想的扩展。

在深度强化学习中，状态可能是一张图像、一段文本或一组传感器数值。模型先把它变成向量，再通过矩阵乘法和非线性函数得到价值或动作概率。

## 第 06 讲：范数衡量“有多大”

范数可以理解成向量的长度。给定向量：

$$
\boldsymbol{g} =
\begin{bmatrix}
3 \\
4
\end{bmatrix},
$$

它的 L2 范数是：

$$
\|\boldsymbol{g}\|_2 = \sqrt{3^2 + 4^2} = 5.
$$

在训练神经网络时，$\boldsymbol{g}$ 常常表示梯度。如果梯度范数过大，参数更新会非常剧烈，训练可能不稳定。梯度裁剪就是限制这个长度。

例如如果最大范数设为 $2$，而当前梯度范数是 $5$，就把梯度缩小为原来的 $2/5$：

$$
\boldsymbol{g}_{clipped} = \frac{2}{5}
\begin{bmatrix}
3 \\
4
\end{bmatrix}
=
\begin{bmatrix}
1.2 \\
1.6
\end{bmatrix}.
$$

新梯度的范数正好是 $2$。这就是很多 RL 代码中 `max_grad_norm` 的数学含义。
