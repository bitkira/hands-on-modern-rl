# E.1.2 贝尔曼方程的矩阵形式

## 第 03 讲：从手算贝尔曼方程到矩阵形式

沿用导读页中的两状态例子：

- 在 $s_1$，奖励是 $2$，下一步到 $s_2$。
- 在 $s_2$，奖励是 $1$，下一步到 $s_1$。
- 折扣因子 $\gamma = 0.5$。

手写贝尔曼方程是：

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

如果把价值写成向量：

$$
\boldsymbol{v} =
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix},
\qquad
\boldsymbol{r} =
\begin{bmatrix}
2 \\
1
\end{bmatrix},
\qquad
P =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix},
$$

两个方程可以合并成：

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}.
$$

验证一下右边第一行：

$$
2 + 0.5 \times (0v_1 + 1v_2) = 2 + 0.5v_2.
$$

右边第二行：

$$
1 + 0.5 \times (1v_1 + 0v_2) = 1 + 0.5v_1.
$$

所以矩阵形式并没有引入新魔法，只是把很多相似的方程压缩成一个式子。

## 第 04 讲：解线性方程组

从

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}
$$

开始，把含有 $\boldsymbol{v}$ 的项移到左边：

$$
\boldsymbol{v} - \gamma P\boldsymbol{v} = \boldsymbol{r}.
$$

提取公共项：

$$
(I - \gamma P)\boldsymbol{v} = \boldsymbol{r}.
$$

如果矩阵 $I - \gamma P$ 可逆，就可以写成：

$$
\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}.
$$

带入具体数字：

$$
I - \gamma P =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
-0.5
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}.
$$

于是我们要解：

$$
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}.
$$

这和手算方程完全等价，解仍然是：

$$
v_1 = 3.33, \qquad v_2 = 2.67.
$$

实际算法中，状态很多时不一定真的求逆，因为矩阵可能巨大。值迭代、策略评估和时序差分方法，本质上都是在用更可扩展的方法逼近这个解。
