# E.3.6 微积分与优化公式速查与练习

## 本书中你会遇到的优化公式

| 概念             | 公式                                                                                   | 强化学习含义                   |
| ---------------- | -------------------------------------------------------------------------------------- | ------------------------------ |
| 梯度上升         | $\theta \leftarrow \theta + \alpha \nabla J(\theta)$                                   | 最大化平均回报                 |
| 梯度下降         | $\theta \leftarrow \theta - \alpha \nabla L(\theta)$                                   | 最小化价值误差或模型损失       |
| 链式法则         | $\frac{dL}{d\theta}=\frac{dL}{dy}\frac{dy}{d\theta}$                                   | 反向传播的基础                 |
| 策略梯度         | $\nabla J \approx G_t\nabla\log\pi_\theta(a_t\mid s_t)$                                | 好结果对应动作概率上升         |
| 策略梯度定理     | $\nabla_\theta J=\sum_s d^\pi(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a)$      | 参数变化如何影响平均回报       |
| 对数导数技巧     | $\nabla_\theta\pi=\pi\nabla_\theta\log\pi$                                             | 把难算的概率梯度改成易采样形式 |
| 优势加权更新     | $\nabla J \approx \hat{A}_t\nabla\log\pi_\theta(a_t\mid s_t)$                          | 相对平均更好的动作被加强       |
| PPO 概率比       | $r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}$                           | 衡量新旧策略变化               |
| 二阶 Taylor 展开 | $f(\theta+\Delta)\approx f(\theta)+\nabla f^\top\Delta+\frac{1}{2}\Delta^\top H\Delta$ | 理解曲率和信任域               |
| PPO 裁剪项       | $\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)$                 | 限制策略更新过大               |
| GRPO 组优势      | $\hat{A}_i=\frac{r_i-\mu}{\sigma}$                                                     | 用组内相对奖励替代 Critic      |

## 阶段小结：先掌握这些

到这里，你已经能理解导数、梯度、链式法则、优势函数和 PPO 裁剪的直觉。下面进入更完整的公式层：策略梯度定理、DQN 损失、GAE、PPO 目标和 GRPO 归一化优势。

## 最终小结

本页的层次是：先从一维导数和一次参数更新开始，再推广到策略梯度定理、DQN 的 TD 损失、GAE、PPO 裁剪目标和 GRPO 组归一化。读复杂优化公式时，先找出三件事：目标函数是什么、梯度方向来自哪里、更新幅度如何被控制。

## 常见误区

1. **把梯度当成最终答案。** 梯度只是方向，真正更新还要乘以学习率，并可能经过裁剪、归一化或 Adam 调整。
2. **以为回报越大更新越安全。** 高回报样本可能导致过大更新，所以 PPO 才需要概率比裁剪。
3. **忽略 Critic 的误差。** Actor 的优势估计依赖 Critic；Critic 不准时，策略更新方向也会受影响。

## 小练习

1. 对 $J(\theta)=-(\theta-1)^2+2$ 求导，并从 $\theta=0$、学习率 $0.1$ 做一步梯度上升。
2. 旧策略概率为 $0.4$，新策略概率为 $0.6$，优势为 $3$，未裁剪 PPO 项是多少？若 $\epsilon=0.2$，裁剪后是多少？
3. 若 $V(s_t)=5$，$R_{t+1}=2$，$V(s_{t+1})=6$，$\gamma=0.9$，一步 TD 误差是多少？
