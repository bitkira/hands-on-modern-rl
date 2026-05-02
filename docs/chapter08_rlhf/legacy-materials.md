# 8.8 旧稿补充与实战材料

这页专门保留本轮重构时不适合压进 8.1-8.7 主线的旧稿。新主线负责讲清楚 RLHF 的标准流程，旧稿补充页负责保存更长的展开、实验代码和工程排查清单，避免内容在重写时丢失。

## 被恢复的旧稿

| 旧稿                 | 现在位置                                                   | 适合什么时候读                                                                       |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 训练稳定性与奖励黑客 | [training-stability-hacking](./training-stability-hacking) | 读完 8.5 PPO-RLHF 后，用来补 KL、warmup、梯度裁剪、奖励归一化和失败模式排查          |
| RLAIF 与自我博弈     | [rlaif-self-play](./rlaif-self-play)                       | 读完 8.2 标准流水线后，用来补 AI Judge、Constitutional AI、自我博弈和自我进化循环    |
| 自我博弈与数据飞轮   | [rlaif-and-data-cycle](./rlaif-and-data-cycle)             | 想看 RLAIF、数据飞轮、主动学习和护栏式自进化的合并版说明时阅读                       |
| 数据循环体系         | [data-cycle](./data-cycle)                                 | 想把 RLHF 看成持续迭代的数据系统时阅读，重点是 badcase、数据质量、合成策略和实践案例 |
| Reward Hacking 实战  | [reward-hacking-hands-on](./reward-hacking-hands-on)       | 读完 8.6 评估后，亲手制造一个被长度奖励 hack 的实验，再看如何修复                    |

## 和新主线的关系

8.1-8.7 是重新组织后的学习主线：base model 为什么不是 assistant、SFT、Reward Model、PPO-RLHF、评估和工程放大。上面这些旧稿不是废弃内容，而是补充材料：

- RLAIF、CAI、Self-Play 和数据飞轮已经在 [8.2 标准 RLHF 流水线](./standard-rlhf-pipeline) 中保留了压缩版。
- KL 约束、训练稳定性和失败模式已经在 [8.5 PPO-RLHF](./ppo-rlhf-loop) 中保留了压缩版。
- Reward hacking 的检测框架已经在 [8.6 评估](./evaluation) 中保留了压缩版。

如果读者按课程第一次学习，先走 8.1-8.7；如果要做项目、查细节或复用旧实验，再回到这里查完整材料。
