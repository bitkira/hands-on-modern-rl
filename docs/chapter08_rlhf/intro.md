# 第 8 章：复刻 InstructGPT——从预训练基座到 RLHF 对齐助手

上一章我们把 PPO 的核心机制讲清楚了：策略不能一步走太远，所以要用裁剪、优势估计和 KL 约束来稳定更新。现在我们把这套算法搬到大语言模型上，复刻 OpenAI InstructGPT 风格的经典 RLHF 流水线。

先澄清一个边界：**RLHF 不包括从零预训练**。预训练模型是 RLHF 的起点，不是 RLHF 本身。本章会从一个已经公开发布的 base model 开始，例如 `HuggingFaceTB/SmolLM2-360M`、`Qwen/Qwen2.5-0.5B` 或 `EleutherAI/pythia-410m`。它们已经学会了语言建模，但还没有被稳定地训练成助手。我们的任务是把它们一步步后训练成更有用、更符合偏好的模型。

本章的方法论参考 OpenAI 的 InstructGPT：先用监督微调（SFT）教模型按指令回答，再用偏好数据训练奖励模型（Reward Model, RM），最后用 PPO 按奖励模型给出的信号优化策略。小参数实验用 Hugging Face TRL 跑通，大参数扩展则参考 OpenRLHF、NVIDIA NeMo RL / NeMo Aligner 这类工程框架。

```mermaid
flowchart LR
    PT["公开预训练基座\nbase LM"] --> SFT["SFT\n教它怎么回答"]
    SFT --> RM["Reward Model\n教裁判什么是好回答"]
    SFT --> PPO["PPO-RLHF\n按奖励练习"]
    RM --> PPO
    SFT -.->|"冻结为 Reference"| PPO
    PPO --> Eval["评估与回归测试\n确认真的变好"]

    style PT fill:#e3f2fd,stroke:#1565c0
    style SFT fill:#fff3e0,stroke:#e65100
    style RM fill:#fff3e0,stroke:#e65100
    style PPO fill:#e8f5e9,stroke:#2e7d32
    style Eval fill:#f3e5f5,stroke:#6a1b9a
```

## 为什么不用从零预训练

从零预训练一个几百 M 参数模型当然可以做教学项目，但它不是经典 RLHF 的重点。真实的大厂流程通常不是“先造一个小 LLM 再对齐”，而是从一个已经训练好的 base model 或 checkpoint 出发，围绕数据、偏好、奖励和稳定性做后训练。

因此，本章会把“预训练”处理成一个输入 artifact：我们先加载 base model，观察它的原始行为，再进入 SFT、RM、PPO 三个阶段。如果你想理解从零预训练，可以把它看作 RLHF 之前的独立工程；如果你想理解现代后训练，本章才是主线。

## 本章路线图

| 小节                                                            | 核心问题                                               | 产物                             |
| --------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------- |
| [为什么 base model 还不是 assistant](./base-model-to-assistant) | pretrained checkpoint 能做什么、缺什么？               | base / SFT / RLHF 三阶段对照     |
| [标准 RLHF 流水线](./standard-rlhf-pipeline)                    | InstructGPT 风格 SFT → RM → PPO 的输入输出是什么？     | 标准流程图与 artifact 清单       |
| [SFT：教模型按指令回答](./imitation-learning-pipeline)          | SFT 数据和偏好数据怎么构造？                           | SFT 训练数据与 SFT 模型          |
| [Reward Model：教一个裁判](./reward-function-design)            | 如何把 chosen/rejected 偏好对变成可优化的标量奖励？    | Reward Model 与 RM 验证指标      |
| [PPO-RLHF：按奖励练习](./ppo-rlhf-loop)                         | Actor、Reference、Reward Model、Critic 如何协作？      | PPO-RLHF 四模型结构              |
| [评估：RLHF 到底有没有变好](./evaluation)                       | 如何证明模型真的变好，而且没有掉点或 reward hacking？  | 自动评测、偏好评测、人工抽检方案 |
| [从小参数到大参数](./scaling-to-large-models)                   | TRL 小实验如何迁移到 OpenRLHF / NeMo RL 的大参数训练？ | 小模型和生产工程的映射表         |

## 8.6 为什么必须做评估

RLHF 最容易制造一种错觉：训练日志很好看，模型实际变差了。RM 分数上涨可能只是模型学会了讨好 RM；回答变长可能被误判为更有帮助；偏好数据上的胜率提高，也可能伴随数学、代码、事实性能力下降。

所以本章的评估不是可选项，而是 RLHF 流水线的一部分。8.6 会采用三层评估：

- **自动 benchmark**：用固定任务集检查通用能力和专项能力有没有掉点，例如指令遵循、简单推理、事实问答、格式遵循。
- **偏好评估**：让 base / SFT / RLHF 的回答做 pairwise battle，用人类或强模型 judge 判断哪个更好。
- **人工抽检**：小规模但高质量地看样本，特别关注 reward hacking、长度膨胀、空洞模板、安全退化和事实幻觉。

一个合格的 RLHF 实验，不能只报告 “reward 变高”。它至少要回答三个问题：模型是不是更符合人类偏好？原有能力有没有明显回退？高分回答是不是真的可用？

## 与第 9 章的边界

本章讲的是**经典 RLHF 标准流程**。第 9 章会从这套流程出发，解释现代 post-training 为什么要一步步简化它：

- DPO 试图省掉显式 Reward Model。
- GRPO 试图省掉 Critic。
- RLVR 把主观偏好换成可验证奖励。
- DAPO、RLAIF、蒸馏和数据飞轮则进一步改变训练信号的来源和扩展方式。

换句话说，08 是“标准答案”，09 是“现代演进”。先把标准 RLHF 跑通，再理解为什么大家要改造它，整个大模型强化学习的脉络才会顺。

准备好后，我们从第一步开始：先看为什么 pretrained base model 还不是 assistant——[为什么 base model 还不是 assistant](./base-model-to-assistant)。
