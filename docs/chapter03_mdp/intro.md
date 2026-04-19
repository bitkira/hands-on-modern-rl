# Chapter 3. Foundations of RL: MDP, Value Functions & Policy Optimization

在前两章中，我们做了两件事：第 1 章让一根木棍在小车上学会了直立不倒，第 2 章让一个小模型学会了在被拒绝时保持礼貌。两次都做成了——CartPole 拿到了高分，DPO 产出了更友好的回复。

但有一个问题一直被悄悄绕过了：**这些算法到底是怎么"学会"的？**

PPO 为什么能在 20000 步之后突然"开窍"？DPO 的损失函数为什么能把偏好信号注入模型参数？要回答这些问题，我们需要一套数学语言来精确描述"学习"的过程。这套语言就是马尔可夫决策过程（Markov Decision Process, MDP）及其衍生出的价值函数、贝尔曼方程和策略优化理论。

## Core Questions

所有 RL 算法都在回答同一个问题：**How to select actions to maximize cumulative reward?** 围绕这个问题，本章将沿着一条问题驱动的路径展开：

- 怎么量化策略的好坏？→ 期望回报
- 怎么定义游戏规则？→ MDP 五元组
- 怎么评估当前局面？→ $V(s)$
- $V$ 怎么算？→ 贝尔曼方程 → DP / MC / TD
- $V$ 的局限：不告诉你选哪个动作 → 两条路线
  - 路线一：$Q(s,a)$——给动作打分，选最高的
  - 路线二：$J(\theta)$——跳过打分，直接优化策略

## Lecture Guide

| Section | Guiding Question |
| --- | --- |
| [3.1 Multi-Armed Bandits: Exploration vs Exploitation](./bandit) | What is the core tension in RL decision-making? How does expected return quantify policy quality? |
| [3.2 Markov Decision Processes](./mdp) | How do we formally define "the rules of the game" and "a policy"? |
| [3.3 Value Functions & Bellman Equations](./value-bellman) | How do we compute "how good is this state" using only a one-step lookahead? |
| [3.4 Dynamic Programming, Monte Carlo & Temporal Difference](./dp-mc-td) | What problems does each of the three methods for computing $V$ solve? |
| [3.5 Value-Based: Action-Value Q(s, a)](./value-q) | How do we score every action and pick the best? |
| [3.6 Policy-Based: Objective J(θ)](./policy-objective) | How do we bypass scoring and directly optimize the policy? |
| [3.7 Reward Shaping & Design](./reward-design) | Where does the optimization objective come from? |
| [3.8 Landscape: Taxonomy & Roadmap](./panorama) | What dilemmas do the two routes face? Where do we go from here? |

下一节将从两台老虎机开始，亲身体验 RL 最核心的决策困境。[3.1 Multi-Armed Bandits](./bandit)
