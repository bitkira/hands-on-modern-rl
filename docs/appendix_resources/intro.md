# 附录 F：强化学习学习资源推荐

> **本附录目标**：本书覆盖了从 MDP 基础到 PPO、DPO、GRPO 的完整链路，但 RL 领域远不止于此。如果你想深入某个方向、对比不同教学风格、或者寻找上手实践的资源，这份清单可以作为你的起点。所有资源均免费或公开可访问。

## 如何使用这份清单？

按你的目标选择：

| 你的情况 | 建议从哪里开始 |
| --- | --- |
| 刚学完第 3 章，想看其他教材怎么讲基础理论 | 赵世钰《RL 数学原理》或 Sutton & Barto 原版 |
| 想看视频课程跟学 | David Silver（经典）或李宏毅（中文） |
| 想上手写代码 | OpenAI Spinning Up 或《动手学强化学习》 |
| 关注 LLM 对齐 / RLHF / GRPO | Nathan Lambert RLHF Book 或 Ernest Ryu RL-LLM 课程 |
| 想了解前沿理论 | Princeton ECE 524 或 Alberta CMPUT 609 |

---

## 一、经典教材

### Reinforcement Learning: An Introduction（Sutton & Barto，第 2 版，2018）

**地址**：[incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) ｜ [中文翻译](https://rl.qiwihui.com/)

强化学习领域的标准教材，被几乎所有大学 RL 课程列为必读。全书分三部分：Part I（表格方法，Ch1-8）覆盖 MDP、DP、MC、TD、n-step Bootstrapping、Planning；Part II（近似方法，Ch9-13）覆盖函数近似、资格迹、策略梯度；Part III（Ch14-17）讨论心理学、神经科学和应用。免费 PDF，中文翻译质量较高。**适合系统性地打基础。**

### 强化学习的数学原理（赵世钰）

**地址**：[github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)（GitHub 10k+ stars）

Springer 出版 + 清华大学出版社。全书 10 章从数学角度严格推导 RL 核心算法：贝尔曼方程 → VI/PI → MC → TD（含 Sarsa、Q-Learning、n-step Sarsa）→ 函数近似 → 策略梯度 → Actor-Critic。每章配有数学证明和练习题。**适合喜欢严格推导、想从数学层面理解"为什么这些算法有效"的读者。**

### 深度强化学习（张志华，北京大学）

**地址**：[PDF 初稿](https://www.math.pku.edu.cn/teachers/zhzhang/drl_v1.pdf)

北京大学数学学院课程配套教材。假设读者了解机器学习基础但不一定熟悉 RL，从基本概念出发，覆盖价值学习（DQN）、策略学习（Policy Gradient）、Actor-Critic、TRPO 等。配套王树森的 B 站视频课程。**适合中文读者快速入门 DRL。**

### 动手学强化学习（张伟楠、沈键、俞勇）

**地址**：[在线版](https://hrl.boyuai.com/) ｜ 上海交大 RL 课程教材

实践导向，全书配有可运行的 Jupyter 代码。三部分：基础（Bandit → MDP → DP → MC → 规划）→ 进阶（函数近似 → DQN → 策略梯度 → PPO）→ 前沿（Model-Based RL、Offline RL）。**适合边读边敲代码的学习者。**

---

## 二、大学课程

### 欧美课程

#### Stanford CS234: Reinforcement Learning（Emma Brunskill）

**地址**：[web.stanford.edu/class/cs234/](https://web.stanford.edu/class/cs234/)

斯坦福 RL 基础课。从表格 MDP 开始，覆盖策略评估、Q-Learning、函数近似、策略梯度、Offline RL、探索、MCTS，最后涉及 RLHF。约一半课时打理论基础，一半进入高级主题。教材：Sutton & Barto。

#### Stanford CS224R: Deep Reinforcement Learning（Chelsea Finn）

**地址**：[cs224r.stanford.edu](https://cs224r.stanford.edu/) ｜ [YouTube 2025](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL)

斯坦福深度 RL 课程。假设学生已有 RL 基础，直接从模仿学习开始，快速进入策略梯度、Actor-Critic、Q-Learning、Model-Based RL、Offline RL、Reward Learning、RLHF、Meta-RL。**适合已掌握基础、想深入 DRL 各方向的学习者。**

#### MIT 6.7920: Reinforcement Learning Foundations and Methods（Cathy Wu）

**地址**：[web.mit.edu/6.7920/www/](https://web.mit.edu/6.7920/www/)

MIT 的 RL 理论课。三分之二"exploitation"（已知理论：DP 7 讲 + RL 核心方法 9 讲），三分之一"exploration"（前沿专题）。DP 部分非常扎实，覆盖有限/无限视界、LQR、策略/价值迭代、收敛性证明。**适合追求理论深度的学习者。**

#### UC Berkeley CS285: Deep Reinforcement Learning（Sergey Levine）

**地址**：[rail.eecs.berkeley.edu/deeprlcourse/](https://rail.eecs.berkeley.edu/deeprlcourse/)

伯克利的 Deep RL 旗舰课。仅 1 讲回顾 RL 基础，随后深入模仿学习、策略梯度、Actor-Critic、Value-Based RL、高级策略梯度、变分推断与 RL、LLM RL、Model-Based RL、Offline RL、探索。2026 春季版新增了 LLM RL 和 Offline RL 的实战作业。**内容最贴近当前工业前沿。**

#### CMU 10-703: Deep Reinforcement Learning and Control

**地址**：[cmudeeprl.github.io/703website_f25/](https://cmudeeprl.github.io/703website_f25/)

CMU 的深度 RL 课程。覆盖经典理论（MDP、DP、MC、TD）后，进入函数近似、Deep Q-Learning、MCTS、策略梯度、模仿学习、逆 RL、最优控制、Model-Based RL、探索。**理论与实践并重，覆盖面广。**

#### University of Alberta CMPUT 365: Introduction to RL（Marlos Machado）

**地址**：[Syllabus PDF](https://webdocs.cs.ualberta.ca/~machado/cmput365/w26/syllabus.pdf)

Sutton 所在大学的 RL 入门课，严格遵循 Sutton & Barto 教材顺序：Bandits → MDP → DP（含 PI、VI、GPI）→ MC 预测与控制 → TD 预测 → **TD 控制（Sarsa、Q-Learning）** → Planning（Dyna-Q）→ 函数近似 → 策略梯度。**Sutton & Barto 教材最忠实的课程实现。**

#### Georgia Tech CS 7642: Reinforcement Learning（OMSCS）

**地址**：[omscs.gatech.edu/cs-7642-reinforcement-learning](https://omscs.gatech.edu/cs-7642-reinforcement-learning)

可在线修读的 RL 课程。覆盖 DP、TD（含 Sarsa）、n-step TD、Lambda Return、DQN、策略梯度、多智能体 RL、博弈论、POMDP。**OMSCS 项目中口碑最好的 RL 课程之一。**

#### Princeton ECE 524: Foundations of RL（Chi Jin）

**地址**：[sites.google.com/view/cjin/teaching/ece524](https://sites.google.com/view/cjin/teaching/ece524) ｜ [YouTube](https://www.youtube.com/playlist?list=PLYXvCE1En13epbogBmgafC_Yyyk9oQogl)

理论导向，侧重有限样本分析和收敛性证明。Part I 覆盖表格 MDP、规划、探索（Bandit 和 MDP）、下界；Part II 覆盖大状态空间、线性 VI、函数近似、多智能体、POMDP。**适合想做 RL 理论研究的学习者。**

#### David Silver RL Course（UCL / DeepMind）

**地址**：[davidsilver.uk/teaching](https://www.davidsilver.uk/teaching/) ｜ [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

10 讲经典课程：MDP → DP → Model-Free Prediction → Model-Free Control → 函数近似 → 策略梯度 → Learning & Planning → 探索 → 经典游戏案例。David Silver 是 AlphaGo/AlphaZero 的第一作者。**结构精炼，讲解清晰，是最广泛传播的 RL 视频课程。**

#### DeepMind x UCL RL Lecture Series（2021）

**地址**：[YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

David Silver 课程的更新版，由 DeepMind 研究员（Hado van Hasselt 等）主讲。13 讲覆盖探索与控制、MDP 与 DP、无模型方法、函数近似、Planning、策略梯度与 Actor-Critic、近似 DP、Multi-step 与 Off-Policy、Deep RL。**比 2015 版更深入，增加了前沿内容。**

### 中国大学课程

#### 清华大学 Reinforcement Learning（Fall 2025）

**地址**：[coai.cs.tsinghua.edu.cn/Courses/RL2025/_site/](https://coai.cs.tsinghua.edu.cn/Courses/RL2025/_site/)

本科生 RL 课程。从多臂老虎机开始，覆盖 MDP、Planning（DP）、MC、TD Learning、策略梯度、函数近似、Deep RL。4 个编程作业（Bandit → MDP → TD & PG → Deep RL）+ 课程项目。课件公开。

#### 南京大学 强化学习基础（俞扬，2024）

**地址**：[lamda.nju.edu.cn/introrl](https://www.lamda.nju.edu.cn/introrl/)

基于 Sutton & Barto 教材，9 讲覆盖 RL 基础概念、MDP、DP、MC、TD、DQN。5 个编程作业（Dagger → Q-Learning → DQN → Model-Based → Offline RL）。**中文大学课程中理论基础最扎实的之一。**

#### 南京大学 高级强化学习（袁雷，2025）

**地址**：[lamda.nju.edu.cn/advanceRL](https://www.lamda.nju.edu.cn/advanceRL/)

研究生进阶课程。覆盖 DDPG/TD3、PPO 及技巧、多智能体、RLHF/DPO 理论推导、论文阅读。

#### 上海交通大学 强化学习（张伟楠，2024）

**地址**：[wnzhang.net/teaching/sjtu-rl-2024](https://wnzhang.net/teaching/sjtu-rl-2024/)

使用《动手学强化学习》作为教材，9 章覆盖基础到前沿。

---

## 三、中文在线课程与教程

### 李宏毅 深度强化学习（台湾大学）

**地址**：[课程主页](https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) ｜ [B站 2025 版](https://www.bilibili.com/video/BV1SJvAzfEL2/)

以 Policy Gradient 为主线切入，深入讲解 PPO（含 Importance Sampling、On-policy → Off-policy 推导），然后讲 Q-Learning（DQN、Double DQN、Dueling DQN）、Actor-Critic。讲解生动，PPT 精美。**PPO 部分是中文课程中讲得最深入的。**

### 王树森 深度强化学习

**地址**：[B站视频](https://www.bilibili.com/video/BV1oEWDz1Ez5/) ｜ [知乎笔记](https://zhuanlan.zhihu.com/p/588047970)

北大数学学院课程配套视频。五大模块：基本概念 → 价值学习（DQN）→ 策略学习（Policy Gradient）→ Actor-Critic（A3C、TRPO）→ 进阶（DDPG、AlphaGo、多智能体）。配套张志华《深度强化学习》教材。**内容精炼，适合快速入门。**

### 蘑菇书 EasyRL（Datawhale）

**地址**：[在线版](https://datawhalechina.github.io/easy-rl/) ｜ [GitHub](https://github.com/datawhalechina/easy-rl)

综合了周博磊《强化学习纲要》、李宏毅课程、百度《世界冠军带你从零实践强化学习》的精华。13 章 + 专题，覆盖基础到 DQN、PPO、DDPG、AlphaStar。**中文社区最活跃的开源 RL 教程。**

### Spinning Up 中文版

**地址**：[spinningup.qiwihui.com](https://spinningup.qiwihui.com/)

OpenAI Spinning Up 的中文翻译。包含核心概念、算法分类、策略梯度推导，以及 VPG、TRPO、PPO、DDPG、TD3、SAC 六个算法的实现。

---

## 四、LLM 强化学习专项资源

### Nathan Lambert — RLHF Book + Course

**地址**：[rlhfbook.com](https://rlhfbook.com/) ｜ [Course](https://rlhfbook.com/course) ｜ [GitHub](https://github.com/natolambert/rlhf-book) ｜ [YouTube](https://www.youtube.com/playlist?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

AI2 研究员 Nathan Lambert 编写的 RLHF 专著。覆盖 RLHF 全流程：指令微调 → 奖励模型训练 → Rejection Sampling → PPO → DPO。代码库实现了 PPO、REINFORCE、GRPO、RLOO 等策略梯度方法。视频课程 4 讲。**LLM 对齐领域最系统的公开教材。**

### Ernest Ryu — Reinforcement Learning of Large Language Models（UCLA）

**地址**：[ernestryu.com/courses/RL-LLM.html](https://ernestryu.com/courses/RL-LLM.html)

唯一一门把经典 RL 理论和 LLM RL 系统结合的大学课程。三部分：Ch1（5 讲经典 RL：MDP → VI → PG → PPO/GRPO → AlphaGo）→ Ch2（4 讲 LLM 基础：NLP → Transformer → ICL/SFT）→ Ch3（2 讲 LLM RL：RLHF/PPO/DPO → RLVR）。**RL 基础讲得最深的 LLM RL 课程。**

### DeepLearning.AI — Reinforcement Fine-Tuning LLMs with GRPO

**地址**：[deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)

1 小时短课程，10 节课。以 Wordle 游戏为贯穿案例，讲解 GRPO 算法、奖励函数设计、LLM-as-Judge、Reward Hacking。7 个代码实验。**适合已有 LLM 基础、想快速上手 GRPO 的实践者。**

### Hugging Face — Deep RL Course

**地址**：[huggingface.co/learn/deep-rl-course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

8 个单元覆盖 Q-Learning → DQN → Policy Gradient → A2C/A3C → PPO → 多智能体 → Offline RL。每个单元配有理论和代码实践。Bonus 单元讲 RLHF。**适合想用 Hugging Face 生态做 RL 实验的学习者。**

---

## 五、实践工具与框架

### OpenAI Spinning Up in Deep RL

**地址**：[spinningup.openai.com](https://spinningup.openai.com/en/latest/)

RL 基础教学的金标准。三部分：核心概念（V/Q/Bellman/Advantage）→ 算法分类（Model-Based vs Model-Free）→ 策略优化推导（从零推导 Policy Gradient）。实现了 VPG、TRPO、PPO、DDPG、TD3、SAC 六个算法。**理论讲解和代码实现的最佳结合点。**

### Cameron Wolfe — Deep (Learning) Focus

**地址**：[PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm) ｜ [Online vs Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl)

系列博客，用通俗语言深入讲解 PPO 在 LLM 中的应用、Online vs Offline RL 的取舍、DPO 原理等。**适合想理解"为什么 LLM RL 用这些算法"的读者。**

### Sebastian Raschka — Ahead of AI

**地址**：[LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) ｜ [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)

《Build a Large Language Model From Scratch》作者的技术博客。覆盖 RLHF、DPO、RLVR、GRPO、推理时扩展等前沿话题。

