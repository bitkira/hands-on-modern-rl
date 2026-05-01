# 在线策略蒸馏（OPD）——把 Teacher 变成密集奖励

上一节我们讨论了 RL Scaling：给模型更多在线试错机会，能不能换来更强的推理能力？这一节看另一条越来越重要的路线：**不让小模型自己从零探索，而是让它在自己的轨迹上接受强模型的逐 token 指导**。

这就是在线策略蒸馏（On-Policy Distillation, OPD）。

它听起来像知识蒸馏，但更接近 RL。普通蒸馏让 student 模仿 teacher 写过的答案；OPD 让 student 自己先写，再让 teacher 在 student 真实走到的上下文里给反馈。

先不用急着理解“策略”“动作”“log-prob”这些术语。想象 student 正在一个词一个词地写答案。每写下一小段文字，它都要做一次选择：下一步写 “therefore”，还是写 “so”，还是写一个不该出现的词？在 RL 里，**负责做选择的东西叫策略（policy）**，所以 student 模型就是策略；**被选出来的下一小段文字叫 token**，它可能是一个字、一个词，或者词的一部分；**选择这个 token 的行为就是动作（action）**。

那 teacher 做什么？teacher 不一定重新写一份标准答案，而是看 student 已经写出的前文，再判断 student 刚选的这个 token 合不合理。如果 teacher 觉得“这个位置写这个词很像一个好答案”，分数就高；如果 teacher 觉得“这里写得很奇怪”，分数就低。论文里常用 teacher 对这个 token 的 **log-prob** 表示这个分数。你可以先把 log-prob 理解成：**老师心里觉得这个 token 有多合理**。

最后，“密集奖励”是什么意思？它不是等整篇答案写完才给一个总分，而是几乎每生成一个 token 都能给一次小反馈。GRPO/RLVR 常常像考试：最后答案对了给 1 分，错了给 0 分。OPD 更像老师坐在旁边批改草稿：学生每写一步，老师都能说“这一步像样”或“这一步不太对”。

为什么这件事最近重要？因为 LLM 后训练的核心瓶颈不是“有没有一个 loss”，而是**训练信号从哪里来、信号有多密、是不是落在 student 自己会遇到的状态上**。

| 方法        | 谁生成训练轨迹 | 反馈来自哪里         | 反馈粒度            | 主要问题                  |
| ----------- | -------------- | -------------------- | ------------------- | ------------------------- |
| SFT / SeqKD | 人类或 teacher | 标准答案 token       | token-level         | student 不练自己的错误    |
| PPO / GRPO  | student        | RM 或规则验证器      | 多为 sequence-level | 奖励稀疏，采样昂贵        |
| DPO         | 离线偏好数据   | chosen / rejected 对 | sequence-pair-level | 不能在线探索              |
| **OPD**     | **student**    | **teacher log-prob** | **token-level**     | 需要好 teacher 和好初始化 |

一句话概括：**OPD 想同时拿到 RL 的 on-policy 分布和蒸馏的密集监督。**

## Teacher 的角色怎么变了

OPD 容易被误解成“又一种 SFT”。真正关键的差别不是 student 有没有学 teacher，而是 **teacher 在训练里扮演什么角色**。

**方式一：抄笔记（SFT / SeqKD）。** 老师写了一份完整答案，学生逐字抄。这里 teacher 的角色是**答案作者**：它决定了回答长什么样，也决定了 student 要拟合哪些 token。好处是快、简单、稳定；坏处是 student 只在 teacher 写好的上下文里学习，主要学到的是“老师怎么写”。

**方式二：做老师做过的题（off-policy 蒸馏）。** 老师不只是写答案，还展示完整解题轨迹：如何拆题、如何推理、在哪里下结论。这里 teacher 的角色是**专家示范者**：student 学的不只是表面措辞，还包括 teacher 走过的推理路径。它比单纯抄笔记更有用，所以常被拿来做小模型冷启动。但问题仍然在：题是老师做过的，状态也是老师走到的，不是 student 自己卡住的地方。

**方式三：自己做题，老师逐题批改（on-policy 蒸馏）。** student 先自己生成回答，teacher 再看 student 真实写出来的前缀，逐 token 判断“这一步像不像一个好答案”。这里 teacher 的角色变成了**在线评价器**：它不一定重新写一份标准答案，而是在 student 自己访问到的状态上给反馈。

这就是 OPD 的核心区别：**teacher 从“数据生产者”变成了“奖励提供者”，student 从“模仿老师轨迹”变成了“在自己的轨迹上被老师纠偏”。** 这为什么重要？因为 student 推理时看到的是自己生成的上下文。如果训练时只见过 teacher 的上下文，一旦自己走偏，就会进入训练数据没有覆盖的区域。OPD 直接在这些 student 自己走到的区域给信号，缓解的正是这个分布偏移。

## 先把三个词分清

要理解 OPD，先把 **distillation**、**off-policy / offline** 和 **on-policy** 三个词分清。它们经常被混在一起，但指的是不同层面的事情。

**蒸馏（distillation）** 说的是 teacher-student 关系：一个强模型 $q$ 把能力迁移给一个较小或较便宜的模型 $\pi_\theta$。最朴素的蒸馏是让 teacher 生成答案，student 对这些答案做交叉熵训练：

$$
\mathcal{L}_{\text{hard KD}}
= -\sum_t \log \pi_\theta(y_t^T \mid x, y_{<t}^T)
$$

这里 student 看到的是 teacher 选出来的 token $y_t^T$。如果你能拿到 teacher 的完整概率分布，还可以做“软蒸馏”：不只告诉 student 正确 token 是什么，还告诉它其他 token 有多合理。

$$
\mathcal{L}_{\text{soft KD}}
= \sum_t D_{\text{KL}}\left(q(\cdot \mid c_t) \| \pi_\theta(\cdot \mid c_t)\right)
$$

软蒸馏的信息量更大。比如 teacher 可能认为 “therefore” 很好，“so” 也可以，“banana” 完全不行。硬标签只告诉你 teacher 最后选了哪个词，软标签告诉你 teacher 对整个动作空间的判断。

**policy（策略）** 在 LLM 里就是“给定上下文，输出下一个 token 分布”的模型：

$$
\pi_\theta(a_t \mid s_t)
\quad \Longleftrightarrow \quad
\pi_\theta(y_t \mid x, y_{<t})
$$

这里的状态 $s_t$ 是 prompt 加已生成前缀，动作 $a_t$ 是下一个 token。谁生成轨迹，谁就是行为策略（behavior policy）。这件事决定了训练数据的分布。

**Off-policy** 指的是：训练数据不是当前 student 自己生成的，而是来自另一个策略 $\mu$。这个 $\mu$ 可以是 teacher、旧 checkpoint、用户日志、历史模型，也可以是一批固定数据集。普通 teacher 蒸馏就是典型 off-policy：

$$
y \sim q(\cdot \mid x),
\quad
\text{update } \pi_\theta
$$

数据来自 teacher $q$，更新的是 student $\pi_\theta$。好处是便宜、可复用、稳定；坏处是 student 没有在自己的错误上下文里训练。

**Offline** 比 off-policy 更严格：不只是数据来自别的策略，而且训练时不再新采样，只使用一份固定数据集。SFT、DPO、离线偏好训练通常都是 offline。可以这样记：

| 概念       | 数据从哪里来            | 训练时还采新数据吗 | 例子                        |
| ---------- | ----------------------- | ------------------ | --------------------------- |
| offline    | 固定历史数据集          | 不采               | SFT、DPO、离线 SeqKD        |
| off-policy | 不是当前 student 的策略 | 可以采，也可以不采 | teacher 轨迹、旧模型 replay |
| on-policy  | 当前 student 自己生成   | 要采               | PPO、GRPO、OPD rollout      |

所以，**offline 在 LLM 后训练里通常也是 off-policy，但 off-policy 不一定 offline**。如果你每一轮都让 teacher 重新生成数据，再训练 student，它不是 offline，但仍然是 off-policy，因为行为策略还是 teacher，不是 student。

**On-policy** 指的是：用当前 student 自己生成的数据来更新当前 student。形式上是：

$$
y \sim \pi_\theta(\cdot \mid x),
\quad
\text{update } \pi_\theta
$$

它的优势是训练分布和推理分布一致。student 推理时会走到什么上下文，训练时就真的让它走过去，然后在那里给反馈。代价也很明显：每轮更新前都要重新 rollout，旧数据很快过期，样本效率低。

把这几个概念合起来看，OPD 的位置就很清楚了：

- 它是 **distillation**，因为反馈来自 teacher。
- 它是 **on-policy**，因为轨迹来自 student 自己。
- 它通常不是纯 **offline**，因为训练中要不断重新采 student rollout。
- 它和普通 off-policy 蒸馏的区别，不在于有没有 teacher，而在于 teacher 是在谁的轨迹上给信号。

## 为什么普通蒸馏不够

知识蒸馏（Knowledge Distillation, KD）的老问题是：大模型好用但贵，小模型便宜但弱。做法也很直接：让 teacher 生成数据，student 在这些数据上做监督学习。LLM 时代的 KD 综述通常把它分成几类：白盒蒸馏看 teacher logits，黑盒蒸馏只看 teacher 输出；也可以按能力分成推理、对齐、领域知识、工具使用等技能蒸馏。[^kd_survey_xu][^kd_survey_yang]

这条路非常有用。DeepSeek-R1 的蒸馏模型就是典型例子：先让强推理模型生成高质量轨迹，再把这些轨迹 SFT 到小模型上。对小模型来说，这往往比直接做 RL 更稳。

但它有一个根本缺口：**训练时 student 看到的是 teacher 的状态分布，推理时 student 走的是自己的状态分布。**

设 prompt 是 $x$，teacher 轨迹是：

$$y^{T} = (y_1^T, y_2^T, \dots, y_T^T) \sim q(\cdot \mid x)$$

普通蒸馏训练的是：

$$\mathcal{L}_{\text{off-policy}}(\theta) = -\sum_t \log \pi_\theta(y_t^T \mid x, y_{<t}^T)$$

这里的上下文 $x, y_{<t}^T$ 来自 teacher。可一旦 student 在第 3 步写错了，后面的上下文就变成了 $x, y_{<3}^{S}$。这个状态 teacher 可能从来不会走到，SFT 数据里也没有“从这里怎么救回来”的示范。错误会沿着自回归生成不断放大，这就是 exposure bias，也可以理解成模仿学习里的分布偏移。DAgger 早就指出过：要缓解这种问题，必须把 learner 自己访问到的状态也纳入训练。[^dagger]

OPD 就是把这个思想搬到 LLM 蒸馏里。

## OPD 到底在优化什么

OPD 的训练循环很短：

1. 给定 prompt $x$，student $\pi_\theta$ 自己采样回答 $y \sim \pi_\theta(\cdot \mid x)$。
2. 对 student 生成的每个前缀 $c_t=(x,y_{<t})$，teacher $q$ 计算当前 token $y_t$ 的概率。
3. 用 teacher 对 student 自己 token 的评价更新 student。

Google DeepMind 的 GKD 是这条线的代表性起点：它不再只依赖固定 teacher 输出，而是让 student 在自生成序列上接受 teacher 反馈，并允许 forward KL、reverse KL、JSD 等不同散度。[^gkd]

如果采用最常见的 reverse KL 形式，目标可以写成：

$$
\mathcal{L}_{\text{OPD}}(\theta)
= \mathbb{E}_{y \sim \pi_\theta}
\left[
\sum_t
\log \frac{\pi_\theta(y_t \mid c_t)}{q(y_t \mid c_t)}
\right]
$$

这就是在 student 自己访问到的状态上最小化：

$$D_{\text{KL}}(\pi_\theta(\cdot \mid c_t) \| q(\cdot \mid c_t))$$

展开后，每个 token 都有一个很自然的训练信号：

$$
r_t
= \log q(y_t \mid c_t) - \log \pi_\theta(y_t \mid c_t)
$$

teacher 比 student 更认可这个 token，$r_t$ 就高；teacher 觉得这个 token 不像好策略，$r_t$ 就低。Thinking Machines Lab 的实现里，几乎就是把 RL 训练脚本里的 KL regularizer model 换成 teacher：采 student rollout，算 student log-prob，再让 teacher 对同一条轨迹算 log-prob，最后把负 reverse KL 当作 per-token advantage。[^tml_opd]

所以 OPD 和 RL 的对应关系非常直接：

| RL 概念    | OPD 中的对应物                                           |
| ---------- | -------------------------------------------------------- |
| 状态 $s_t$ | prompt + 已生成前缀 $c_t$                                |
| 动作 $a_t$ | 下一个 token $y_t$                                       |
| 策略       | student $\pi_\theta$                                     |
| 奖励       | teacher 给 student token 的相对认可度                    |
| 采样分布   | student 自己的分布                                       |
| 优势估计   | 常用 $\log q(y_t \mid c_t)-\log \pi_\theta(y_t\mid c_t)$ |

这里要注意一个细节：teacher 不是环境真理，它只是一个强策略。OPD 不是让模型“发现超过 teacher 的新策略”，而是把 teacher 在 student 自己状态上的行为压进 student。它更像有过程监督的模仿学习，而不是纯探索型 RL。

## 为什么是 reverse KL

蒸馏里最容易混淆的是 KL 方向。

经典 KD 常用 forward KL：

$$D_{\text{KL}}(q \| \pi_\theta)$$

它要求 student 覆盖 teacher 的概率质量。对分类任务这很自然：teacher 说猫 0.7、狗 0.2、狐狸 0.1，student 也应该学这个软标签。但对长文本生成来说，覆盖所有 teacher 可能说的话会让 student 变得“摊大饼”：许多低概率、边缘的 token 也被抬高，生成会发散。

MiniLLM 的核心判断是：生成式 LLM 蒸馏更适合 reverse KL：

$$D_{\text{KL}}(\pi_\theta \| q)$$

reverse KL 是 mode-seeking：它鼓励 student 集中到 teacher 高概率的少数模式上，而不是覆盖 teacher 的所有可能模式。MiniLLM 还把这个目标落成 on-policy 优化，从而减少长文本生成里的 exposure bias。[^minillm]

但 reverse KL 有一个现实限制：**它只能在 student 当前会采到的区域里修正策略。** 如果某个关键 token 在 student 初始化里概率几乎为 0，student 根本采不到它，teacher 再懂也没机会给它高分。这就是为什么很多实用 recipe 都不是“直接 OPD”，而是：

1. 先用 off-policy SFT / SeqKD 做冷启动，把 student 拉到 teacher 支持集附近。
2. 再用 OPD 在 student 自己轨迹上做精修。

Thinking Machines Lab 的复现实验也是这样：先做 off-policy reasoning distillation，再用 OPD 做后训练提升。2026 年的 OPD 机制分析也得出类似结论：OPD 成功依赖 teacher 和 student 的思维模式兼容，并且 teacher 必须提供 student 训练数据之外的新能力；失败时常用 off-policy cold start 和 teacher-aligned prompt selection 来救。[^rethinking_opd]

还有一条工程化路线是调散度本身。DistiLLM 使用 skew KL 和 adaptive off-policy 策略，目标是在 teacher 信号和 student 可学习性之间找更平滑的折中。[^distillm]

## 一个最小可运行的打分例子

下面这段代码不是完整训练器，只做 OPD 最核心的一步：student 生成，teacher 在 student 的轨迹上逐 token 打分。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "Qwen/Qwen2.5-0.5B-Instruct"
teacher_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
student = AutoModelForCausalLM.from_pretrained(
    student_name, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
)
student.eval()
teacher.eval()

prompt = "Solve: if x + 3 = 7, what is x? Show your work."
inputs = tokenizer(prompt, return_tensors="pt").to(student.device)
prompt_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    output_ids = student.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
    )

full_ids = output_ids

def next_token_logps(model, input_ids):
    logits = model(input_ids).logits
    logps = torch.log_softmax(logits[:, :-1], dim=-1)
    next_ids = input_ids[:, 1:]
    return logps.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

with torch.no_grad():
    student_logps = next_token_logps(student, full_ids)
    teacher_logps = next_token_logps(teacher, full_ids.to(teacher.device)).to(student_logps.device)

# logps[:, i] 是第 i+1 个 token 的 log-prob。
# 第一个生成 token 位于 full_ids[:, prompt_len]，由位置 prompt_len-1 的 logits 预测。
gen_mask = torch.zeros_like(student_logps, dtype=torch.bool)
gen_mask[:, prompt_len - 1 :] = True

token_rewards = teacher_logps - student_logps
generated_ids = full_ids[:, 1:][gen_mask]
generated_rewards = token_rewards[gen_mask]

for tok_id, reward in zip(generated_ids[:32], generated_rewards[:32]):
    token = tokenizer.decode([tok_id.item()])
    print(f"{token!r:12s} reward={reward.item():+.3f}")
```

如果要把它变成训练循环，核心只差三件事：

1. 用 batch rollout 采样更多 prompt 和回答。
2. 对 reward 做 mask、归一化、裁剪或 advantage 估计。
3. 用 PPO / importance sampling / policy gradient 更新 student。

伪代码可以写成：

```python
for prompts in dataloader:
    trajectories = student.rollout(prompts)
    student_logps = student.logprobs(trajectories)
    teacher_logps = teacher.logprobs(trajectories)

    advantages = teacher_logps - student_logps
    advantages = normalize_and_mask(advantages, trajectories.response_mask)

    loss = policy_gradient_loss(
        new_logps=student.logprobs(trajectories),
        old_logps=student_logps.detach(),
        advantages=advantages.detach(),
    )
    loss.backward()
    optimizer.step()
```

真实系统还会加 KL 到 reference、长度控制、重复惩罚、prompt 难度采样和 eval gating。OPD 不是“一个公式就完事”，它是把 teacher log-prob 接到现有 RL 训练基础设施里。

## OPD 和几条相邻路线的关系

### 和 SFT / SeqKD 的关系

SFT 是必要的地基。它便宜、稳定、能快速把 student 拉到合理输出区域。OPD 不是替代 SFT，而是在 SFT 之后解决 student 自己轨迹上的错误修正。

可以粗略理解成：

- SFT / SeqKD：把路铺到 teacher 经常走的区域。
- OPD：student 自己开车，teacher 坐副驾逐步纠偏。

### 和 GRPO / RLVR 的关系

GRPO / RLVR 的奖励通常来自外部验证器：答案对不对、代码能不能跑、格式是否合格。这类奖励客观，但常常稀疏。一个 2000-token 的解题过程，最后只得到 0 或 1。

OPD 的奖励来自 teacher，每个 token 都能给信号。它不需要标准答案，也不需要 RM，但上限受 teacher 限制。

所以两者适合组合：数学题可以用规则奖励告诉模型最终答案是否正确，再用 teacher log-prob 给中间过程更密的 shaping 信号。Thinking Machines Lab 也把“蒸馏式 per-token reward + sequence-level environment reward”列为值得继续探索的方向。[^tml_opd]

### 和 DPO 的关系

DPO 的漂亮之处是：语言模型自己可以被看成隐式奖励模型。OPD 更进一步：直接把一个强 teacher 当作逐 token reward model。

但 DPO 是离线偏好优化，适合已有 chosen / rejected 数据的场景；OPD 是在线采样，适合你有 teacher、但缺少偏好数据或验证器的场景。

### 和黑盒蒸馏的关系

上面的 OPD 默认你能拿到 teacher log-prob，这是白盒或至少 logprob-access teacher。现实里很多 teacher 只有 API 文本输出，没有 logits。黑盒 OPD 就要换信号：例如 2025 年的 GAD 把 student 当 generator，训练 discriminator 区分 teacher 和 student 的回答，再用 discriminator 作为随 student 演化的 on-policy reward model。[^blackbox_opd]

这条线很有实践价值，但工程复杂度也高：你又引入了一个会被 hack、会漂移、需要稳定训练的判别器。

还有 teacher-free 的变体：Self-Distilled Reasoner 让同一个模型在不同上下文下同时扮演 teacher 和 student，用自蒸馏减少对外部强 teacher 的依赖。[^self_distilled_reasoner]

## 什么时候该用 OPD

OPD 最适合这些场景：

| 场景                     | 为什么 OPD 合适                                    |
| ------------------------ | -------------------------------------------------- |
| 小模型继承大模型推理能力 | 小模型探索弱，teacher 能给密集过程信号             |
| 没有规则验证器           | 不需要标准答案，只需要 teacher 能评估 token        |
| 领域模型后训练           | 可以用强 teacher 恢复 instruction following 或格式 |
| 已经有不错的 SFT 初始化  | student 能采到 teacher 支持集附近的 token          |
| 想降低 RL 成本           | teacher forward 比完整 RL 探索和稀疏奖励更高效     |

不适合这些场景：

| 场景                            | 风险                                          |
| ------------------------------- | --------------------------------------------- |
| 想明显超过 teacher              | OPD 本质是压缩 teacher 行为，不负责发现新策略 |
| student 太弱，采不到关键 token  | reverse KL 没法强化零概率附近的行为           |
| teacher 和 student 推理风格冲突 | 逐 token 信号可能互相拉扯，训练不稳定         |
| 长程任务只靠局部 token reward   | 局部对齐不一定等于全局任务成功                |
| 只有黑盒 teacher API            | 需要额外 reward/discriminator 设计            |

2026 年的 OPD survey 把这个领域整理成三个维度：反馈信号（logit-based、outcome-based、self-play）、teacher access（white-box、black-box、teacher-free）和 loss granularity（token-level、sequence-level、hybrid）。[^opd_survey] 这个分类很好用：一看到一个“OPD 新方法”，先问这三个问题，就不会被名字绕晕。

## 一个实用 recipe

如果你真的要跑 OPD，可以按这个顺序做：

**第一步，选 teacher。** teacher 不一定越大越好，关键是它在目标任务上比 student 强，并且和 student 的 tokenizer、输出风格、推理格式尽量兼容。同族模型通常更省心。

**第二步，做 off-policy 冷启动。** 用 teacher 生成一批高质量答案，先 SFT student。目标不是一步到位，而是让 student 进入 teacher 的支持集附近。

**第三步，采 student rollout。** 每个 prompt 采 2-8 条回答，保留 token、log-prob、mask、长度、停止原因。这里和 PPO / GRPO 的 rollout 基础设施基本相同。

**第四步，teacher 打分。** 对 student 生成的完整上下文做 teacher forward，拿生成 token 的 log-prob。白盒 teacher 可以直接算 logits；黑盒 teacher 需要另设 reward 近似。

**第五步，更新 student。** 用 teacher_logp - student_logp 做 per-token advantage，再接 PPO-style loss 或 importance-sampling loss。实践中要监控 entropy、KL、response length 和重复率，避免 student 过早塌缩。

**第六步，和任务奖励混合。** 如果任务有验证器，不要浪费它。可以把 final reward 用于序列级方向，把 OPD reward 用于 token-level shaping。

**第七步，做 eval gating。** OPD 很容易把 teacher 的风格也压进 student。除了目标 benchmark，还要测通用能力、格式、拒答、安全和长度分布。

## 本节小结

OPD 的核心不是“蒸馏又火了”，而是一个更基础的训练范式选择：

- off-policy 蒸馏有密集 token 监督，但不训练 student 自己会犯的错。
- RL 是 on-policy，但奖励常常稀疏，样本效率低。
- OPD 把两者接起来：student 自己采样，teacher 逐 token 给密集反馈。

它特别适合小模型、专用模型和后训练能力迁移。但它不是 RL 的替代品，也不是万能压缩器。OPD 的上限来自 teacher，稳定性来自初始化，价值来自“在 student 自己状态上给密集信号”。

从本章主线看，DPO、GRPO、RLVR 和 OPD 都是在回答同一个问题：**当我们不想完整跑传统 RLHF 时，训练信号还能从哪里来？** DPO 用偏好对，RLVR 用验证器，OPD 用 teacher。理解这三类信号的边界，才是真正能迁移到新项目里的能力。

## 参考文献

[^kd_survey_xu]: Xu X, Li M, Tao C, et al. [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116), arXiv 2024.（从 algorithm、skill、verticalization 三个角度整理 LLM KD）

[^kd_survey_yang]: Yang C, Lu W, Zhu Y, et al. [Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application](https://arxiv.org/abs/2407.01885), arXiv 2024.（按 white-box / black-box KD、评测和应用整理）

[^opd_survey]: Song M, Zheng M. [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626), arXiv 2026.（把 OPD 统一到 f-divergence 框架，并按反馈信号、teacher access、loss 粒度分类）

[^dagger]: Ross S, Gordon G, Bagnell D. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.（DAgger：让 learner 访问到的状态进入训练集）

[^gkd]: Agarwal R, Vieillard N, Zhou Y, et al. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024.（GKD：在 student 自生成序列上用 teacher 反馈缓解分布偏移）

[^minillm]: Gu Y, Dong L, Wei F, Huang M. [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), ICLR 2024.（用 reverse KL 和 on-policy 优化做生成式 LLM 蒸馏）

[^distillm]: Ko J, Kim S, Chen T, Yun S. [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://proceedings.mlr.press/v235/ko24c.html), ICML 2024.（用 skew KL 和 adaptive off-policy 提升 LLM 蒸馏效率）

[^tml_opd]: Lu K, Thinking Machines Lab. [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.（工程化 OPD 复现与 Tinker 实现，包含 Qwen3 对比和 personalization 实验）

[^rethinking_opd]: Li Y, Zuo Y, He B, et al. [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016), arXiv 2026.（分析 OPD 成功条件、token-level 机制和失败恢复策略）

[^self_distilled_reasoner]: Zhao S, Xie Z, Liu M, et al. [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734), arXiv 2026.（单模型在不同上下文下同时扮演 teacher 和 student）

[^blackbox_opd]: Ye T, Dong L, Chi Z, et al. [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643), arXiv 2025.（GAD：没有 teacher logits 时，用 discriminator 提供 on-policy 奖励）
