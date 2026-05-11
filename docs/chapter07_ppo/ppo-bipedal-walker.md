# 7.2 动手：BipedalWalker 连续控制

> **本节目标**：训练 PPO 控制二足机器人在随机地形上行走，理解连续动作空间下的策略学习与离散动作的本质区别。

> **本节代码**：[ppo_bipedal_walker.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo/ppo_bipedal_walker.py) · [render_bipedal_walker.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo/render_bipedal_walker.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter07_ppo/requirements.txt)

上一节用 PPO 训练了 LunarLander，但 LunarLander 只有 4 个离散动作。现实中的控制问题——机器人关节扭矩、汽车油门刹车、无人机旋翼转速——都是**连续动作空间**。PPO 的一个核心优势是原生处理连续动作：策略网络直接输出高斯分布的均值和标准差，从中采样得到连续动作，不需要把动作空间离散化。BipedalWalker-v3 就是这样一个连续控制任务。

## 7.2.1 运行 BipedalWalker 训练

BipedalWalker 的任务是控制一个二足机器人在随机生成的地形上行走。状态有 24 维（包括激光雷达测距、关节角度和速度），动作是 4 维连续向量（两条腿的髋关节和膝关节扭矩）。比 LunarLander 的 8 维状态、4 个离散动作复杂得多。

![BipedalWalker 的二足机器人需要在随机地形上稳定行走](./images/bipedalwalker_demo.gif)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 7.2-1：BipedalWalker 的目标是让二足机器人学会在地形上行走，而不是摔倒。</em>
</div>

安装依赖：

```bash
pip install -r code/chapter07_ppo/requirements.txt
```

运行训练：

```bash
python code/chapter07_ppo/ppo_bipedal_walker.py \
  --total-timesteps 1000000
```

BipedalWalker 的训练比 LunarLander 慢很多。LunarLander 用 20 万步就能看到明显学习趋势，BipedalWalker 通常需要 100 万步以上才能稳定行走。在普通 CPU 上，100 万步大约需要 30-60 分钟。如果只是验证管线能跑通，可以先用 `--total-timesteps 100000` 快速测试。

PPO 在 BipedalWalker 上的超参数配置：

```python
model = PPO(
    policy="MlpPolicy",       # 多层感知机策略
    env=vec_env,              # 8 个并行环境
    learning_rate=3e-4,       # 学习率
    n_steps=2048,             # 每次 rollout 步数
    batch_size=256,           # 小批量大小（比 LunarLander 的 64 大）
    n_epochs=10,              # 每批更新轮数
    clip_range=0.2,           # PPO 裁剪范围
    ent_coef=0.005,           # 熵系数（连续空间探索更丰富，稍低即可）
    gamma=0.99,               # 折扣因子
    gae_lambda=0.95,          # GAE λ
)
```

`batch_size` 从 64 提高到 256，因为连续动作空间下策略更新的方差更大，更大的批量有助于稳定梯度估计。`ent_coef` 从 0.01 降到 0.005，因为高斯策略本身就有持续的探索能力（每次采样都有随机性），不需要额外加太多熵正则化。并行环境从 4 个增加到 8 个，因为 BipedalWalker 的 episode 更长（最多 1600 步），更多并行环境能保持采样吞吐量。

## 7.2.2 查看训练曲线

训练完成后，脚本会在 `output/ppo_bipedal_walker_curves.png` 生成 2×2 的训练指标图。BipedalWalker 的训练曲线比 LunarLander 波动更大，收敛更慢——这是连续控制任务的常态。

![PPO 训练 BipedalWalker-v3 的四项训练指标：回合奖励逐步上升，策略熵缓慢下降，裁剪比例和 KL 散度保持在合理范围](./images/ppo-bipedalwalker-curves.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 7.2-2：PPO 训练 BipedalWalker-v3 的训练指标监控。1M 步后 20 回合评估均值为 205.51 ± 121.08。</em>
</div>

本节的 1M 步训练结果：20 回合评估均值为 `205.51 ± 121.08`，其中 18/20 回合得分超过 200。大多数成功回合的得分集中在 275-283 范围，但仍有少数回合因为初始扰动导致摔倒。这说明策略已经学到了有效的行走步态，但对所有初始状态还不够稳健。

## 7.2.3 成功行走的判定标准

BipedalWalker-v3 的奖励由几个部分组成：

- **前进奖励**：每步根据向右移动的距离给正奖励，走得越快越高。
- **关节效率惩罚**：使用过大的关节扭矩会扣分，鼓励高效运动。
- **坠落惩罚**：如果机器人摔倒（头部接触地面），扣 100 分并终止回合。

环境定义的"解决"标准是连续 100 个回合平均奖励 $\geq$ 300。实际训练中，单回合得分可以这样理解：

- **$\geq$ 300**：高质量行走，速度快、姿态稳、效率高。
- **200-300**：能走但不够稳，步态效率或速度还有问题。
- **100-200**：勉强能走一段，经常摔倒。
- **$<$ 100**：基本没学会，大多数回合都摔倒。

先看随机策略的基线：

```python
import gymnasium as gym
import numpy as np

env = gym.make("BipedalWalker-v3")
rng = np.random.default_rng(0)

returns = []
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    total_reward = 0.0
    for step in range(1600):
        action = rng.uniform(-1, 1, size=4)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    returns.append(total_reward)

print(f"随机策略平均回报: {np.mean(returns):.1f}")
print(f"最好一轮: {np.max(returns):.1f}")
print(f"最差一轮: {np.min(returns):.1f}")
```

随机策略的平均回报大约在 -100 到 -50 之间，几乎所有回合都会摔倒（扣 100 分）。如果 PPO 训练后评估回报仍然在这个范围，说明策略没有学到任何有效行为。

一次运行的结果为：

```text
随机策略平均回报: -46.3
最好一轮: -0.2
最差一轮: -100.0
```

## 7.2.4 典型回放：行走、蹒跚与摔倒

训练完成后，可以用渲染脚本生成回放 GIF：

```bash
python code/chapter07_ppo/render_bipedal_walker.py \
  --model output/ppo_bipedal_walker.zip \
  --output-dir output/bipedalwalker_episodes \
  --episodes 10 --seeds 0 1 2 3 4 5 6 7 8 9
```

训练完成后的典型回放如下。三段 GIF 来自同一个训练后的模型，使用不同的 reset seed，展示同一策略在不同初始条件下的表现。

**Episode 1（回报 283.5，1428 步）** 是高质量行走。机器人快速进入稳定的步态周期，关节协调流畅，在地形上高效前进。步态效率高，关节扭矩惩罚低，因此总得分接近 284。

![BipedalWalker Episode 1：稳定高效的行走步态，回报 283.5](./images/bipedalwalker_ep1.gif)

**Episode 2（回报 161.4，1354 步）** 是中等质量。机器人最终完成了行走，但过程中步态不协调，出现了多次踉跄和修正。与高分样例相比，速度更慢，关节扭矩使用效率更低。

![BipedalWalker Episode 2：步态不稳但仍完成行走，回报 161.4](./images/bipedalwalker_ep2.gif)

**Episode 3（回报 -29.9，494 步）** 是失败。机器人在 494 步内摔倒，触发 -100 的坠落惩罚。低分来自坠落惩罚加上行走距离不足——策略在这个初始条件下没有找到稳定步态。

![BipedalWalker Episode 3：步态失控后摔倒，回报 -29.9](./images/bipedalwalker_ep3.gif)

这三段回放说明：训练到 1M 步时，PPO 已经学到了有效步态，但策略对某些初始状态仍不够稳健。继续训练到 2-3M 步通常能进一步降低摔倒率。

## 7.2.5 状态、动作与连续策略

BipedalWalker 的 24 维状态可以分成几组：

| 状态分量 | 维度 | 含义 |
| --- | --- | --- |
| `hull_angle` | 1 | 躯干倾斜角 |
| `hull_angular_velocity` | 1 | 躯干角速度 |
| `vx, vy` | 2 | 躯干水平/垂直速度 |
| `hip1, hip2` | 2 | 两条腿的髋关节角度 |
| `knee1, knee2` | 2 | 两条腿的膝关节角度 |
| `leg1_contact, leg2_contact` | 2 | 两只脚是否接触地面 |
| `lidar[0..9]` | 10 | 激光雷达测距（探测前方地形） |
| `hip_speed1, hip_speed2` | 2 | 髋关节角速度 |
| `knee_speed1, knee_speed2` | 2 | 膝关节角速度 |

动作是 4 维连续向量，每个分量在 $[-1, 1]$ 之间：

| 动作分量 | 含义 |
| --- | --- |
| `action[0]` | 腿 1 髋关节扭矩 |
| `action[1]` | 腿 1 膝关节扭矩 |
| `action[2]` | 腿 2 髋关节扭矩 |
| `action[3]` | 腿 2 膝关节扭矩 |

PPO 处理连续动作的方式和处理离散动作不同。在离散动作空间（如 LunarLander），策略网络输出每个动作的概率，然后从中采样。在连续动作空间，策略网络输出一个高斯分布——均值 $\mu(s)$ 和标准差 $\sigma(s)$，然后从 $\mathcal{N}(\mu, \sigma^2)$ 中采样得到动作：

$$a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$$

这意味着 PPO 不需要把连续动作离散化成有限个选择。对于关节扭矩这种需要精细控制的量，离散化会导致分辨率损失：如果只允许 -1、0、+1 三档扭矩，机器人会非常笨拙。连续策略可以直接输出 0.37 或 -0.82 这样的精细值，控制精度远高于离散化方案。

BipedalWalker 的学习过程通常经历三个阶段：

1. **站立（0-200k 步）**：策略先学会不摔倒。这个阶段奖励从 -100 缓慢上升到 0 附近，机器人从"一启动就倒"变成"能勉强站着"。
2. **挪步（200k-600k 步）**：策略开始试探性地迈步，但步态不协调，经常走几步就摔倒。奖励波动剧烈，偶尔出现 100 分以上的回合。
3. **行走（600k 步以后）**：步态逐步成型，行走越来越稳。如果训练充分，奖励可以稳定在 200-300 以上。

这三个阶段不是严格划分的，不同训练 run 的边界会偏移。但整体趋势是：先学会"不摔倒"，再学会"迈步"，最后学会"高效行走"。

## 7.2.6 常见失败与调参

BipedalWalker 比 LunarLander 更容易训练失败。如果结果不理想，按以下顺序排查。

第一，确认训练步数是否足够。100 万步是起点，不是终点。如果曲线还在上升但斜率不够，可以继续训练：

```python
from stable_baselines3 import PPO
model = PPO.load("output/ppo_bipedal_walker.zip")
model.learn(total_timesteps=2_000_000, reset_num_timesteps=False)
```

第二，确认 batch_size 是否够大。连续动作空间的梯度方差比离散空间高，`batch_size=64` 可能不够稳定。本节使用 256，如果训练曲线仍然剧烈震荡，可以尝试 512。

第三，检查策略熵。如果熵在训练早期就快速降到接近零，说明策略过早收敛到次优行为（比如一直站着不动）。可以适当增大 `ent_coef` 到 0.01。

第四，考虑网络容量。默认的 `MlpPolicy` 使用两层 64 个神经元的网络。对于 24 维状态，这个容量可能不够。可以通过 `policy_kwargs` 增大网络：

```python
model = PPO(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[128, 128]),
    ...
)
```

常见超参数参考：

| 参数 | 本节设置 | 如果不合适会怎样 |
| --- | --- | --- |
| `learning_rate` | `3e-4` | 太大会让关节扭矩输出剧烈震荡，太小学得很慢 |
| `batch_size` | `256` | 太小梯度估计不稳定，太大单次更新太慢 |
| `n_steps` | `2048` | 太少每次 rollout 数据不够，太多延迟策略更新频率 |
| `ent_coef` | `0.005` | 太小策略过早收敛到站着不动，太大始终无法形成稳定步态 |
| `clip_range` | `0.2` | 太大步态突变导致摔倒，太小训练停滞 |
| `gamma` | `0.99` | 太低只关注短期不摔，忽略长期行走效率 |

## 7.2.7 为什么选择 BipedalWalker

BipedalWalker 在教学上承接了 LunarLander 的几个关键提升：

- **连续动作空间**。LunarLander 用 4 个离散动作控制发动机，BipedalWalker 用 4 维连续扭矩控制关节。这正是 PPO 相对于 DQN 的核心优势所在——DQN 无法直接处理连续动作，必须先离散化，而 PPO 通过高斯策略原生支持。
- **更丰富的状态空间**。24 维状态中包含 10 个激光雷达测距信号，这是对真实机器人传感器（激光雷达、触觉传感器）的简化模拟。智能体必须从这些传感器数据中推断地形变化，而不是简单地读取位置坐标。
- **更复杂的学习动态**。LunarLander 的策略本质上是一个控制问题——控制下降速度和姿态。BipedalWalker 需要发现一个完整的步态周期，涉及多关节协调、重心转移和地形适应。策略的复杂度上了一个台阶。

从 CartPole（第 5 章）到 LunarLander（本章 7.1 节）再到 BipedalWalker，实验的难度和真实性逐步提升。但 PPO 的核心机制没有变：裁剪约束更新幅度，多轮复用同一批数据，Actor-Critic 同时优化策略和价值。复杂度的提升来自环境和任务，而不是算法本身。

下一节将拆解 PPO 背后的数学推导——[PPO 数学推导](./ppo-math)。

## 本节小结

- `BipedalWalker-v3` 承接 LunarLander 实验，从离散动作升级到连续动作空间，从 8 维状态升级到 24 维。
- PPO 通过高斯策略（输出均值和标准差，从中采样）原生支持连续动作，不需要离散化。
- BipedalWalker 的学习过程通常经历"站立→挪步→行走"三个阶段，训练步数需求远高于 LunarLander。
- 本节训练入口是 `code/chapter07_ppo/ppo_bipedal_walker.py`，回放 GIF 用 `render_bipedal_walker.py`。
- 环境定义的"解决"标准是 100 回合平均奖励 $\geq$ 300；实际训练中，1M 步通常能达到 200-300 的范围。

## 参考文献

[^1]: Raffin, A., et al. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. _Journal of Machine Learning Research_, 22(268), 1-8.

[^2]: Schulman, J., et al. (2017). Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_.
