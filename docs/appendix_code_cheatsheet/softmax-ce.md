# C.5 Softmax 与 Cross-Entropy

面试热身题。写 DPO / PPO 之前，面试官可能先让你手写一个数值稳定的 softmax 和交叉熵。

---

## 数值稳定的 Softmax

### 一句话记忆

> **先减 max 再 exp，分母是所有 exp 之和。**

### 伪代码

```
x_shifted = x - max(x)
exp_x = exp(x_shifted)
softmax = exp_x / sum(exp_x)
```

### 记忆方法

为什么不直接 `exp(x) / sum(exp(x))`？因为 `exp(1000)` = inf。减去 max 后最大值变成 `exp(0)=1`，其余都小于 1，不会溢出。

### Python 实现

```python
import numpy as np

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

# 工程中直接用
probs = F.softmax(logits, dim=-1)

# 手写版（面试用）
def manual_softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x_shifted)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

---

## Log-Sum-Exp 技巧

### 一句话记忆

> **$\log\sum\exp(x) = \text{max}(x) + \log\sum\exp(x - \text{max}(x))$。**

面试追问：`log(softmax(x))` 怎么算才不会溢出？答：不要先 softmax 再 log，直接用 log-softmax。

### Python 实现

```python
def log_softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
```

### PyTorch 实现

```python
# 内置，数值稳定
log_probs = F.log_softmax(logits, dim=-1)

# 手写
def manual_log_softmax(x, dim=-1):
    max_val = x.max(dim=dim, keepdim=True).values
    return x - max_val - torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))
```

---

## Cross-Entropy Loss

### 一句话记忆

> **one-hot 标签的负对数概率：$-\sum_k y_k \log p_k$。简化版（整数标签）：$-\log p_{y}$。**

### 伪代码

```
log_probs = log_softmax(logits)
loss = -log_probs[target].mean()
```

### 记忆方法

交叉熵 = 一个预测对了多少的度量。预测越准，$p_{y}$ 越大，$-\log p_{y}$ 越小。loss 小 = 好。

### Python 实现

```python
def cross_entropy(logits, targets, ignore_index=-100):
    """
    logits: [N, C]
    targets: [N]  (整数类别)
    """
    log_probs = log_softmax(logits, axis=-1)
    total, count = 0.0, 0
    for i in range(len(targets)):
        if targets[i] == ignore_index:
            continue
        total += -log_probs[i, targets[i]]
        count += 1
    return total / max(count, 1)
```

### PyTorch 实现

```python
def manual_cross_entropy(logits, targets, ignore_index=-100):
    """
    logits: [B, C]
    targets: [B]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # gather 取出 target 位置的 log 概率
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # mask 掉 ignore_index
    mask = targets != ignore_index
    return -target_log_probs[mask].mean()
```

---

## 易错点

| 易错 | 说明 |
|------|------|
| softmax 忘了减 max | 面试手写第一步就扣分 |
| 先 softmax 再 log | 数值不稳定，用 `log_softmax` 一步到位 |
| Cross-Entropy 从 logits 算 | 不要先 softmax 再 log 再 CE，直接 `F.cross_entropy(logits, targets)` |
| `ignore_index` | 面试追问 SFT loss 时会问，padding token 怎么处理 |
| temperature | `logits / temperature` 再 softmax，T 越大分布越平 |
