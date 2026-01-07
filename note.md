Minimind 笔记

跟据gemini提问学习

## [RoPE](model\model_minimind.py)

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        #RoPE 的核心数学操作，作用是对输入张量的最后一维（特征维度）做 “半部分旋转”：
        #[x,y]->[-y,x] ,变换前后的两个向量是正交的,因此可以视为做一个旋转变换,从而体现位置信息
        #
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

$$\vec{v}_{new} = \vec{v} \cdot \cos\theta + \vec{v}_\perp \cdot \sin\theta$$ 
其中cos控制的是：保留多少“原来的自己”**,**sin 控制的是：混入多少“垂直方向的分量”**(即变化)

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)... / dim))
```

**前面的维度（index 小）**：指数小 $\rightarrow$ 分母小 $\rightarrow$ **频率高**。

**后面的维度（index 大）**：指数大 $\rightarrow$ 分母大 $\rightarrow$ **频率低**。

因此前面的维度擅长短期依赖,后面的擅长长期依赖,从而实现了短期到长期的捕捉

### 外推性

**只看“相对角度”**：Attention 机制在计算时，不关心向量绝对转到了 100 度还是 1000 度，只关心两个向量之间的**角度差**（相对距离）。

**规律恒定**：无论你是在第 5 个位置，还是在第 50000 个位置，只要两个词相距 1 格，它们之间的**旋转角度差**永远是固定的 $\theta$。

虽然理论上RoPE可以无限外推,但是它的长文本注意力是有限的;如果只用基础 RoPE），模型处理超过 2048 长度的文本时会“即使读了也记不住位置”，导致输出崩坏 , 我们需要用YaRN来扩展它在更长文本的"视力"

### YaRN

也在model_minimind中,用于解决模型在处理**超出训练长度**的文本时性能下降的问题。

前排的高频维度:不(or 少)插值,因为短期依赖基本不变
后排的高频维度:**进行线性插值（Interpolation），即“拉伸”**;让它的旋转周期覆盖更长的范围



###  [Pretrain](trainer\train_pretrain.py)

模型看到 A 预测 B，看到 B 预测 C ,学习语言的通用规律,对整段文本计算 Loss

```python
# input_ids: [A, B, C, D]
X = input_ids[:-1]  # 输入: [A, B, C]
Y = input_ids[1:]   # 目标: [B, C, D]
```

这段在happy-llm中也有做过

### SFT:

###### SFT Loss Mask

相比于 Pretrain 对整段文本计算 Loss（误差），**为什么在 SFT 阶段，我们要把“User（用户指令）”部分的 Loss Mask 设为 0，只计算“Assistant（模型回答）”部分的 Loss**

```python
# (伪代码简化示意)
对话: User: "你好" -> Assistant: "我是MiniMind"
输入: [User, 你, 好, Assistant, 我, 是, ...]
Mask: [0,    0,  0,  0,         1,  1,  ...]
  #            User部分      Assistant部分
```

- **正常情况 (Mask User)**：模型学到的是：“看到用户的指令后，我应该切换状态，开始输出对应的回答。”
- **计算 User Loss**：模型会误以为它的任务是 **“文本续写”** 或 **“模仿人类提问”**。
  - **后果**：当你问“在这个代码库中...”，模型可能不会回答你，而是接着生成“...有哪些文件？...怎么运行？...作者是谁？”。它学会了预测“接下来用户可能会问什么”，而不是“我该怎么回答”。

我们的目标是让模型在 **Assistant（回答）** 部分表现完美。

如果让模型分心去学习如何预测 **User（提问）** 部分，它会把大量的“脑力”浪费在拟合各种千奇百怪的提问方式上，而不是专注于提升回答的质量。





### RLHF

#### [PPO:](\trainer\train_ppo.py)

##### Reward Model (奖励模型) —— **最终**的“阅卷老师” 👩‍🏫

- **角色**：它是**裁判**。它不参与写作业，只在作业写完后打分。且不参与训练,在训练开始前就已经训练完毕
- **特点**：
  - **输入**：完整的问题 + 完整的回答。
  - **输出**：一个标量分数（Reward）。
  - **时机**：**回合结束时**（生成完所有文字后）才给出一个分。

#####  Critic Model (评论/价值模型) ——**实时**的“补习教练” 🧢

- **角色**：它是**预言家**。它在学生（Actor）每写一个字的时候，都在旁边估算：“嗯，照你**目前这个写法，最后大概能得多少分**。”
- **代码体现**： 在 `trainer/train_ppo.py` 中，它被定义为 `CriticModel`，将原本的 `lm_head`（预测词的层）换成了 `value_head`（预测分数的层）。
- **特点**：
  - **输入**：问题 + 目前已生成的**部分**回答。
  - **输出**：一个标量数值（Value）。
  - **时机**：**每生成一个 Token（字）** 都会给出一个估值。

**奖励 (Reward)**：是你**走到终点**时拿到的宝箱（比如 +100 分），或者是掉进陷阱时的惩罚（-100 分）。它是**既定事实**，通常很稀疏（只有最后有）。

**价值 (Value)**：是你站在迷宫**某一个路口**时，心里盘算的：“既然我到了这个路口，按我的经验，后面大概率能拿到多少分？”。它是**预估值**，覆盖全程。

```python
advantages = rewards - values.detach()
```

advantages>0应当鼓励,<0应当抑制

##### 重要性采样

```python
ratio =torch.exp(actor_logp - old_logp)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

数据由旧策略得到,用于更新新策略,我们使用**重要性采样模拟新策略的概率分布** 

通过clip约束ratio不太大也不太小,每次更新都在信任区域” (Trust Region) 内

这里使用了裁切,但也会有用kl散度作为软约束的做法

##### PPO loss

```python
 loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps  # scalar
```

1. policy_loss,即上面提到的部分
2. Value Loss: Critic（评论家）的任务。它的目标是：**预测得分要尽可能接近真实得分。**
3. kl散度惩罚 :`kl_ref = (actor_logp - ref_logp).mean()`防止模型为了讨好 Reward Model 而“胡言乱语”（Reward Hacking）,让**actor不走这么远**
   - Reward Model毕竟只是在有限数据集上训练出来的,很容易被摸到边界
   - 通过kl_ref可以防止 Actor 为了去够那个虚假的“高分”，越跑越远，彻底脱离人类语言的分布。
4. aux_loss :  MoE (Mixture of Experts) 模型特有的。如果你的模型不是 MoE 架构，这项就是 0。它的作用是确保每个“专家”都能得到充分的训练

##### **Old Actor Model (旧策略模型)**：

- **身份**：它是 **Actor 的“昨天”**（或者说几步之前的快照）。
- **状态**：**动态更新**。它不是一直冻结的，而是每隔几步就会追上现在的 Actor。
- 隔几步就更新,能够提高数据的利用率,大大加快训练进程