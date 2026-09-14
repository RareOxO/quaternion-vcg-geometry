# Quaternion-VCG 动态几何表示学习 实现与验证方案

用于 Codex 实现的理论与工程说明（第一阶段：PTB-XL 主验证）

本文档聚焦当前最小、最必要的研究闭环：先验证显式 VCG 动态几何、Quaternion 结构化交互、多尺度和二阶动态是否有效。第一阶段不加入 radial branch、scale attention、Transformer、ECG+VCG fusion、EMD、curvature 或 torsion。

## 1. 研究目标与最小验证闭环

研究对象是由 12 导联 ECG 通过 Kors 变换得到的三维 VCG 序列。核心问题不是"把 VCG 强行四元数化"，而是利用 Quaternion/Hamilton algebra 对两个三维方向之间的几何关系进行结构化表示。

Q1：显式 dynamic geometry 是否优于直接输入 raw XYZ VCG？

Q2：Quaternion structured interaction 是否优于相同输入上的普通 real-valued interaction？

Q3：Multi-scale geometry 是否优于 single-scale？

Q4：Second-order geometry 是否能提供 first-order 之外的额外信息？

模型主线固定为：

**M0 Raw VCG → M1 Real First-order Geometry → M2 Quaternion First-order → M3 + Multi-scale → M4 + Second-order**

## 2. 信号与 VCG 基础定义

标准 12 导联 ECG 经过 Kors transform 后得到三维 VCG：

\[
V_t = [X_t, Y_t, Z_t]
\]

每个时刻的 \(V_t\) 可以理解成一个三维箭头。箭头长度表示当前 cardiac electrical vector 的 magnitude，箭头方向表示当前三维 cardiac electrical orientation。第一阶段只研究方向随时间的变化。

\[
u_t = \frac{V_t}{\lVert V_t\rVert_2 + \mathrm{eps}}
\]

\[
\lVert V_t\rVert_2 = \sqrt{X_t^2 + Y_t^2 + Z_t^2}
\]

其中 eps 用于数值稳定。理论上 \(\lVert u_t\rVert \approx 1\)。

## 3. 为什么这里使用 Quaternion

Quaternion 一般写成：

\[
q = a + bi + cj + dk = [a,b,c,d]
\]

其中 \(a\) 是 scalar / real part，\([b,c,d]\) 是 vector / imaginary part。Quaternion 的核心运算是 Hamilton product。对于

\[
q_1 = [a,b,c,d], \qquad q_2 = [e,f,g,h]
\]

Hamilton product \(q_1 \otimes q_2\) 展开为：

\[
\begin{aligned}
\mathrm{real} &= ae - bf - cg - dh,\\
i &= af + be + ch - dg,\\
j &= ag - bh + ce + df,\\
k &= ah + bg - cf + de.
\end{aligned}
\]

关键性质：若把三维向量 \(x\)、\(y\) 写成 pure quaternion：

\[
p_x = [0,x], \qquad p_y = [0,y]
\]

\[
p_x \otimes p_y = [-x\cdot y,\;x\times y]
\]

也就是说，一个 Hamilton product 同时产生 dot product 与 cross product。因此，Quaternion 在本研究中的合理性来自"两个 3D 方向之间的结构化关系"，而不是"VCG 有 XYZ 三个通道，Quaternion 恰好有四个分量"。

## 4. First-order Quaternion Geometry 的构建

对时间尺度 \(\Delta\)，比较 \(u_t\) 与 \(u_{t+\Delta}\)。

\[
\mathrm{dot}_t^\Delta = u_t \cdot u_{t+\Delta}
\]

由于两个方向向量都归一化：

\[
\mathrm{dot}_t^\Delta = \cos(\theta)
\]

dot \(\approx 1\)：前后方向几乎一致；

dot \(\approx 0\)：夹角约为 90°；

dot \(\approx -1\)：方向接近反向。

同时计算 cross：

\[
\mathrm{cross}_t^\Delta = u_t \times u_{t+\Delta}
\]

对于单位向量：

\[
\lVert \mathrm{cross}_t^\Delta\rVert = \sin(\theta)
\]

cross 的方向是由 \(u_t\) 与 \(u_{t+\Delta}\) 所张成平面的法向方向，因此它保留"转动朝哪个三维方向发生"的信息。

### 4.1 Quaternion Relation Descriptor

本研究定义：

\[
q_t^\Delta =
[
\mathrm{dot}_t^\Delta,\;
\mathrm{cross}_{x,t}^\Delta,\;
\mathrm{cross}_{y,t}^\Delta,\;
\mathrm{cross}_{z,t}^\Delta
]
\]

\[
q_t^\Delta =
[
u_t\cdot u_{t+\Delta},\;
u_t\times u_{t+\Delta}
]
\]

注意：直接对两个 pure quaternion 做 Hamilton product 得到的是 \([-\mathrm{dot},\mathrm{cross}]\)。为了让 scalar part 更直观地表示"方向一致程度"，本文主表示定义为 \([+\mathrm{dot},\mathrm{cross}]\)。代码必须区分这两个概念：

raw pure-quaternion Hamilton product：\([-\mathrm{dot},\mathrm{cross}]\)；

论文使用的 relation descriptor：\([+\mathrm{dot},\mathrm{cross}]\)。

### 4.2 与标准 rotation quaternion 的区别

这是实现和论文表述中最需要避免混淆的部分。对于夹角 \(\theta\) 与法向 \(n\)，当前 relation descriptor 近似具有形式：

\[
q_{\mathrm{relation}} =
[
\cos(\theta),\;
n\sin(\theta)
]
\]

而标准描述三维刚体旋转 \(\theta\) 的 unit rotation quaternion 通常为：

\[
q_{\mathrm{rotation}} =
[
\cos(\theta/2),\;
n\sin(\theta/2)
]
\]

因此，\([\mathrm{dot},\mathrm{cross}]\) 不能直接称为 standard physical rotation quaternion。推荐术语：Quaternion Relation Descriptor、Hamilton Angular Relation、Quaternion-based Directional Relation。

### 4.3 模长性质

\[
\mathrm{dot}^2 + \lVert \mathrm{cross}\rVert^2
=
\cos^2(\theta) + \sin^2(\theta)
=
1
\]

所以理论上 \(\lVert q_{\mathrm{relation}}\rVert \approx 1\)。实现中应加入数值检查；是否再次归一化应作为配置项，而不是静默执行。

## 5. First-order、Multi-scale 与 Second-order

First-order 描述从 \(t\) 到 \(t+\Delta\) 的一次局部方向关系：

\[
u_t \rightarrow u_{t+\Delta} \rightarrow q_t^\Delta
\]

它不是传统 angular velocity 的直接复现，而是完整保留 scalar alignment 与 3D turning orientation 的时间序列。

### 5.1 Multi-scale Geometry

第一阶段使用固定候选尺度：10、20、40、80 ms。在 500 Hz 下分别对应 5、10、20、40 个采样点。实现中必须根据采样率动态换算：

\[
\mathrm{lag\_samples}
=
\operatorname{round}
\left(
\frac{\mathrm{lag\_ms}\times f_s}{1000}
\right)
\]

每一个时间尺度分别得到一个 Quaternion relation sequence：

\[
q_t^{10},\quad q_t^{20},\quad q_t^{40},\quad q_t^{80}
\]

Multi-scale 不能被错误地理解成"把四个尺度塞进 Quaternion 的 real/i/j/k"。Quaternion 表示空间方向关系，scale 表示时间尺度，两者必须分离。第一阶段可将不同尺度视为不同 quaternion channels。

### 5.2 Second-order Geometry

First-order 回答"这一段方向怎么变"；Second-order 回答"前一段和后一段的方向变化模式相比，又发生了怎样的变化"。对于固定 \(\Delta\)：

\[
q_{\mathrm{prev}} = \mathrm{relation}(u_{t-\Delta},u_t)
\]

\[
q_{\mathrm{next}} = \mathrm{relation}(u_t,u_{t+\Delta})
\]

第一阶段主定义采用最安全、最容易解释的差分：

\[
s_t^\Delta = q_{\mathrm{next}} - q_{\mathrm{prev}}
\]

它表示 Quaternion relation descriptor 自身随时间发生的变化，因此可解释为 second-order geometric change。

第一阶段不要把 \(\mathrm{inverse}(q_{\mathrm{prev}})\otimes q_{\mathrm{next}}\) 作为主定义，也不要把它直接解释为 relative physical rotation，因为当前 \(q_{\mathrm{relation}}\) 采用 \(\cos(\theta)\)、\(\sin(\theta)\)，并非标准 half-angle rotation quaternion。该组合可作为未来探索性 ablation。

## 6. 必须通过的理论 Sanity Checks

| 测试 | 输入/构造 | 预期 |
| --- | --- | --- |
| 恒定方向 | \(u_t\) 恒定 | \(q_{\mathrm{first}}=[1,0,0,0]\)；second-order \(\approx 0\) |
| 90° 已知旋转 | \([1,0,0]\rightarrow[0,1,0]\) | \(\mathrm{dot}=0\)，\(\mathrm{cross}=[0,0,1]\)，\(q=[0,0,0,1]\) |
| 反向 | \([1,0,0]\rightarrow[-1,0,0]\) | \(\mathrm{dot}=-1\)，\(\mathrm{cross}=0\) |
| 匀速圆周旋转 | \(\theta(t)=\omega t\) | 固定 \(\Delta\) 下 first-order 近似恒定；second-order \(\approx 0\) |
| 加速旋转 | \(\theta(t)=a t^2\) | first-order 随时间变化；second-order 非零 |
| Hamilton 基础 | \(i\otimes j\) | 结果应为 \(k\)，即 \([0,0,0,1]\) |

## 7. Quaternion Neural Layer 的使用

M2 之后必须真正使用 Quaternion layer，而不是仅把 \(q_{\mathrm{relation}}\) 当普通 4 维向量。QuaternionLinear / QuaternionConv1d 应基于 Hamilton coupling 展开。若输入

\[
Q = Q_r + Q_i i + Q_j j + Q_k k
\]

权重

\[
W = W_r + W_i i + W_j j + W_k k
\]

则：

\[
\begin{aligned}
Y_r &= W_rQ_r - W_iQ_i - W_jQ_j - W_kQ_k,\\
Y_i &= W_rQ_i + W_iQ_r + W_jQ_k - W_kQ_j,\\
Y_j &= W_rQ_j - W_iQ_k + W_jQ_r + W_kQ_i,\\
Y_k &= W_rQ_k + W_iQ_j - W_jQ_i + W_kQ_r.
\end{aligned}
\]

这意味着 scalar 与三个 vector components 被 Hamilton 规则强制耦合。该结构化耦合就是我们需要验证的 Quaternion inductive bias。

### 7.1 Real baseline 的公平性要求

Real baseline 与 Quaternion model 必须使用完全相同的输入数值 \([\mathrm{dot},\mathrm{cross}_x,\mathrm{cross}_y,\mathrm{cross}_z]\)。Real 模型把它当 4 个普通 channels；Quaternion 模型把它解释为 one quaternion channel，并使用 QuaternionConv1d。除了 interaction rule 外，尽量控制参数量相近。

## 8. 五个主模型的严格定义

| 模型 | 输入 | 运算 | 目的 |
| --- | --- | --- | --- |
| M0 Raw VCG | \([X,Y,Z]\) | Small real TCN | Raw VCG 基线 |
| M1 Real First-order Geometry | 单尺度（默认 20 ms）的 \([\mathrm{dot},\mathrm{cross}_x,\mathrm{cross}_y,\mathrm{cross}_z]\) | Real TCN | 验证显式 geometry 是否有价值 |
| M2 Quaternion First-order | 与 M1 完全相同输入 | QuaternionConv1d + 轻量时序编码 | 验证 Hamilton structured interaction |
| M3 Quaternion Multi-scale | \(q^{10},q^{20},q^{40},q^{80}\) | 不同尺度作为 quaternion channels | 验证多时间尺度互补性 |
| M4 Full Proposed | 每个尺度的 first-order + second-order difference | 轻量 Quaternion temporal encoder | 验证 second-order 增量价值 |

## 9. 第一阶段数据与训练范围

第一阶段只使用 PTB-XL，不实现 Chapman 或 Georgia。数据配置：12-lead、10 s、500 Hz；通过 Kors transform 得到 XYZ VCG；标签为 NORM、MI、STTC、CD、HYP；multi-label；fold 1-8 训练、fold 9 验证、fold 10 测试。主指标为 Macro AUROC，同时输出 per-class AUROC。损失使用 BCEWithLogitsLoss。

## 10. 最终必须生成的结果表

Table 1：PTB-XL 主结果

| Model | Geometry | Quaternion | Multi-scale | 2nd-order | Macro AUROC | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 Raw VCG | × | × | × | × | --- | --- | --- | --- | --- | --- |
| M1 Real Geometry | ✓ | × | × | × | --- | --- | --- | --- | --- | --- |
| M2 Quaternion Geometry | ✓ | ✓ | × | × | --- | --- | --- | --- | --- | --- |
| M3 + Multi-scale | ✓ | ✓ | ✓ | × | --- | --- | --- | --- | --- | --- |
| M4 Full Proposed | ✓ | ✓ | ✓ | ✓ | --- | --- | --- | --- | --- | --- |

Table 2：Temporal-scale Analysis

| Scale | Macro AUROC | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- | --- |
| 10 ms | --- | --- | --- | --- | --- | --- |
| 20 ms | --- | --- | --- | --- | --- | --- |
| 40 ms | --- | --- | --- | --- | --- | --- |
| 80 ms | --- | --- | --- | --- | --- | --- |
| 10+20+40+80 ms | --- | --- | --- | --- | --- | --- |

Table 3：Real vs Quaternion

| Representation | Operator | Params | Macro AUROC |
| --- | --- | --- | --- |
| dot+cross | Real Conv | --- | --- |
| dot+cross | Real MLP | --- | --- |
| dot+cross | Quaternion Conv | --- | --- |

Table 4：组件增量

| Component Added | \(\Delta\) Macro AUROC |
| --- | --- |
| Explicit Geometry | M1 − M0 |
| Quaternion Interaction | M2 − M1 |
| Multi-scale | M3 − M2 |
| Second-order | M4 − M3 |

## 11. Codex 实现顺序

检查当前 repository，给出简洁 implementation plan。

先实现理论相关 geometry functions：normalize direction、dot、cross、q_relation、Hamilton product、inverse、first-order、multi-scale、second-order difference。

实现 QuaternionLinear 与 QuaternionConv1d。

先完成 synthetic tests，并输出 expected vs actual。

只在测试全部通过后，接入 PTB-XL 与 Kors transform。

先实现 M0 与 M1，确认 geometry 是否值得继续。

再实现 M2，做 Real vs Quaternion 公平比较。

之后实现 M3 Multi-scale。

最后实现 M4 Second-order。

自动生成主结果表、scale 表、Real-vs-Quaternion 表和组件增量表。

## 12. 关键禁止事项

不要用"VCG 有 3 个通道，所以 Quaternion 天然适合"作为理论依据。

不要把 \([\mathrm{dot},\mathrm{cross}]\) 直接称为标准 physical rotation quaternion。

不要把 10/20/40/80 ms 塞入 quaternion 的 real/i/j/k。

不要默认 Quaternion 一定优于 Real。

Quaternion 与 Real baseline 必须使用完全相同的 geometry input。

第一阶段不要加入 radial branch、scale attention、Transformer、ECG+VCG fusion、EMD、curvature 或 torsion。

不要调参于 test set，不要改变官方 PTB-XL split。
