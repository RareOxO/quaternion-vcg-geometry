#set text(font: ("New Computer Modern", "Noto Serif CJK SC", "Source Han Serif SC", "SimSun"), size: 10pt)
#set page(margin: 2cm)
#show table.cell.where(y: 0): strong

= 参考网络与实现模型对照

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  table.header([参考文献], [原始领域], [对应模型], [状态]),

  [Brignone et al., _Efficient Sound Event Localization and Detection in the Quaternion Domain_, TCAS-II 2022],
  [3-D 声场事件定位], [`E1_qtcn`], [已改编实现],

  [Jiang et al., _QFormer: An Efficient Quaternion Transformer for Image Denoising_, IJCAI],
  [图像去噪], [`E1_qtransformer`], [已改编实现],

  [Liu et al., _QSTGNN: Quaternion Spatio-Temporal Graph Neural Networks_, TKDE 2025],
  [时空图预测], [`E1_qgnn`], [已改编实现],

  [Parcollet et al., _Quaternion Recurrent Neural Networks_, ICLR 2019 (arXiv:1806.04418; QLSTM arXiv:1811.02566)],
  [语音识别], [`E1_qlstm`], [已实现，待训练],

  [Chen et al., _Quantum Long Short-Term Memory_],
  [量子机器学习], [—], [不适用：量子而非四元数],

  [Cruces & Arini, _A novel method for cardiac vector velocity measurement_, BSPC 2016],
  [心电向量速度指标], [Exp 5 基准 B], [未实现],

  [Cruces et al., _Dynamic features of cardiac vector as alternative markers of drug-induced spatial dispersion_, JPTM 2020],
  [心电向量动态生物标志], [Exp 5 基准 C], [未实现],
)

四元数网络三篇均为跨域改编，保留其算子结构并适配为一维 $(B, 4Q, T)$ 序列，非原 pipeline 复现。

= Experiment 1 结果

R + L + Q 三分支，Q 固定为旋转四元数；预处理、数据划分、融合、分类头与训练配方完全一致，仅时间编码器不同。四元数编码器只替换 Q 分支算子，R、L 分支使用配对的通用编码器。seed 42。

#table(
  columns: 12,
  align: (left, left, right, right, right, right, right, right, right, right, right, right),
  table.header(
    [编码器], [类型], [参数量], [感受野],
    [Val\ AUROC], [Val\ AUPRC], [Test\ AUROC], [Test\ AUPRC],
    [NORM], [MI], [STTC], [CD],
  ),
  [QGNN], [四元数], [183,419], [2640 ms], [*0.9146*], [*0.7826*], [0.9032], [0.7693], [0.9337], [0.8879], [0.9228], [0.8912],
  [QTCN], [四元数], [245,811], [2640 ms], [0.9121], [0.7816], [0.8988], [0.7612], [0.9284], [0.8846], [0.9165], [0.8809],
  [LSTM + Attention], [通用], [62,617], [全局], [0.9117], [0.7784], [0.9025], [0.7671], [0.9334], [0.8846], [0.9185], [0.8810],
  [GRU], [通用], [44,665], [全局], [0.9116], [0.7785], [*0.9044*], [*0.7717*], [0.9300], [0.8868], [0.9219], [0.8972],
  [TCN], [通用], [244,513], [2640 ms], [0.9095], [0.7769], [0.8977], [0.7574], [0.9278], [0.8778], [0.9196], [0.8849],
  [LSTM], [通用], [56,545], [全局], [0.9095], [0.7767], [0.8998], [0.7649], [0.9300], [0.8761], [0.9214], [0.8818],
  [TCN + Attention], [通用], [250,585], [2640 ms], [0.9091], [0.7761], [0.9031], [0.7657], [0.9334], [0.8845], [0.9207], [0.8942],
  [Transformer], [通用], [217,189], [全局], [0.8847], [0.7213], [0.8722], [0.7105], [0.9087], [0.8657], [0.8949], [0.8234],
  [Q-Transformer], [四元数], [224,867], [全局], [0.8774], [0.7200], [0.8682], [0.7008], [0.9032], [0.8610], [0.8777], [0.8288],
  [QLSTM], [四元数], [58,019], [全局], [—], [—], [—], [—], [—], [—], [—], [—],
)

HYP 逐类 AUROC：QGNN 0.8806，QTCN 0.8838，LSTM + Attention 0.8951，GRU 0.8862，TCN 0.8785，LSTM 0.8899，TCN + Attention 0.8827，Transformer 0.8684，Q-Transformer 0.8703。

四元数与配对通用编码器之差（验证集 / 测试集 Macro AUROC）：QTCN − TCN 为 $+0.0026$ / $+0.0011$；QGNN − TCN 为 $+0.0051$ / $+0.0055$；Q-Transformer − Transformer 为 $-0.0073$ / $-0.0040$。

= 分析与结论

九个已完成模型使用完全相同的 R/L/Q 表示与下游结构，仅时间编码器不同，因此差异可归因于时间建模方式本身。结果显示排名前七的编码器验证集 Macro AUROC 集中在 0.9091–0.9146，跨度仅 0.0055，不足本项目既有 seed 波动带（0.005）的两倍，说明在此任务上时间架构的选择并非性能瓶颈；唯一明显落后的是 Transformer 与 Q-Transformer（0.8847 / 0.8774，低于首位约 3 个百分点），与 125 步序列、1.7 万条训练样本的规模下自注意力难以充分训练相符。四元数编码器与其配对通用编码器的三组对比呈两正一负：QTCN 较 TCN 高 0.0026、QGNN 较 TCN 高 0.0051、Q-Transformer 较 Transformer 低 0.0073，其中仅 QGNN 在验证集与测试集上均超出噪声带，但该对照是三组中最弱的一组——QGNN 没有对应的实数图网络，以 TCN 作为配对基准，二者在算子形式与参数量（183k 对 245k）上均不相同，故不能据此断言四元数算子有效，只能说该架构表现居前。此外验证集与测试集的排名并不一致（测试集首位为 GRU，验证集首位为 QGNN），且九个 run 分布在三种不同 GPU 上，而验证集前两名与其配对基准恰好来自不同硬件，在前七名仅差 0.0055 的前提下，cuDNN 算法选择造成的千分位差异足以翻转该排名。因此当前尚不足以确定 E\*：应先补齐 QLSTM 以完成最干净的 LSTM/QLSTM 配对，并至少将 TCN、QTCN、QGNN 置于同一硬件重跑，方可依「验证集 Macro AUROC → 验证集 Macro AUPRC → 参数量更少」的规则作出选择。
