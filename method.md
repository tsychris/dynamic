# Method

This section describes the training procedure implemented in `train_kitti.py` and its supporting modules (`kitti_dataloader.py`, `lpr_models.py`, `occlusion_generator.py`). The goal is **LiDAR place recognition** under **adversarial, geometry-aware dynamic occlusion**: a descriptor \(f\) is trained to remain discriminative while an occlusion generator \(G\) seeks perturbations that degrade place-wise metric learning.

---

## Problem formulation

Let \(x \in \mathbb{R}^{N \times C}\) denote a fixed-size point cloud (submap) with \(N\) points and \(C \in \{3,4\}\) channels (XYZ, optionally intensity). Given a batch with externally defined positive and negative pairs (from a query graph), we use a **batch-hard triplet** objective \(\mathcal{L}_{\mathrm{place}}\) on \(\ell_2\) distances in embedding space.

Training follows a **min–max** game:

\[
\min_f \max_G \ \mathcal{L}_{\mathrm{place}}\bigl(f(G(x))\bigr),
\]

with structured perturbations \(G(x)\) that remove (and optionally replace) points according to learned vehicle-like cuboids, rather than unstructured random dropout.

In practice, the implementation adds **descriptor-side** terms on both clean and occluded inputs plus an **embedding consistency** regularizer, as detailed below.

---

## Dataset and supervision

**Source.** Queries are loaded from a pickle file (`KITTIPointCloudQueryDataset`), where each query index maps to a record containing a submap path (e.g., `query_submap` / `submap_path` / `query`) and precomputed **positive** and **negative** query indices (stored as bitarrays when `bitarray` is available for efficiency).

**Point clouds.** KITTI `.bin` scans are read as float32; each point uses XYZ and optionally the fourth reflectance channel. Paths are resolved against a primary `kitti_root` with fallback to `fallback_root` for relocated datasets.

**Resampling.** Each cloud is subsampled or padded to exactly \(N\) points (default \(N{=}4096\)): random subset without replacement when over budget, repetition-based padding when under budget.

**Batch construction (`KITTIPairBatchSampler`).** Mini-batches are built so that a substantial fraction of samples form **anchor–positive pairs** drawn from the query graph, ensuring that the batch-hard triplet loss has valid positives. Remaining slots are filled with random indices and the full batch is shuffled.

**Collate function.** For batch indices with global query keys \(\{q_i\}\), the collate builds Boolean matrices \(\mathbf{P}, \mathbf{N} \in \{0,1\}^{B \times B}\) where \(P_{ij}=1\) iff query \(q_i\) lists query \(q_j\) as positive (and similarly for negatives). These masks generalize “same label” triplet mining to graph-based place supervision.

---

## Descriptor backbone \(f\)

The descriptor is selected via `build_descriptor_model`:

- **`pointnetvlad` (default):** PointNet-style local embedding with NetVLAD aggregation, \(\ell_2\)-normalized global descriptor of dimension \(d\) (default \(d{=}256\)).
- **`pointnet`:** Lightweight PointNet + MLP, max pooling, normalized embedding.
- **`dgcnn_vlad` / `dgcnnvlad`:** DGCNN local features followed by the same NetVLAD head.

Input channel count matches the dataset (3 or 4 with intensity).

---

## Occlusion generator \(G\)

**Architecture (`AdversarialOcclusionGenerator`).** Per batch element, point XYZ is encoded with a small MLP; a global feature is obtained by max pooling over points and concatenated with the **fraction of active boxes** (see below). A second MLP conditions on this vector and predicts \(M\) cuboids, each parameterized by center, size, and yaw (7 scalars per box after decoding). Default \(M{=}10\); with fixed vehicle dimensions, sizes can be held at a nominal \((l,w,h)\) while centers and yaws are learned.

**Active boxes.** For each sample, an integer \(k \sim \mathcal{U}\{1,\ldots,M\}\) determines how many of the \(M\) predicted boxes are active; a random subset of exactly \(k\) boxes is selected. Only active boxes contribute to occlusion.

**Occlusion geometry.** For each box, a soft occlusion score combines (i) **inside-cuboid** membership and (ii) an approximate **angular shadow cone**: points that lie in the sensor-centered angular neighborhood of the box and are farther along the ray than the box are treated as occluded. Per-point scores over active boxes are fused by max pooling, then clipped to \([0,1]\).

**Straight-through threshold.** A hard drop mask is obtained by thresholding scores at \(0.5\); gradients for the generator use a **straight-through** estimator so the forward pass is hard while backward signals follow the soft scores.

**Differentiable vs. hard perturbation.**

- **Generator update:** `apply_soft_drop` scales coordinates (and extra channels) by \((1 - \text{soft mask})\), keeping the operation differentiable through the soft mask path used in that step.
- **Descriptor update:** Under `torch.no_grad()`, the generator is run to produce a **hard** mask; `apply_hard_drop_and_insert` zeros dropped point slots and, if enabled, **inserts** synthetic returns by placing points sampled on the surfaces of active boxes into freed indices (optional object insertion).

**Regularization on \(G\).** To keep cuboids plausible, the generator is penalized with (weighted) terms: **size prior** (when sizes are not fixed), **height prior** favoring ground-appropriate vertical placement, and **range prior** discouraging centers beyond a nominal horizontal range. Weights are hyperparameters (defaults: size \(0.1\), height \(0.05\), range \(0.05\)).

---

## Loss functions

**Batch-hard triplet with external masks (`masked_batch_hard_triplet_loss`).** For embeddings \(\{\mathbf{e}_i\}_{i=1}^B\) and margin \(m\), pairwise Euclidean distances \(d_{ij}=\|\mathbf{e}_i-\mathbf{e}_j\|_2\) are computed. For each anchor \(i\) with at least one positive and one negative in the batch mask, the loss uses the **hardest positive** \(\max_{j: P_{ij}=1} d_{ij}\) and **hardest negative** \(\min_{j: N_{ij}=1} d_{ij}\), averaged over valid anchors:

\[
\mathcal{L}_{\mathrm{place}} = \mathbb{E}_i \left[ \max\bigl(0,\ d_{i,p^*(i)} - d_{i,n^*(i)} + m \bigr) \right].
\]

**Embedding consistency.**

\[
\mathcal{L}_{\mathrm{cons}} = \frac{1}{B}\sum_{i=1}^{B} \bigl(1 - \cos(\mathbf{e}^{\mathrm{clean}}_i, \mathbf{e}^{\mathrm{adv}}_i)\bigr),
\]

implemented as mean \(1 - \) cosine similarity between \(\ell_2\)-normalized descriptors of clean and occluded point clouds.

---

## Optimization schedule

Each training iteration performs **two alternating steps** (both on the same batch):

1. **Update \(G\) (maximize place loss on occluded data).** Freeze \(f\), enable gradients on \(G\). Forward uses **soft drop** (no insertion in this step). Objective:
   \[
   \mathcal{L}_G = -\mathcal{L}_{\mathrm{place}}\bigl(f(G_{\mathrm{soft}}(x))\bigr) + \lambda_{\mathrm{reg}}\,\mathcal{R}(G),
   \]
   where \(\mathcal{R}\) aggregates the size, height, and range priors with their respective weights.

2. **Update \(f\) (minimize robust recognition).** Freeze \(G\), enable gradients on \(f\). Re-run \(G\) without gradients with **hard drop** and optional **insertion** to form \(x'\). Then:
   \[
   \mathcal{L}_f = \mathcal{L}_{\mathrm{place}}(f(x)) + \lambda_{\mathrm{adv}}\,\mathcal{L}_{\mathrm{place}}(f(x')) + \lambda_{\mathrm{cons}}\,\mathcal{L}_{\mathrm{cons}}(f(x), f(x')).
   \]

Both networks use **Adam** with separate learning rates (defaults \(10^{-3}\) for \(f\) and \(G\)). Default hyperparameters include triplet margin \(m=0.2\), \(\lambda_{\mathrm{adv}}=1.0\), \(\lambda_{\mathrm{cons}}=0.2\), and \(20\) epochs with batch size \(32\) and a fixed number of batches per epoch (\(200\) by default).

---

## Logging, checkpoints, and optional visualization

Training logs scalars and histograms to TensorBoard (losses, occlusion fraction, active box counts, box parameters). Checkpoints store the descriptor, generator, optimizers, and arguments. If enabled, a **range-view** visualization projects clean vs. adversarial points to a front depth map (using KITTI calibration when available) and saves comparison figures for inspection.

---

## Summary

The pipeline implements **graph-supervised metric learning** on KITTI submaps with a **structured adversarial occlusion generator** that learns cuboid parameters and a min–max game **stabilized** by realism priors on \(G\) and by **joint clean/adversarial triplet loss** plus **embedding consistency** on \(f\), using alternating optimization and straight-through hard masks for end-to-end training.

---

# 方法（中文版）

本章对应实现代码 `train_kitti.py` 及其依赖模块 `kitti_dataloader.py`、`lpr_models.py`、`occlusion_generator.py`，在不改变上文英文表述的前提下，以下采用中文学术论文常用体例，对**双分支训练架构**、**交替优化流程**以及**对抗式遮挡生成器的思想与网络结构**作系统说明。

## 3.1 问题描述与总体框架

设激光雷达子图经固定规模采样后为点云 \(\boldsymbol{x} \in \mathbb{R}^{N \times C}\)，其中 \(N\) 为点数，\(C \in \{3,4\}\) 分别对应仅三维坐标或坐标与强度。位姿检索任务在嵌入空间中进行度量学习：给定由查询图（query graph）预先给定的正样本集合与负样本集合，采用 batch-hard 三元组损失 \(\mathcal{L}_{\mathrm{place}}\)，在 \(\ell_2\) 距离下挖掘困难样本对。

本文训练框架包含两个可学习模块：（1）**全局描述子网络** \(f\)，将点云映射为 \(\ell_2\) 归一化的全局描述向量；（2）**结构化遮挡生成器** \(G\)，在三维空间中预测若干车辆近似长方体，并据此产生点级遮挡掩码，对 \(\boldsymbol{x}\) 进行删点及可选的伪回波插入。与无结构随机丢点不同，\(G\) 的扰动受几何约束，用于模拟动态目标引起的遮挡效应。

理想化的博弈形式可写为

\[
\min_{f}\max_{G}\ \mathcal{L}_{\mathrm{place}}\bigl(f(G(\boldsymbol{x}))\bigr).
\]

实际实现中，为稳定训练并在遮挡存在时仍保持判别力，对 \(f\) 同时施加**干净分支**与**对抗分支**上的度量损失，并引入**嵌入一致性**正则项；对 \(G\) 则在最大化对抗目标的同时施加**尺度、高度与距离先验**，详见 3.5–3.6 节。

## 3.2 数据集构建与监督形式

训练样本来自 KITTI 场景下预构建的查询 pickle：`KITTIPointCloudQueryDataset` 将每个查询索引映射为子图路径及与其几何或拓扑相关的**正查询索引**与**负查询索引**。点云自 `.bin` 读取为 float32；路径在 `kitti_root` 与 `fallback_root` 之间解析，以适应数据存放位置差异。

对每个查询，子图随机或确定性重采样至固定 \(N\) 点：点数过多时无放回随机下采样，不足时采用有放回补齐，以保证网络输入张量形状一致。

批构造采用 `KITTIPairBatchSampler`：在每个 mini-batch 中优先采样若干 **锚点–正样本对**，以保证 batch 内存在有效正样本，从而 batch-hard 三元组损失良定；其余位置由随机查询填充，并对整批索引打乱，以增加负样本多样性。

批整理函数根据批内全局查询键 \(\{q_i\}_{i=1}^{B}\) 构造布尔矩阵 \(\mathbf{P},\mathbf{N}\in\{0,1\}^{B\times B}\)，其中 \(P_{ij}=1\) 当且仅当在查询 \(q_i\) 的监督定义下 \(q_j\) 为其正样本，\(N_{ij}\) 同理对应负样本。该形式将基于类别标签的三元组挖掘推广为**基于查询图的度量监督**。

## 3.3 描述子分支（全局嵌入网络 \(f\)）

描述子分支由 `build_descriptor_model` 实例化，与遮挡生成器**参数不共享**，单独使用优化器 \(\mathrm{Adam}(\cdot;\eta_f)\)。可选结构包括：

- **PointNetVLAD（默认）**：PointNet 式逐点嵌入经最大池化得到全局特征，再经 NetVLAD 聚合成固定维描述子，最后 \(\ell_2\) 归一化；
- **PointNet**：轻量 MLP 逐点编码与最大池化；
- **DGCNN+VLAD**：DGCNN 提取局部几何特征后接同一 NetVLAD 头。

输入通道数与数据一致（3 或 4）。该分支在训练的一个子步中更新，在另一子步中冻结梯度，以实现与 \(G\) 的交替优化（见 3.6 节）。

## 3.4 对抗式遮挡生成器分支（\(G\)）：对抗思想与网络结构

### 3.4.1 对抗思想

生成器 \(G\) 与描述子 \(f\) 构成**零和式对抗关系**的变体：在给定 batch 与查询图监督掩码的条件下，\(G\) 倾向于产生使 \(\mathcal{L}_{\mathrm{place}}\) **增大**的遮挡（即恶化 batch-hard 度量学习）；而 \(f\) 则通过最小化干净样本与对抗样本上的 \(\mathcal{L}_{\mathrm{place}}\)，并拉近同一子图在遮挡前后的嵌入，从而学习对结构化遮挡**不敏感**且具有判别力的表示。

因此，训练并非单次最小化，而是在每个迭代内对**同一批数据**先后执行：（i）固定 \(f\)、更新 \(G\) 以增大对抗路径上的度量损失（并附带先验惩罚）；（ii）固定 \(G\)、更新 \(f\) 以减小多目标组合损失。该过程属于**交替梯度下降/上升**，两模块各有一套参数与优化器，形成清晰的**双分支、双优化器**架构。

### 3.4.2 网络结构（`AdversarialOcclusionGenerator`）

生成器为**条件于 batch 与随机激活策略**的编码–解码式结构，流程如下。

**（1）逐点编码。** 对 \(\boldsymbol{x}\) 的坐标通道 \(\mathbb{R}^{N\times 3}\) 施加多层感知机 `point_mlp`（线性层–ReLU 堆叠），得到逐点特征 \(\mathbf{H}\in\mathbb{R}^{N\times F}\)。

**（2）场景级条件向量。** 对 \(\mathbf{H}\) 在点维做**最大池化**，得到全局向量 \(\mathbf{g}\in\mathbb{R}^{F}\)。与此同时，对每个样本随机采样**激活长方体个数** \(k\in\{1,\ldots,M\}\)（默认 \(M=10\)），并随机选取 \(k\) 个长方体为“激活”，其余不参与遮挡融合。将激活比例 \(k/M\) 标量与 \(\mathbf{g}\) 拼接，输入 `global_mlp`，得到条件特征 \(\mathbf{c}\in\mathbb{R}^{128}\)。该设计使生成器在**不同遮挡强度**（激活长方体数目）下具备可区分的行为，并与随机激活策略一致。

**（3）长方体参数预测。** 线性层 `box_head` 将 \(\mathbf{c}\) 映射为 \(M\) 组原始参数，每组 7 维，经解码得到各长方体的**中心** \(\boldsymbol{\mu}\)、**尺寸** \((l,w,h)\) 与**绕竖直轴转角** \(\psi\)。默认可采用**固定车辆尺寸**，此时尺寸为常数，网络主要学习位置与朝向；亦可学习尺寸并在先验项约束下逼近典型车辆尺度。

**（4）几何遮挡得分与融合。** 对每个长方体，逐点计算软遮挡分数：一项刻画点是否落在长方体内部；另一项在传感器-centric 球坐标下近似**角向阴影锥**——当点与长方体中心方向一致且沿径向更远时，视为被该长方体遮挡。各长方体得分在**激活掩码**下屏蔽非激活长方体后，在长方体维做**最大值融合**，得到每点单一几何得分，并截断至 \([0,1]\)。

**（5）硬掩码与直通估计器。** 前向传播中对比值 0.5 二值化得到**硬删点掩码**，以满足“删点/插点”的离散语义；反向传播中对生成器采用 **straight-through** 技巧：前向使用硬掩码，梯度回传仍沿软得分路径，从而使长方体参数可微优化。

**（6）与描述子更新相衔接的两套前向。** 更新 \(G\) 时，采用**软删点**（坐标按 \((1-\text{软掩码})\) 缩放），整条路径对 \(G\) 可微，且该步**关闭物体插入**以简化对抗梯度。更新 \(f\) 时，在 `torch.no_grad()` 下重算 \(G\) 得到**硬掩码**，执行硬置零删点，并可选择将激活长方体表面采样的点**插入**被清空位置，以模拟动态物体回波。两套前向体现**同一生成器、不同扰动实现**：前者服务 \(\max_G\)，后者服务 \(\min_f\) 下的真实离散遮挡分布近似。

**（7）生成器正则。** 为抑制不合理长方体，引入尺寸先验（可学习尺寸时）、中心高度先验（贴近路面高度带）及平面径向距离先验（避免中心过度外漂），加权后并入 \(G\) 的损失，与对抗项形成折中。

## 3.5 损失函数

**（1）基于外部掩码的 batch-hard 三元组损失。** 设批内嵌入为 \(\{\mathbf{e}_i\}\)，两两欧氏距离 \(d_{ij}=\lVert\mathbf{e}_i-\mathbf{e}_j\rVert_2\)。对每个存在至少一个正样本与一个负样本的锚点 \(i\)，取最难正样本距离与最难负样本距离，在间隔 \(m\) 下构造 hinge 损失并平均，记为 \(\mathcal{L}_{\mathrm{place}}\)。式中与常规 batch-hard 的区别在于正、负由矩阵 \(\mathbf{P},\mathbf{N}\) 指定，而非仅靠类别标签相等。

**（2）嵌入一致性损失。** 对同一子图的干净嵌入 \(\mathbf{e}^{\mathrm{clean}}\) 与对抗嵌入 \(\mathbf{e}^{\mathrm{adv}}\)，采用

\[
\mathcal{L}_{\mathrm{cons}}=\frac{1}{B}\sum_{i=1}^{B}\bigl(1-\cos(\mathbf{e}^{\mathrm{clean}}_i,\mathbf{e}^{\mathrm{adv}}_i)\bigr),
\]

即最小化 \(1\) 减余弦相似度，促使遮挡前后全局描述方向一致。

## 3.6 交替训练流程与双分支协作机制

每个训练迭代对当前 mini-batch \(\{\boldsymbol{x}_b\}\) 顺序执行下列两步（数据相同、梯度角色不同）。

**步骤一：生成器分支上升步。** 冻结 \(f\) 的全部参数（`requires_grad=False`），启用 \(G\)。前向为软删点得到 \(\tilde{\boldsymbol{x}}=G_{\mathrm{soft}}(\boldsymbol{x})\)，计算 \(\mathcal{L}_{\mathrm{place}}(f(\tilde{\boldsymbol{x}}))\)。由于 \(f\) 无梯度，对 \(G\) 的优化目标取

\[
\mathcal{L}_G=-\mathcal{L}_{\mathrm{place}}\bigl(f(G_{\mathrm{soft}}(\boldsymbol{x}))\bigr)+\lambda_{\mathrm{reg}}\mathcal{R}(G),
\]

即对 \(\mathcal{L}_{\mathrm{place}}\) 做梯度上升等价于最小化 \(\mathcal{L}_G\)。随后仅对 \(G\) 执行 `backward` 与 `optimizer_g.step()`。

**步骤二：描述子分支下降步。** 冻结 \(G\)，启用 \(f\)。在无梯度条件下重算 \(G\) 得到硬掩码与可选插入点，构造对抗样本 \(\boldsymbol{x}'\)。再分别计算 \(\mathcal{L}_{\mathrm{place}}(f(\boldsymbol{x}))\)、\(\mathcal{L}_{\mathrm{place}}(f(\boldsymbol{x}'))\) 以及 \(\mathcal{L}_{\mathrm{cons}}(f(\boldsymbol{x}),f(\boldsymbol{x}'))\)，组合为

\[
\mathcal{L}_f=\mathcal{L}_{\mathrm{place}}(f(\boldsymbol{x}))+\lambda_{\mathrm{adv}}\mathcal{L}_{\mathrm{place}}(f(\boldsymbol{x}'))+\lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}},
\]

对 \(f\) 反向传播并 `optimizer_f.step()`。

上述两步构成**单迭代内的交替优化**：两分支不共享优化器，通过冻结对方参数避免同一迭代内相互“拉扯”；软/硬两种扰动形式分别适配 \(\partial \mathcal{L}/\partial G\) 与 \(\partial \mathcal{L}/\partial f\) 的可行性与稳定性。默认采用 Adam，\(f\) 与 \(G\) 学习率均可独立设定；实现中典型超参为 \(m=0.2\)，\(\lambda_{\mathrm{adv}}=1.0\)，\(\lambda_{\mathrm{cons}}=0.2\)，以及各先验权重如代码 `TrainConfig` 所示。

## 3.7 实现细节与辅助功能

训练过程可将损失、遮挡比例、激活长方体数目及长方体几何统计写入 TensorBoard；按周期保存描述子与生成器权重及优化器状态。可选开启前视图深度投影可视化，将干净点云与对抗点云投影至图像平面以便定性分析，其依赖 KITTI 标定外参与内参解析。

## 3.8 小结

本文方法在 KITTI 子图与查询图监督下，构建**描述子分支与对抗遮挡生成器分支**并行的双模块架构，通过**单迭代内交替更新**实现 \(\min_f\max_G\) 思想的实用化近似；遮挡生成器以**点云编码–全局条件–长方体解码–几何得分–直通二值化**为脉络，结合随机激活长方体数目与软硬两套前向，在保持端到端可训练的同时逼近离散删点过程。描述子则在干净与对抗双路径度量损失及嵌入一致性共同作用下，学习对结构化动态遮挡鲁棒的位姿检索表示。
