# 面向动态遮挡鲁棒性的激光雷达位置识别对抗式训练方法

学校名称：XXXX大学  
学院名称：XXXX学院  
学科专业：XXXX  
研究方向：自动驾驶感知与定位  
作者姓名：XXX  
指导教师：XXX 教授  
完成日期：XXXX年XX月

---

## 摘要

激光雷达位置识别（LiDAR Place Recognition, LPR）旨在通过当前观测点云与历史地图点云之间的相似性匹配，实现车辆所在位置的快速检索，是自动驾驶重定位与闭环检测中的关键技术之一。现有基于深度学习的点云位置识别方法通常通过全局描述符对场景进行表征，并在特征空间中完成检索。然而，在真实道路环境中，车辆、行人及其他动态目标会对激光点云造成局部遮挡、几何缺失和回波干扰，使得同一地点在不同时间采集到的点云存在显著观测差异，从而导致描述符稳定性下降并影响检索性能。

针对上述问题，本文提出一种面向动态遮挡鲁棒性的对抗式训练方法。该方法在激光雷达位置识别框架中引入动态遮挡生成器，通过对输入点云预测若干具有车辆形态先验的三维长方体遮挡体，对原始点云施加结构化扰动，以模拟真实动态场景中的遮挡现象。在训练过程中，生成器以最大化位置识别损失为目标，主动搜索对描述符网络最不利的遮挡模式；描述符网络则同时利用干净样本监督、对抗样本监督以及表示一致性约束进行优化，从而提高模型在动态遮挡条件下的鲁棒性。

与传统随机点丢弃增强方式相比，本文方法生成的扰动具有更强的几何结构性和场景合理性，更符合车辆等动态目标在激光雷达观测中的遮挡规律。本文进一步从问题定义、损失函数设计及交替优化策略三个方面对该方法进行了系统建模，并给出了适用于 KITTI 数据集的实验设置与评价方案。该方法可作为研究动态场景下点云位置识别鲁棒性的重要训练框架，为后续开展不同描述符骨干网络对比实验和遮挡生成策略分析提供统一基础。

**关键词：** 激光雷达位置识别；动态遮挡；对抗训练；全局描述符；点云鲁棒性

---

## Abstract

LiDAR Place Recognition (LPR) aims to retrieve the corresponding historical place from a large-scale point cloud map using the current LiDAR observation, and it plays an important role in relocalization and loop closure detection for autonomous driving. Existing deep learning based LPR methods usually encode each point cloud into a global descriptor and perform place retrieval in the embedding space. However, in real traffic environments, dynamic objects such as vehicles and pedestrians often introduce partial occlusion, geometric incompleteness, and return interference, which significantly change the observation of the same place at different times and thus degrade descriptor stability.

To address this issue, this thesis proposes an adversarial training framework for dynamic-occlusion-robust LiDAR place recognition. A dynamic occlusion generator is introduced into the LPR pipeline to predict multiple vehicle-shaped 3D cuboid occluders for each input point cloud. These structured occlusions are then used to perturb the original point cloud so as to simulate realistic dynamic scene interference. During training, the generator is optimized to maximize the place recognition loss under geometric priors, while the descriptor network is jointly optimized with clean-sample supervision, adversarial-sample supervision, and representation consistency regularization.

Compared with conventional random point dropout augmentation, the proposed method produces more structured and physically plausible perturbations, which better match real occlusion patterns caused by dynamic objects in road scenes. The proposed framework is systematically formulated from the perspectives of problem definition, loss design, and alternating optimization, and can serve as a unified basis for further experiments on different descriptor backbones and occlusion strategies on the KITTI dataset.

**Key Words:** LiDAR place recognition; dynamic occlusion; adversarial training; global descriptor; point cloud robustness

---

## 第一章 绪论

### 1.1 研究背景与意义

随着自动驾驶和移动机器人技术的快速发展，复杂环境中的高精度定位问题受到了广泛关注。在 GNSS 信号受限或失效的场景中，基于环境感知信息的位置识别与重定位方法成为保障系统稳定运行的重要手段。激光雷达因其对光照变化不敏感、距离测量精度高、几何信息丰富等优势，被广泛应用于位置识别任务中。

激光雷达位置识别的核心思想是将当前观测点云编码为全局描述符，并在数据库中检索与其最相似的历史场景。然而，实际道路环境并非静态，车辆、行人、非机动车以及临时停放障碍物会在不同采集时刻改变观测视野，使得同一地点的点云在局部区域出现遮挡、缺失或虚假结构。若描述符网络在训练过程中主要依赖静态场景结构，则其在动态环境下往往表现出较差的泛化能力。因此，研究面向动态遮挡鲁棒性的点云位置识别方法具有重要理论意义与工程应用价值。

### 1.2 国内外研究现状

现有激光雷达位置识别方法主要包括基于手工特征的方法和基于深度学习的方法。传统方法依赖局部几何结构、扫描上下文或统计描述子进行场景匹配，在一定程度上具备可解释性，但对复杂动态环境适应性有限。近年来，基于 PointNetVLAD、MinkLoc、DGCNN 等网络结构的深度学习方法逐渐成为主流，它们通过端到端训练获得更具判别性的全局描述符，在多个公开数据集上取得了良好性能。

尽管如此，现有多数方法默认训练样本主要来源于相对干净或静态的场景分布，对于动态遮挡问题的专门建模仍然不足。已有鲁棒性增强手段通常包括随机点丢弃、随机旋转、坐标扰动等通用数据增强策略，但这些方法往往难以逼近真实道路场景中的动态遮挡模式。特别是随机丢点虽然可以在一定程度上提高网络对稀疏采样的适应性，却无法表达车辆遮挡所具有的明显空间连续性和几何结构性。

从对抗训练角度看，生成式扰动方法能够主动构造对模型最具挑战性的样本，已在图像分类、三维识别等任务中展现出提高鲁棒性的潜力。然而，直接将逐点噪声扰动迁移到点云位置识别任务中并不理想，因为其容易破坏物理合理性，且难以与真实动态物体形成对应关系。因此，如何设计一种具有几何先验约束、可用于模拟动态目标遮挡的结构化生成器，是当前值得深入研究的问题。

### 1.3 研究内容

围绕动态场景下激光雷达位置识别鲁棒性不足的问题，本文的主要研究内容如下。

1. 针对随机点丢弃难以反映真实动态遮挡结构的问题，设计一种面向点云场景的动态遮挡生成器，以车辆形态近似建模动态障碍物，并通过结构化点删除或物体插入实现受扰样本构造。
2. 构建描述符网络与遮挡生成器的联合训练框架，将动态遮挡生成过程纳入位置识别训练环节，通过极小极大目标形成对抗式优化机制。
3. 设计适用于上述训练框架的损失函数体系，包括生成器对抗损失、几何先验正则项、干净样本位置识别损失、受扰样本位置识别损失及表示一致性损失。
4. 给出基于 KITTI 数据集的实验评价流程，支持 clean query-clean db、dirty query-clean db 以及 dirty query-dirty db 等多种测试设定，用于分析模型在不同动态扰动条件下的性能变化。

### 1.4 论文组织结构

本文后续内容安排如下：第二章对所提出的面向动态遮挡鲁棒性的激光雷达位置识别方法进行详细描述；第三章介绍实验设置、评价指标及对比方案；第四章总结全文工作并对后续研究方向进行展望。

---

## 第二章 面向动态遮挡鲁棒性的对抗式位置识别方法

### 2.1 问题定义

设训练批次记为

$$
\mathcal{B}=\left\{\mathbf{x}_i\right\}_{i=1}^{B},
$$

其中，$\mathbf{x}_i\in\mathbb{R}^{N\times C}$ 表示第 $i$ 个点云样本，$N$ 表示点数，$C$ 表示每个点的特征维度。记描述符网络为 $f_{\theta}(\cdot)$，其参数为 $\theta$；记动态遮挡生成器为 $G_{\phi}(\cdot)$，其参数为 $\phi$。本文希望通过生成器对输入点云施加结构化遮挡变换 $T_{\phi}(\cdot)$，并在此基础上训练鲁棒的点云位置识别描述符。整体目标可表示为

$$
\min_{\theta}\max_{\phi}\ \mathcal{L}_{\mathrm{place}}\left(f_{\theta}\left(T_{\phi}(\mathbf{x})\right)\right).
$$

该目标表明，生成器试图寻找能够最大程度破坏位置识别性能的遮挡模式，而描述符网络则学习在此类困难样本上保持稳定判别能力。

### 2.2 位置识别描述符学习

对于输入点云 $\mathbf{x}_i$，描述符网络输出其全局嵌入表示

$$
\mathbf{z}_i=f_{\theta}(\mathbf{x}_i)\in\mathbb{R}^{D},
$$

其中，$D$ 为描述符维度。对于任意样本对 $(i,j)$，其在嵌入空间中的欧氏距离定义为

$$
d_{ij}=\left\|\mathbf{z}_i-\mathbf{z}_j\right\|_2.
$$

设锚样本 $i$ 的正样本集合与负样本集合分别为 $\mathcal{P}(i)$ 和 $\mathcal{N}(i)$。本文采用批次内困难三元组损失作为基础位置识别损失：

$$
\mathcal{L}_{\mathrm{place}}
=
\frac{1}{|\mathcal{V}|}
\sum_{i\in\mathcal{V}}
\left[
\max_{p\in\mathcal{P}(i)}d_{ip}
-
\min_{n\in\mathcal{N}(i)}d_{in}
+\delta
\right]_+,
$$

其中，$\delta$ 表示间隔超参数，$\mathcal{V}$ 表示当前批次中同时具有至少一个正样本和至少一个负样本的有效锚样本集合。

### 2.3 动态遮挡生成器建模

对于输入点云 $\mathbf{x}_i$，生成器预测 $M$ 个候选遮挡体：

$$
G_{\phi}(\mathbf{x}_i)\rightarrow
\left\{
\left(\mathbf{c}_{i,m},\mathbf{s}_{i,m},\psi_{i,m}\right)
\right\}_{m=1}^{M},
$$

其中，$\mathbf{c}_{i,m}\in\mathbb{R}^{3}$、$\mathbf{s}_{i,m}\in\mathbb{R}^{3}$ 和 $\psi_{i,m}$ 分别表示第 $m$ 个长方体遮挡体的中心、尺寸与偏航角。每个长方体可视为对车辆等动态目标的近似建模。

训练过程中，对每个样本随机采样一个激活遮挡体数量 $k_i\in\{1,\dots,M\}$，仅保留其中 $k_i$ 个遮挡体参与当前样本的遮挡构造，以模拟不同程度的动态干扰。为同时兼顾可导性和物理一致性，本文定义软遮挡与硬遮挡两种扰动形式：

$$
\tilde{\mathbf{x}}^{\,s}_i=T^{\mathrm{soft}}_{\phi}(\mathbf{x}_i),\qquad
\tilde{\mathbf{x}}^{\,h}_i=T^{\mathrm{hard}}_{\phi}(\mathbf{x}_i).
$$

其中，软遮挡变换主要用于生成器更新阶段，使梯度能够从位置识别损失反向传播至生成器参数；硬遮挡变换用于描述符网络更新阶段，使网络在更加接近真实离散遮挡结果的样本上进行训练。若启用物体插入机制，则可进一步在长方体表面插入合成点，用于模拟动态物体带来的激光回波。

### 2.4 生成器损失函数设计

在生成器更新阶段，固定描述符网络参数 $\theta$，仅优化生成器参数 $\phi$。为使生成器在保持几何合理性的同时尽可能构造困难遮挡样本，其损失函数定义为

$$
\mathcal{L}_{G}
=
-\mathcal{L}_{\mathrm{place}}
\left(
\left\{
f_{\theta}\left(\tilde{\mathbf{x}}^{\,s}_i\right)
\right\}_{i\in\mathcal{B}}
\right)
+
\lambda_{\mathrm{size}}\mathcal{L}_{\mathrm{size}}
+
\lambda_{\mathrm{height}}\mathcal{L}_{\mathrm{height}}
+
\lambda_{\mathrm{range}}\mathcal{L}_{\mathrm{range}}.
$$

其中，第一项前的负号表示最小化 $\mathcal{L}_{G}$ 等价于最大化受扰样本上的位置识别损失。后三项为几何正则项。

尺寸先验损失用于约束遮挡体尺寸接近典型车辆大小：

$$
\mathcal{L}_{\mathrm{size}}
=
\frac{1}{BM}\sum_{i=1}^{B}\sum_{m=1}^{M}
\mathrm{SmoothL1}\left(\mathbf{s}_{i,m},\mathbf{s}_{\mathrm{car}}\right).
$$

高度先验损失用于约束遮挡体中心靠近地面区域：

$$
\mathcal{L}_{\mathrm{height}}
=
\frac{1}{BM}\sum_{i=1}^{B}\sum_{m=1}^{M}
\left(c_{i,m}^{(z)}-z_0\right)^2,
$$

其中，$z_0$ 为地面附近的高度先验中心。

距离先验损失用于限制遮挡体分布范围：

$$
\mathcal{L}_{\mathrm{range}}
=
\frac{1}{BM}\sum_{i=1}^{B}\sum_{m=1}^{M}
\left[
\max\left(
0,\sqrt{\left(c_{i,m}^{(x)}\right)^2+\left(c_{i,m}^{(y)}\right)^2}-r_{\max}
\right)
\right]^2.
$$

通过上述设计，生成器并非无约束地制造任意扰动，而是在车辆几何先验与空间范围先验限制下，学习对位置识别最具挑战性的结构化遮挡模式。

### 2.5 描述符网络损失函数设计

在描述符网络更新阶段，固定生成器参数 $\phi$，重新生成硬遮挡点云，并分别计算干净样本与对抗样本的描述符：

$$
\mathbf{z}_i^{\,c}=f_{\theta}(\mathbf{x}_i),\qquad
\mathbf{z}_i^{\,a}=f_{\theta}\left(\tilde{\mathbf{x}}^{\,h}_i\right).
$$

描述符网络总损失定义为

$$
\mathcal{L}_{F}
=
\mathcal{L}_{\mathrm{clean}}
+
\lambda_{\mathrm{adv}}\mathcal{L}_{\mathrm{adv}}
+
\lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}.
$$

其中，干净样本损失和对抗样本损失分别为

$$
\mathcal{L}_{\mathrm{clean}}
=
\mathcal{L}_{\mathrm{place}}
\left(
\left\{
f_{\theta}(\mathbf{x}_i)
\right\}_{i\in\mathcal{B}}
\right),
$$

$$
\mathcal{L}_{\mathrm{adv}}
=
\mathcal{L}_{\mathrm{place}}
\left(
\left\{
f_{\theta}\left(\tilde{\mathbf{x}}^{\,h}_i\right)
\right\}_{i\in\mathcal{B}}
\right).
$$

为约束同一样本在干净观测和遮挡观测下具有一致的全局语义表示，引入余弦一致性损失：

$$
\mathcal{L}_{\mathrm{cons}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left(
1-
\frac{
\left\langle \mathbf{z}_i^{\,c},\mathbf{z}_i^{\,a}\right\rangle
}{
\left\|\mathbf{z}_i^{\,c}\right\|_2
\left\|\mathbf{z}_i^{\,a}\right\|_2
}
\right).
$$

该损失设计使得描述符网络既要保持原始点云上的位置判别能力，又要学习在结构化动态遮挡条件下提取稳定特征，同时避免同一地点的特征表示发生过度漂移。

### 2.6 交替优化训练策略

本文采用逐批次交替优化策略。在每次训练迭代中，首先固定描述符网络，依据生成器损失更新生成器参数：

$$
\phi\leftarrow\phi-\eta_G\nabla_{\phi}\mathcal{L}_G.
$$

随后固定生成器，依据描述符网络损失更新描述符网络参数：

$$
\theta\leftarrow\theta-\eta_F\nabla_{\theta}\mathcal{L}_F.
$$

其中，$\eta_G$ 和 $\eta_F$ 分别表示生成器与描述符网络的学习率。

上述过程可以理解为一个受几何先验约束的稳定对抗博弈。生成器持续搜索更具破坏性的动态遮挡模式，而描述符网络则通过干净监督、对抗监督和表示一致性约束逐步提高鲁棒性。与同时更新两个网络相比，交替优化能够更清晰地分离“攻击者”和“防御者”的优化目标，从而提高训练稳定性与收敛可控性。

---

## 第三章 实验设计与评价方案

### 3.1 数据集与数据组织

本文实验基于 KITTI 数据集开展。训练数据由查询帧及其对应的正负样本关系构成，测试阶段则进一步划分为数据库集（database）与查询集（query）。为了更准确评估模型在动态场景中的鲁棒性，测试阶段可分别构造以下几种设定：

1. clean query-clean db：查询点云与数据库点云均为原始点云。
2. dirty query-clean db：仅查询点云施加动态遮挡扰动，数据库保持干净。
3. dirty query-dirty db：查询点云与数据库点云均施加动态遮挡扰动。

上述测试设定能够分别反映模型在单侧动态干扰和双侧动态干扰条件下的性能变化。

### 3.2 评价指标

本文采用召回率（Recall）作为主要评价指标。对于每个查询点云，在数据库中检索最相似的前 $K$ 个结果。若其中至少存在一个真实匹配样本，则记为一次命中。Recall@$K$ 定义为：

$$
\mathrm{Recall@}K
=
\frac{\text{Top-}K\ \text{中命中的查询数}}
{\text{查询总数}}
\times 100\%.
$$

实验中通常报告 Recall@1、Recall@5、Recall@10 以及 Recall@1\% 等指标，以全面衡量位置识别模型的检索精度与鲁棒性。

### 3.3 对比实验设计

为验证所提方法的有效性，可设置如下对比模型：

1. PointNetVLAD baseline：仅训练原始位置识别网络，不引入动态遮挡生成器。
2. DGCNN+NetVLAD baseline：采用不同描述符骨干网络进行基线训练，用于分析骨干网络对结果的影响。
3. Descriptor + adversarial generator：在相同描述符骨干基础上引入本文所提动态遮挡生成器，进行交替对抗训练。

通过上述对比，可分析生成器对描述符鲁棒性的提升效果，以及不同骨干网络在结构化遮挡训练下的性能差异。

### 3.4 消融分析建议

为了进一步分析方法有效性，后续实验可从以下角度开展消融研究：

1. 不同遮挡预算比例对性能的影响，例如最大点丢弃比例为 10\%、20\%、30\%、40\% 和 50\%。
2. 不同激活遮挡体数量对性能的影响，例如固定 1 个、3 个、5 个或随机数量遮挡体。
3. 不同几何先验设置对生成器行为的影响，例如高度先验、距离先验和尺寸先验的变化。
4. 仅使用对抗样本损失、仅使用一致性损失以及联合使用时的性能对比。

---

## 第四章 结论与展望

本文围绕动态场景下激光雷达位置识别的鲁棒性问题，提出了一种基于结构化动态遮挡生成器的对抗式训练方法。该方法通过车辆形态先验对动态目标进行近似建模，以结构化点删除或物体插入的方式构造更符合真实场景的困难样本，并通过生成器与描述符网络的交替优化，提高了位置识别模型在动态遮挡条件下的稳定性。

本文方法的优势在于：其一，相较于随机点丢弃增强，结构化遮挡更符合真实道路环境中的动态干扰形式；其二，对抗式训练机制能够主动暴露描述符网络在困难样本上的脆弱性；其三，该框架对描述符骨干具有较好的兼容性，可与 PointNetVLAD、DGCNN+NetVLAD 等多种网络结合开展实验。

后续工作可进一步从以下几个方向展开：一是增强生成器对场景内容的条件感知能力，使遮挡体位置随输入场景变化而变化；二是引入更真实的动态目标几何形状与运动轨迹约束，提高扰动的物理真实性；三是结合语义分割、时序信息或多帧点云，提高动态遮挡建模与位置识别的协同能力。

---

## 参考文献

[1] 待补充。  
[2] 待补充。  
[3] 待补充。

---

## 附录：写作说明

1. 该 Markdown 文件已经按中文硕士论文常见的章节组织方式进行了改写，但 Markdown 本身不负责学校模板中的页边距、字号、页眉页脚、目录自动生成等版式要求。
2. 若后续需要提交正式论文，建议再将本文件内容迁移至学校规定的 `Word` 或 `LaTeX` 学位论文模板中。
3. 当前公式均采用 LaTeX 数学语法书写，便于后续直接迁移到正式模板。
