# Dynamic Adversarial Occlusion for LiDAR Place Recognition

这个原型实现了你描述的核心目标：

- `min_f max_G L_place(f(G(x)))`
- `G` 不是随机点 dropout，而是生成“车辆样式”遮挡
- 每个样本随机启用若干个 box，删除被这些 box 几何遮挡到的点

## 设计要点

1. `AdversarialOcclusionGenerator` 先预测若干 3D 盒体（车辆近似）：
- center `(x,y,z)`
- size `(l,w,h)`
- yaw

2. 由盒体产生两类遮挡分数：
- `inside-box`：点在盒体内部
- `shadow-cone`：点落在盒体角域后方（模拟被动态物体遮挡）

3. 每个样本随机激活 `num_box in [1, num_boxes]` 个 box：
- 凡是落在激活 box 内部或其遮挡锥后的点都会被删除
- 用 threshold + `straight-through` 估计器支持对 `G` 反向传播

4. 可选 object insertion：
- 在生成盒体表面采样点，插入到被删除位置，模拟动态目标回波

## 文件说明

- `occlusion_generator.py`
  - 生成器主体
  - 遮挡几何计算
  - 硬约束 mask 投影与点云修改函数
- `lpr_models.py`
  - 轻量 PointNet descriptor
  - batch-hard triplet loss
- `train.py`
  - 交替优化 `G` 与 `f`
  - synthetic 数据 smoke test（可直接替换为你的 LPR dataloader）

## 快速运行

```bash
cd /media/autolab/tsy
python dynamic/train.py --epochs 2
```

## 直接在 KITTI 训练

已提供 `dynamic/train_kitti.py`，默认直接使用 `/TIEVNAS/KITTI` 作为 `.bin` 真实路径：

```bash
python dynamic/train_kitti.py \
  --query-file /TIEVNAS/jyf/KITTI/kitti_vxp_training_queries_baseline_p10_n25_yaw.pickle \
  --kitti-root /TIEVNAS/KITTI \
  --fallback-root /TIEVNAS/jyf/KITTI \
  --epochs 20 \
  --batch-size 32 \
  --num-batches-per-epoch 200 \
  --num-workers 4
```

说明：

- `query-file` 用训练查询 pickle（通常在 `/TIEVNAS/jyf/KITTI`）
- 点云 `.bin` 会优先从 `--kitti-root=/TIEVNAS/KITTI` 读取
- descriptor 默认是 `PointNetVLAD`
- 训练是 `min_f max_G` 的对抗遮挡训练（随机 active boxes）

## KITTI Dataloader

已提供 `dynamic/kitti_dataloader.py`，默认使用 `/TIEVNAS/jyf/KITTI`：

```bash
python dynamic/kitti_dataloader.py
```

代码入口：

- `KITTIPointCloudQueryDataset`: 读取 `kitti_vxp_*queries*.pickle`（带正负样本关系）
- `KITTICsvPointCloudDataset`: 读取 `00.csv / all_annotation.csv` 逐帧数据
- `build_kitti_query_dataloader`: 一步构建 `DataLoader`

路径策略：

- 优先用 `/TIEVNAS/jyf/KITTI`
- 若 `.bin` 实际在 `/TIEVNAS/KITTI`，会自动回退解析

## KITTI Recall 测试

已提供 `dynamic/test_recall_kitti.py`，参考了 `inference_kitti.py` 的评估思路，使用 KDTree 计算：

- `Recall@K`
- `Recall@1%`

示例：

```bash
python dynamic/test_recall_kitti.py \
  --query-file /TIEVNAS/jyf/KITTI/kitti_vxp_test_queries_baseline_p10_n25_yaw.pickle \
  --checkpoint /media/autolab/tsy/dynamic/checkpoints/kitti_adv_epoch_020.pt \
  --kitti-root /TIEVNAS/KITTI \
  --topk 25 \
  --batch-size 32
```

快速 smoke test：

```bash
python dynamic/test_recall_kitti.py --device cpu --num-workers 0 --max-elems 64 --max-evals 32 --topk 5
```

可选参数：

- `--no-object-insertion`: 关闭对象插入，只做遮挡删除
- `--p-classes`, `--k-samples`: 控制 PK batch 采样
- `--size-prior-weight`, `--height-prior-weight`: 控制 realism 正则强度

## 接入你现有 LPR 框架时建议

1. 用你自己的 `descriptor network f` 替换 `PointNetDescriptor`
2. 用你自己的 metric loss / contrastive loss 替换 triplet loss
3. 先从较小的 active box 数开始，例如 `1~3`，再扩展到更多 box
4. 监控 clean recall 与 dynamic recall，避免 `G` 过强导致 clean 性能下降
