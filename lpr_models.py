from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetDescriptor(nn.Module):
    """
    Minimal PointNet-style descriptor for place recognition.
    """

    def __init__(self, in_channels: int = 3, emb_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = self.mlp(points)
        global_feat = x.max(dim=1).values
        emb = self.proj(global_feat)
        return F.normalize(emb, dim=-1)


class GatingContext(nn.Module):
    def __init__(self, dim: int, add_batch_norm: bool = True) -> None:
        super().__init__()
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.matmul(x, self.gating_weights)
        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases
        gates = self.sigmoid(gates)
        return x * gates


class NetVLADLoupe(nn.Module):
    def __init__(
        self,
        feature_size: int,
        max_samples: int,
        cluster_size: int,
        output_dim: int,
        gating: bool = True,
        add_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm

        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) / math.sqrt(feature_size)
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)
        self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm) if gating else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3).contiguous()
        x = x.view(-1, self.max_samples, self.feature_size)

        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = activation.transpose(2, 1)
        vlad = torch.matmul(activation, x)
        vlad = vlad.transpose(2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)
        vlad = self.bn2(vlad)
        if self.context_gating is not None:
            vlad = self.context_gating(vlad)
        return vlad


class STN3d(nn.Module):
    def __init__(self, num_points: int = 2500, k: int = 3, use_bn: bool = True) -> None:
        super().__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = nn.Conv2d(64, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 1024, (1, 1))
        self.mp1 = nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x).view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        return x.view(-1, self.k, self.k)


class PointNetFeat(nn.Module):
    def __init__(
        self,
        num_points: int = 2500,
        global_feat: bool = True,
        feature_transform: bool = False,
        max_pool: bool = True,
    ) -> None:
        super().__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = nn.Conv2d(1, 64, (1, 3))
        self.conv2 = nn.Conv2d(64, 64, (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (1, 1))
        self.conv4 = nn.Conv2d(64, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x: torch.Tensor):
        batchsize = x.size(0)
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x, dim=1), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x

        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x, dim=-1)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x

        x = self.mp1(x).view(-1, 1024)
        if self.global_feat:
            return x, trans
        x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([x, pointfeat], 1), trans


class PointNetVLADDescriptor(nn.Module):
    """
    PointNetVLAD-style global descriptor. Input is [B, N, C], output is [B, D].
    """

    def __init__(
        self,
        num_points: int = 4096,
        emb_dim: int = 256,
        feature_transform: bool = True,
        cluster_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.point_net = PointNetFeat(
            num_points=num_points,
            global_feat=True,
            feature_transform=feature_transform,
            max_pool=False,
        )
        self.net_vlad = NetVLADLoupe(
            feature_size=1024,
            max_samples=num_points,
            cluster_size=cluster_size,
            output_dim=emb_dim,
            gating=True,
            add_batch_norm=True,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[..., :3]
        if xyz.ndim != 3:
            raise ValueError(f"Expected input [B, N, C], got {tuple(points.shape)}")
        if xyz.shape[1] != self.num_points:
            raise ValueError(
                f"PointNetVLAD expects exactly {self.num_points} points, got {xyz.shape[1]}"
            )
        x = xyz.unsqueeze(1).contiguous()
        x = self.point_net(x)
        x = self.net_vlad(x)
        return F.normalize(x, dim=-1)


def build_descriptor_model(
    arch: str = "pointnetvlad",
    num_points: int = 4096,
    emb_dim: int = 256,
    in_channels: int = 3,
) -> nn.Module:
    arch = arch.lower()
    if arch == "pointnet":
        return PointNetDescriptor(in_channels=in_channels, emb_dim=emb_dim)
    if arch == "pointnetvlad":
        return PointNetVLADDescriptor(num_points=num_points, emb_dim=emb_dim)
    raise ValueError(f"Unsupported descriptor arch: {arch}")


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Batch-hard triplet:
    - hardest positive: max d(a, p)
    - hardest negative: min d(a, n)
    """
    dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    pos_mask = same & (~eye)
    neg_mask = ~same

    pos_dist = torch.where(pos_mask, dist, torch.zeros_like(dist))
    hardest_pos = pos_dist.max(dim=1).values

    big = torch.full_like(dist, 1e6)
    neg_dist = torch.where(neg_mask, dist, big)
    hardest_neg = neg_dist.min(dim=1).values

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean()


def embedding_consistency_loss(clean_emb: torch.Tensor, adv_emb: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(clean_emb, adv_emb, dim=-1)).mean()
