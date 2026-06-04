'''
# @Time    : 2025/5/14 9:34
# @Author  : Fengcheng Ji
# @Function: 双曲空间原型分类器，基于 Poincaré 球的对比聚类损失
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from hyperbolic import Hyperbolic
from encoder import PatchEncoder


class HyperbolicClassifier(nn.Module):
    def __init__(self, in_channels, embed_dim=16, num_classes=16, queue_length=20, momentum=0.99, delta=0.5,
                 cc_weight=0.1, curvature=0.5):
        super().__init__()
        self.encoder = PatchEncoder(in_channels, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.queue_length = queue_length
        self.momentum = momentum
        self.delta = delta
        self.cc_weight = cc_weight
        self.curvature = curvature

        self.register_buffer('prototypes', torch.randn(num_classes, embed_dim) * 0.001)
        self.prototypes = Hyperbolic.project(self.prototypes, c=self.curvature)

        self.feature_queues = defaultdict(
            lambda: torch.zeros(self.queue_length, self.embed_dim).to(self.prototypes.device))
        self.queue_ptr = defaultdict(lambda: 0)

    def update_feature_queue(self, features, labels):
        for i in range(labels.size(0)):
            cls = labels[i].item()
            feat = features[i].detach()
            self.feature_queues[cls][self.queue_ptr[cls]] = feat
            self.queue_ptr[cls] = (self.queue_ptr[cls] + 1) % self.queue_length
            if self.queue_ptr[cls] == 0:
                self.update_prototype(cls)

    def update_prototype(self, cls):
        queue = self.feature_queues[cls]
        prototype = Hyperbolic.compute_hyperbolic_prototype(queue, c=self.curvature, dim=0)
        new_prototypes = self.prototypes.clone()
        a = Hyperbolic.mobius_mul(self.momentum, self.prototypes[cls], c=self.curvature)
        b = Hyperbolic.mobius_mul(1 - self.momentum, prototype.squeeze(), c=self.curvature)
        new_prototypes[cls] = Hyperbolic.mobius_add(a, b, c=self.curvature)
        if torch.isnan(new_prototypes[cls]).any():
            print()

        # 超出边界时映射回双曲空间（留1%余量）
        R = (1.0 / torch.sqrt(torch.tensor(self.curvature))) * 0.99
        norm = torch.norm(new_prototypes[cls], dim=-1)
        if norm > R:
            new_prototypes[cls] = Hyperbolic.project(new_prototypes[cls], c=self.curvature)
            print('xxxxxxxxxxx')
        self.prototypes = new_prototypes

    def contrastive_clustering_loss(self, features, labels):
        dists = torch.stack([
            Hyperbolic.dist(features, p.unsqueeze(0).expand_as(features), c=self.curvature)
            for p in self.prototypes
        ], dim=1)
        mask = F.one_hot(labels, num_classes=self.num_classes).float()
        pos_loss = (mask * dists).sum(dim=1)
        neg_loss = ((1 - mask) * torch.relu(self.delta - dists)).sum(dim=1)
        return (pos_loss + neg_loss).mean()

    def forward(self, x, labels=None, iteration=0, inference=False):
        feat = self.encoder(x)
        feat = Hyperbolic.project(feat, c=self.curvature)

        dists = torch.stack([
            Hyperbolic.dist(feat, p.unsqueeze(0).expand_as(feat), c=self.curvature)
            for p in self.prototypes
        ], dim=1)
        logits = -dists

        if inference:
            return dists

        if labels is not None:
            self.update_feature_queue(feat, labels)

        loss = 0.0
        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)
            loss_cc = self.contrastive_clustering_loss(feat, labels)
            loss = loss_ce + self.cc_weight * loss_cc

        if iteration % 10 == 0 and labels is not None:
            for cls in range(self.num_classes):
                if cls in self.feature_queues:
                    self.update_prototype(cls)

        return logits, loss, feat
