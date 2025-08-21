#!/usr/bin/env python3
"""
Enhanced Heap-CHI Adversarial Training System - Corrected Version
All bugs fixed and spec gaps filled
"""

import os
import json
import math
import time
import heapq
import random
import copy
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Any, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import argparse
from contextlib import nullcontext

# Robust AMP import
try:
    # New API (no deprecation warning)
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler as _GradScaler
    def amp_autocast():
        return _autocast_new(device_type="cuda", dtype=torch.float16)
except Exception:
    try:
        # Old API
        from torch.cuda.amp import autocast as _autocast_old
        from torch.cuda.amp import GradScaler as _GradScaler
        def amp_autocast():
            return _autocast_old(dtype=torch.float16)
    except Exception:
        _GradScaler = None
        def amp_autocast():
            return nullcontext()

# Optional sklearn dependencies
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    SKLEARN_OK = True
except:
    SKLEARN_OK = False

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("heap_chi")

# CIFAR-10 normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# ========================= Configuration =========================
@dataclass
class Config:
    # Experiment
    experiment_name: str = "heap_chi_corrected"
    mode: str = "heap"  # "heap", "no_heap", "random_buffer", "hard_mining", "fifo"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    deterministic: bool = False
    time_budget_minutes: Optional[float] = None

    # Data
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    subset_size: Optional[int] = None
    val_split: float = 0.1
    use_normalization: bool = True

    # Training
    epochs: int = 30
    warmup_epochs: int = 3
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: List[int] = None

    # Attacks - Phase 1 correctness
    epsilon: float = 8/255
    pgd_steps: int = 5  # Start with 5 as per spec
    pgd_step_size: float = 2/255
    eval_pgd_steps: int = 20
    eval_pgd_restarts: int = 2

    # Phase 2 - Routed heaps
    heap_max_size: int = 2000
    initial_bank_size: int = 1000
    heap_ratio: float = 0.4  # ρ_leaf from spec
    global_heap_ratio: float = 0.3  # ρ_global from spec
    k_per_batch: int = 128
    diversity_k_factor: int = 3

    # Per-leaf heap sizes
    leaf_heap_size: int = 512  # K_leaf from spec
    global_heap_size: int = 128  # K_global from spec

    # Routing strategy
    routing_strategy: str = "three_level"  # freq -> attack -> confusion
    num_buckets: int = 20
    tree_refit_interval: int = 200
    tree_max_depth: int = 3
    tree_min_samples_leaf: int = 20

    # Scoring weights from spec
    lambda1: float = 0.5  # misclass weight
    lambda2: float = 0.3  # margin weight
    lambda3: float = 0.01  # age weight
    lambda4: float = 0.2  # redundancy weight
    gamma: float = 0.002  # age decay rate
    w_max: float = 50.0  # max weight for sum-tree
    tau: float = 0.7  # cosine similarity threshold

    # Phase 3 - Robustness plugins
    trades_beta: float = 0.0  # 0 = off, 0.3 = recommended
    mart_enabled: bool = False
    mart_weight: float = 1.5
    curriculum_enabled: bool = False
    yopo_enabled: bool = False

    # Phase 4 - CHI (default OFF)
    human_in_loop: Any = False  # Default OFF
    chi_cooldown: int = 200
    chi_decay_rate: float = 0.95
    chi_budget: int = 5
    chi_web_port: int = 5000

    # Evaluation
    eval_interval: int = 2
    log_interval: int = 50
    metrics_log_interval: int = 20
    early_stopping_patience: int = 10
    calibrate: bool = True

    model_name: str = "resnet18"        # ["resnet18","resnet34","mobilenet_v2"]
    pgd_budget_train: Optional[int] = None  # stop when PGD calls reach this
    focus_mode: str = "none"            # ["none","burst"]  (keep simple & fast)
    burst_len: int = 200                # steps to stay on one leaf

    # Output
    output_dir: str = "./experiments"

    def __post_init__(self):
        if self.lr_milestones is None:
            self.lr_milestones = [int(self.epochs * 0.5), int(self.epochs * 0.75)]
        os.makedirs(self.output_dir, exist_ok=True)
        self.experiment_dir = os.path.join(
            self.output_dir, f"{self.experiment_name}_{self.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

# ========================= Phase 1: Metrics & Correctness =========================
@dataclass
class EfficiencyMetrics:
    # Accurate PGD accounting
    pgd_calls_train: int = 0  # per-sample × steps
    fgsm_calls_train: int = 0
    pgd_calls_eval: int = 0  # per-sample × steps × restarts
    total_forward_passes: int = 0

    # Time tracking
    start_time: float = field(default_factory=time.time)
    wall_clock_seconds: List[float] = field(default_factory=list)

    # Heap dynamics
    heap_turnover_rate: List[float] = field(default_factory=list)
    heap_age_mean: List[float] = field(default_factory=list)
    heap_age_median: List[float] = field(default_factory=list)
    unique_variants_seen: Set[int] = field(default_factory=set)

    # Diversity
    bucket_entropy: List[float] = field(default_factory=list)
    per_class_coverage: List[Dict[int, float]] = field(default_factory=list)
    novelty_mean: List[float] = field(default_factory=list)

    # Batch quality
    batch_loss_mean: List[float] = field(default_factory=list)
    batch_margin_mean: List[float] = field(default_factory=list)
    heap_vs_fresh_ratio: List[Tuple[float, float]] = field(default_factory=list)

    # Anytime performance
    robust_acc_timeline: List[Tuple[float, float]] = field(default_factory=list)
    robust_acc_vs_pgd: List[Tuple[int, float]] = field(default_factory=list)

    def log_snapshot(self, robust_acc: Optional[float] = None):
        elapsed = time.time() - self.start_time
        self.wall_clock_seconds.append(elapsed)
        if robust_acc is not None:
            self.robust_acc_timeline.append((elapsed, robust_acc))
            self.robust_acc_vs_pgd.append((self.pgd_calls_train, robust_acc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pgd_calls_train': self.pgd_calls_train,
            'fgsm_calls_train': self.fgsm_calls_train,
            'pgd_calls_eval': self.pgd_calls_eval,
            'total_forward_passes': self.total_forward_passes,
            'total_unique_variants': len(self.unique_variants_seen),
            'wall_clock_seconds': self.wall_clock_seconds,
            'heap_turnover_rate': self.heap_turnover_rate,
            'heap_age_mean': self.heap_age_mean,
            'heap_age_median': self.heap_age_median,
            'bucket_entropy': self.bucket_entropy,
            'novelty_mean': self.novelty_mean,
            'batch_loss_mean': self.batch_loss_mean,
            'batch_margin_mean': self.batch_margin_mean,
            'heap_vs_fresh_ratio': self.heap_vs_fresh_ratio,
            'robust_acc_timeline': self.robust_acc_timeline,
            'robust_acc_vs_pgd': self.robust_acc_vs_pgd,
            'per_class_coverage': self.per_class_coverage,
        }

# ========================= Attacks =========================
class Attacks:
    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        device = x.device
        mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        device = x.device
        mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        return x * std + mean

    @staticmethod
    def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
             device: str, normalized: bool = False, set_eval: bool = True) -> torch.Tensor:
        was_training = model.training
        if set_eval: model.eval()  # BN-safe

        x = x.clone().detach().to(device).requires_grad_(True)
        y = y.to(device)

        logits = model(Attacks.normalize(x) if normalized else x)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        x_adv = torch.clamp(x + eps * x.grad.sign(), 0.0, 1.0)

        if was_training and set_eval: model.train()
        return x_adv.detach()

    @staticmethod
    def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
            step_size: float, steps: int, device: str, random_start: bool = True,
            normalized: bool = False, set_eval: bool = True) -> torch.Tensor:
        was_training = model.training
        if set_eval: model.eval()  # BN-safe

        x = x.to(device)
        y = y.to(device)

        if random_start:
            x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)
        else:
            x_adv = x.clone()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = model(Attacks.normalize(x_adv) if normalized else x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        if was_training and set_eval: model.train()
        return x_adv.detach()

# ========================= Phase 2: Routing & Features (FIXED) =========================
class FastFeatureExtractor:
    """Extract cheap routing features - FIXED to include confidence"""
    def __init__(self, model: nn.Module, device: str, use_normalization: bool):
        self.model = model
        self.device = device
        self.use_normalization = use_normalization

    def extract_features(self, x_clean: torch.Tensor, x_adv: torch.Tensor,
                        y: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        delta = x_adv - x_clean
        b = delta.size(0)

        # Basic norms
        flat = delta.view(b, -1)
        features['l2_norm'] = flat.norm(p=2, dim=1) / math.sqrt(flat.size(1))
        features['linf_norm'] = flat.abs().max(dim=1)[0]

        # Frequency features (Level 1 routing)
        # Laplacian variance
        laplacian_kernel = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]],
                                       dtype=torch.float32, device=self.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        with torch.no_grad():
            lap_response = F.conv2d(delta.abs(), laplacian_kernel, padding=1, groups=3)
            features['laplacian_var'] = lap_response.var(dim=(1,2,3))

        # Sobel energy
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                              dtype=torch.float32, device=self.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                              dtype=torch.float32, device=self.device)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        with torch.no_grad():
            edge_x = F.conv2d(delta.abs(), sobel_x, padding=1, groups=3)
            edge_y = F.conv2d(delta.abs(), sobel_y, padding=1, groups=3)
            features['sobel_energy'] = (edge_x**2 + edge_y**2).sum(dim=(1,2,3))

        # FFT ratio
        gray = delta.mean(dim=1, keepdim=True)
        fft = torch.fft.rfft2(gray, s=(16, 16))
        mag = torch.abs(fft)
        low = mag[:, :, :4, :4].mean(dim=(1, 2, 3))
        high = mag[:, :, 4:, 4:].mean(dim=(1, 2, 3))
        total_f = mag.mean(dim=(1, 2, 3)) + 1e-8
        features['low_freq_ratio'] = low / total_f
        features['high_freq_ratio'] = high / total_f

        # Spatial concentration
        delta_abs = delta.abs().mean(dim=1, keepdim=True)
        total = delta_abs.view(b, -1).sum(dim=1) + 1e-8
        maximum = delta_abs.view(b, -1).max(dim=1)[0]
        features['spatial_concentration'] = maximum / total

        # Total variation
        tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().sum(dim=(1, 2, 3))
        tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().sum(dim=(1, 2, 3))
        features['total_variation'] = (tv_h + tv_w) / delta[0].numel()

        # Model-based features (FIXED)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits_clean = self.model(Attacks.normalize(x_clean) if self.use_normalization else x_clean)
            logits_adv = self.model(Attacks.normalize(x_adv) if self.use_normalization else x_adv)

            probs_clean = F.softmax(logits_clean, dim=1)
            probs_adv = F.softmax(logits_adv, dim=1)

            pred_adv = logits_adv.argmax(dim=1)
            features['attack_success'] = (pred_adv != y).float()

            # Confidence and confidence drop (FIXED - now storing confidence)
            clean_conf = probs_clean.gather(1, y.unsqueeze(1)).squeeze()
            adv_conf = probs_adv.gather(1, y.unsqueeze(1)).squeeze()
            features['confidence'] = adv_conf  # FIXED: Added this line
            features['confidence_drop'] = clean_conf - adv_conf

            # Margin
            true_logits = logits_adv.gather(1, y.unsqueeze(1)).squeeze()
            other_logits = logits_adv.clone()
            other_logits.scatter_(1, y.unsqueeze(1), float('-inf'))
            max_other = other_logits.max(dim=1)[0]
            features['margin_logit'] = true_logits - max_other
            features['normalized_margin'] = torch.sigmoid(features['margin_logit'])
            features['hardness'] = 1.0 - features['normalized_margin']

            # Entropy
            features['entropy'] = -(probs_adv * torch.log(probs_adv + 1e-8)).sum(dim=1)

            # Top-2 confusion pair (for Level 3 routing)
            top2 = logits_adv.topk(2, dim=1)
            features['pred_class'] = top2.indices[:, 0]
            features['second_class'] = top2.indices[:, 1]

        if was_training: self.model.train()
        return features

# ========================= Variant with Routing Info =========================
class Variant:
    _id_counter = 0
    __slots__ = ('id', 'x_clean', 'delta', 'x_clean_idx', 'y', 'features',
                 'leaf_id', 'score', 'age', 'last_scored_step', 'generation_step',
                 'refresh_count', '_feature_vector', 'attack_meta')

    def __init__(self, x_clean: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor,
                 features: Dict[str, float], idx: int = -1, generation_step: int = 0,
                 attack_meta: Dict[str, Any] = None):
        self.id = Variant._id_counter
        Variant._id_counter += 1
        self.x_clean = x_clean.cpu().half()
        self.delta = (x_adv - x_clean).cpu().half()
        self.x_clean_idx = idx
        self.y = y.item() if isinstance(y, torch.Tensor) else y
        self.features = {k: (v.item() if isinstance(v, torch.Tensor) else v)
                        for k, v in features.items()}
        self.leaf_id = None  # Will be set by router
        self.score = 0.0
        self.age = 0
        self.last_scored_step = 0
        self.generation_step = generation_step
        self.refresh_count = 0
        self._feature_vector = None
        self.attack_meta = attack_meta or {}

    def get_feature_vector(self) -> np.ndarray:
        if self._feature_vector is None:
            keys = ['hardness', 'l2_norm', 'linf_norm', 'spatial_concentration',
                   'high_freq_ratio', 'laplacian_var', 'sobel_energy']
            self._feature_vector = np.array([self.features.get(k, 0.0) for k in keys],
                                           dtype=np.float32)
        return self._feature_vector

    def reconstruct_adv(self, device: str) -> torch.Tensor:
        x = self.x_clean.float().to(device)
        d = self.delta.float().to(device)
        return torch.clamp(x + d, 0.0, 1.0)

# ========================= Phase 2: Sum-Tree for O(log K) sampling =========================
class SumTree:
    """Fenwick/segment tree for proportional sampling"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.size = 0
        self.ptr = 0

    def add(self, priority: float, data: Any):
        idx = self.ptr
        self.data[idx] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def sample(self, v: float) -> Tuple[int, float, Any]:
        parent_idx = 0

        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1

            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self) -> float:
        return self.tree[0]

# ========================= Phase 2: Leaf Heap with Sum-Tree (FIXED) =========================
class LeafHeap:
    """Per-leaf heap with sum-tree sampling - FIXED with proper sync"""
    def __init__(self, capacity: int, cfg: Config):
        self.capacity = capacity
        self.cfg = cfg
        self.heap = []  # (score, counter, variant)
        self.sum_tree = SumTree(capacity)
        self.counter = 0
        self.items: Dict[int, Variant] = {}  # id -> Variant
        self.id_to_idx: Dict[int, int] = {}  # id -> sumtree index

    def _set_weight(self, variant_id: int, w: float):
        """Update weight for existing variant"""
        if variant_id in self.id_to_idx:
            idx = self.id_to_idx[variant_id]
            self.sum_tree.update(idx, w)

    def insert(self, variant: Variant, score: float) -> bool:
        weight = min(math.exp(score), self.cfg.w_max)

        if len(self.heap) < self.capacity:
            # Not full, just add
            heapq.heappush(self.heap, (score, self.counter, variant))
            idx = self.sum_tree.ptr
            self.sum_tree.add(weight, variant)
            self.items[variant.id] = variant
            self.id_to_idx[variant.id] = idx
            self.counter += 1
            return True

        # Heap full: consider replacement
        if score > self.heap[0][0]:
            old_score, old_cnt, old_var = heapq.heapreplace(self.heap, (score, self.counter, variant))

            # Zero out old variant's weight
            self._set_weight(old_var.id, 0.0)
            self.items.pop(old_var.id, None)
            self.id_to_idx.pop(old_var.id, None)

            # Add new variant
            idx = self.sum_tree.ptr
            self.sum_tree.add(weight, variant)
            self.items[variant.id] = variant
            self.id_to_idx[variant.id] = idx
            self.counter += 1
            return True
        return False

    def sample(self, k: int) -> List[Variant]:
        """Sample k unique variants"""
        if not self.heap or k <= 0 or self.sum_tree.total() <= 0:
            return []

        out: List[Variant] = []
        seen: Set[int] = set()
        trials = 0
        max_trials = min(5 * k, 100)

        while len(out) < min(k, len(self.items)) and trials < max_trials:
            v = random.random() * self.sum_tree.total()
            _, _, var = self.sum_tree.sample(v)
            if var is not None and var.id in self.items and var.id not in seen:
                out.append(var)
                seen.add(var.id)
            trials += 1

        return out

    def age_decay(self, gamma: float, current_step: int):
        """Apply exponential age decay and update sum-tree"""
        new_heap = []
        for score, cnt, var in self.heap:
            age = current_step - var.last_scored_step
            new_score = score * math.exp(-gamma * age)
            new_heap.append((new_score, cnt, var))

            # Update sum-tree weight
            new_w = min(math.exp(new_score), self.cfg.w_max)
            self._set_weight(var.id, new_w)

        self.heap = new_heap
        heapq.heapify(self.heap)

# ========================= Phase 2: Routing Tree =========================
class RoutingTree:
    """Three-level routing: freq -> attack -> confusion"""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.leaves = {}  # leaf_id -> LeafHeap
        self.global_heap = LeafHeap(cfg.global_heap_size, cfg)
        self.routing_stats = defaultdict(int)
        self.median_tracker = deque(maxlen=1000)

    def route(self, variant: Variant) -> str:
        """Route variant to appropriate leaf"""
        features = variant.features

        # Level 1: Frequency profile
        freq_score = features.get('laplacian_var', 0) + features.get('sobel_energy', 0)
        self.median_tracker.append(freq_score)

        if len(self.median_tracker) > 100:
            median = np.median(list(self.median_tracker))
        else:
            median = 0.5

        freq_level = "HF" if freq_score > median else "LF"

        # Level 2: Attack strength (from meta)
        attack_type = variant.attack_meta.get('type', 'PGD5')

        # Level 3: Confusion structure
        true_class = variant.y
        pred_class = int(features.get('pred_class', 0))

        # Construct leaf ID
        leaf_id = f"{freq_level}:{attack_type}:{true_class}->{pred_class}"
        variant.leaf_id = leaf_id

        # Create leaf heap if needed
        if leaf_id not in self.leaves:
            self.leaves[leaf_id] = LeafHeap(self.cfg.leaf_heap_size, self.cfg)

        self.routing_stats[leaf_id] += 1
        return leaf_id

    def compute_entropy(self) -> float:
        """Compute routing entropy for diversity monitoring"""
        if not self.routing_stats:
            return 0.0
        total = sum(self.routing_stats.values())
        probs = [count/total for count in self.routing_stats.values()]
        return -sum(p * math.log(p + 1e-8) for p in probs)

# ========================= Phase 3: Robustness Plugins =========================
class RobustnessPlugins:
    """Lightweight robustness enhancements"""

    @staticmethod
    def trades_loss(logits_clean: torch.Tensor, logits_adv: torch.Tensor,
                   y: torch.Tensor, beta: float = 0.3, T: float = 1.0) -> torch.Tensor:
        """TRADES-lite loss"""
        ce_loss = F.cross_entropy(logits_adv, y)
        if beta > 0:
            kl_loss = F.kl_div(
                F.log_softmax(logits_adv / T, dim=1),
                F.softmax(logits_clean / T, dim=1),
                reduction='batchmean'
            )
            return ce_loss + beta * kl_loss
        return ce_loss

    @staticmethod
    def mart_reweight(logits: torch.Tensor, y: torch.Tensor,
                      weight: float = 1.5) -> torch.Tensor:
        """MART-style reweighting for misclassified samples"""
        preds = logits.argmax(dim=1)
        weights = torch.ones_like(y, dtype=torch.float32)
        weights[preds != y] = weight
        return weights

# ========================= Alternative Baseline Modes =========================
class AlternativeBuffers:
    """Simple baseline modes for comparison"""

    @staticmethod
    def random_buffer_sample(buffer: deque, k: int) -> List[Any]:
        """Random sampling from buffer"""
        if not buffer or k <= 0:
            return []
        return random.sample(list(buffer), min(k, len(buffer)))

    @staticmethod
    def hard_mining_sample(buffer: deque, k: int, key_func) -> List[Any]:
        """Hard example mining from buffer"""
        if not buffer or k <= 0:
            return []
        sorted_buffer = sorted(buffer, key=key_func, reverse=True)
        return sorted_buffer[:min(k, len(buffer))]

    @staticmethod
    def fifo_sample(queue: deque, k: int) -> List[Any]:
        """FIFO queue sampling"""
        samples = []
        for _ in range(min(k, len(queue))):
            if queue:
                samples.append(queue.popleft())
        return samples

# ========================= Calibration (Phase 1) =========================
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error"""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_conf - bin_acc)

    return float(ece)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

def fit_temperature(model: nn.Module, valloader: DataLoader, device: torch.device,
                   normalized: bool) -> TemperatureScaler:
    """Fit temperature scaling on validation set"""
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            logits = model(Attacks.normalize(x) if normalized else x)
            logits_list.append(logits)
            labels_list.append(y)

    logits = torch.cat(logits_list, 0)
    labels = torch.cat(labels_list, 0)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

def compute_map_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Macro-average precision from logits"""
    if not SKLEARN_OK:
        return float('nan')

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    n_classes = probs.shape[1]
    aps = []

    for c in range(n_classes):
        y_true = (labels == c).astype(np.int32)
        y_score = probs[:, c]
        try:
            ap = average_precision_score(y_true, y_score)
            aps.append(ap)
        except:
            continue

    return float(np.mean(aps)) if aps else float('nan')

# ========================= Main Trainer (FIXED) =========================
class HeapCHITrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._set_seed(cfg.seed, cfg.deterministic)

        # Model & optimizer
        self.model = self._build_model()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.lr_milestones
        )
        self.scaler = _GradScaler() if (self.cfg.use_amp and _GradScaler and self.cfg.device.startswith("cuda")) else None


        # Components
        self.feature_extractor = FastFeatureExtractor(self.model, str(self.device), cfg.use_normalization)
        self.routing_tree = RoutingTree(cfg) if cfg.mode == "heap" else None
        self.robustness_plugins = RobustnessPlugins()

        # Alternative modes
        if cfg.mode == "random_buffer":
            self.buffer = deque(maxlen=cfg.heap_max_size)
        elif cfg.mode == "hard_mining":
            self.buffer = deque(maxlen=cfg.heap_max_size)
        elif cfg.mode == "fifo":
            self.queue = deque(maxlen=cfg.heap_max_size)

        # Metrics
        self.metrics = EfficiencyMetrics()
        self.history = defaultdict(list)

        # Training state
        self.global_step = 0
        self.current_epoch = 0  # FIXED: Added epoch tracking
        self.best_val_robust = 0.0
        self.patience_counter = 0
        self.temperature_scaler = None

        self._focus_leaf = None
        self._focus_until = -1

        # CHI interface
        self.chi_active = cfg.human_in_loop
        self.feature_weights = {
            'hardness': 1.0,
            'confidence_drop': 1.0,
            'margin_logit': 1.0
        }

    def _set_seed(self, seed: int, deterministic: bool = False):
        """Set all random seeds"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic

    def _build_model(self) -> nn.Module:
        name = self.cfg.model_name
        if name == "resnet18":
            m = torchvision.models.resnet18(weights=None)
            m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
            m.fc = nn.Linear(512, 10)
        elif name == "resnet34":
            m = torchvision.models.resnet34(weights=None)
            m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
            m.fc = nn.Linear(512, 10)
        elif name == "mobilenet_v2":
            m = torchvision.models.mobilenet_v2(weights=None)
            # CIFAR-friendly first conv
            m.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
            m.classifier[1] = nn.Linear(m.last_channel, 10)
        else:
            raise ValueError(f"Unknown model_name: {name}")
        return m.to(self.device)


    def _compute_hardness_score(self, variant: Variant) -> float:
        """Compute hardness score with feature weights integration"""
        w = self.feature_weights
        confidence = variant.features.get('confidence', 0.1)
        ce_loss = -math.log(confidence + 1e-8)
        misclass = float(variant.features.get('attack_success', 0))
        margin = variant.features.get('margin_logit', 0.0)
        conf_drop = variant.features.get('confidence_drop', 0.0)
        age = self.global_step - variant.last_scored_step
        redundancy = 0.0  # TODO: embedding similarity

        score = (
            ce_loss * w.get('hardness', 1.0)
            + self.cfg.lambda1 * misclass
            - self.cfg.lambda2 * margin * w.get('margin_logit', 1.0)
            + 0.5 * conf_drop * w.get('confidence_drop', 1.0)
            - self.cfg.lambda3 * age
            - self.cfg.lambda4 * redundancy
        )
        return float(score)

    def train(self, trainloader: DataLoader, valloader: DataLoader, testloader: DataLoader):
        """Main training loop - FIXED"""
        logger.info(f"Starting {self.cfg.mode} training for {self.cfg.epochs} epochs")
        time_limit = self.cfg.time_budget_minutes * 60 if self.cfg.time_budget_minutes else float('inf')

        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch  # FIXED: Track epoch

            if time.time() - self.metrics.start_time > time_limit:
                logger.info(f"Time budget exceeded at epoch {epoch}")
                break

            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            # Warmup with clean training
            if epoch < self.cfg.warmup_epochs:
                logger.info(f"Warmup epoch {epoch+1}/{self.cfg.warmup_epochs}")
                for x, y in trainloader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(Attacks.normalize(x) if self.cfg.use_normalization else x)
                    loss = F.cross_entropy(logits, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_steps += 1
                    self.metrics.total_forward_passes += x.size(0)

                self.scheduler.step()
                continue

            # Build initial bank after warmup
            if epoch == self.cfg.warmup_epochs and self.cfg.mode == "heap":
                self.model.eval()
                self._build_initial_bank(trainloader)
                self.model.train()

            # Main training
            for batch_idx, (x, y) in enumerate(trainloader):
                if self.cfg.pgd_budget_train and self.metrics.pgd_calls_train >= self.cfg.pgd_budget_train:
                    logger.info(f"PGD budget reached: {self.metrics.pgd_calls_train}")
                    break
                x, y = x.to(self.device), y.to(self.device)

                # Get training batch based on mode (FIXED)
                if self.cfg.mode == "heap":
                    x_clean_b, x_adv_b, y_b = self._get_heap_batch(x, y)
                elif self.cfg.mode == "random_buffer":
                    x_clean_b, x_adv_b, y_b = self._get_random_buffer_batch(x, y)
                elif self.cfg.mode == "hard_mining":
                    x_clean_b, x_adv_b, y_b = self._get_hard_mining_batch(x, y)
                elif self.cfg.mode == "fifo":
                    x_clean_b, x_adv_b, y_b = self._get_fifo_batch(x, y)
                else:  # no_heap
                    x_adv_b, _ = self._get_standard_batch(x, y)
                    x_clean_b, y_b = x, y

                # Training step with correct data (FIXED)
                loss = self._train_step(x_clean_b, x_adv_b, y_b)

                epoch_loss += loss
                epoch_steps += 1
                self.global_step += 1

                if self.cfg.pgd_budget_train and self.metrics.pgd_calls_train >= self.cfg.pgd_budget_train:
                    logger.info(f"PGD budget reached: {self.metrics.pgd_calls_train}")
                    break  # break out of the batch loop


                # Apply CHI actions if enabled
                if self.chi_active and self.global_step % 100 == 0:
                    self._apply_chi_actions()

                # Log metrics
                if self.global_step % self.cfg.metrics_log_interval == 0:
                    self._log_metrics()

                budget_hit = (self.cfg.pgd_budget_train and
                              self.metrics.pgd_calls_train >= self.cfg.pgd_budget_train)
                time_hit = (time.time() - self.metrics.start_time > time_limit)
                if budget_hit or time_hit:
                    logger.info("Stopping training (budget/time reached).")
                    break

            self.scheduler.step()

            # Evaluation
            if (epoch + 1) % self.cfg.eval_interval == 0:
                val_metrics = self._evaluate(valloader)
                self.metrics.log_snapshot(val_metrics['robust_acc'])

                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.epochs} | "
                    f"Loss {epoch_loss/max(epoch_steps,1):.4f} | "
                    f"Val Clean {val_metrics['clean_acc']:.4f} | "
                    f"Val Robust {val_metrics['robust_acc']:.4f} | "
                    f"PGD calls {self.metrics.pgd_calls_train}"
                )

                # Early stopping
                if val_metrics['robust_acc'] > self.best_val_robust:
                    self.best_val_robust = val_metrics['robust_acc']
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.cfg.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        # Final evaluation with calibration
        if self.cfg.calibrate:
            self.temperature_scaler = fit_temperature(
                self.model, valloader, self.device, self.cfg.use_normalization
            )
            logger.info(f"Temperature: {float(self.temperature_scaler.temperature.item()):.4f}")

        test_metrics = self._comprehensive_evaluate(testloader, valloader)
        self._save_results(test_metrics)

        return self.model, self.history, test_metrics

    def _build_initial_bank(self, trainloader: DataLoader):
        """Build initial variant bank - FIXED"""
        logger.info(f"Building initial bank ({self.cfg.initial_bank_size} variants)")
        added = 0  # FIXED: Track actual additions

        for batch_idx, (x, y) in enumerate(trainloader):
            if added >= self.cfg.initial_bank_size:
                break

            x, y = x.to(self.device), y.to(self.device)

            # Generate FGSM adversaries for initial bank
            x_adv = Attacks.fgsm(
                self.model, x, y, self.cfg.epsilon,
                str(self.device), self.cfg.use_normalization, set_eval=True
            )
            self.metrics.fgsm_calls_train += x.size(0)

            features = self.feature_extractor.extract_features(x, x_adv, y)

            for i in range(x.size(0)):
                if added >= self.cfg.initial_bank_size:
                    break

                vf = {k: v[i] for k, v in features.items()}
                variant = Variant(
                    x[i], x_adv[i], y[i], vf,
                    idx=batch_idx * self.cfg.batch_size + i,
                    generation_step=0,
                    attack_meta={'type': 'FGSM'}
                )

                # Route and add to heaps
                leaf_id = self.routing_tree.route(variant)
                score = self._compute_hardness_score(variant)
                variant.score = score
                variant.last_scored_step = 0

                self.routing_tree.leaves[leaf_id].insert(variant, score)
                self.routing_tree.global_heap.insert(variant, score)
                self.metrics.unique_variants_seen.add(variant.id)
                added += 1  # FIXED: Increment counter

        logger.info(f"Initial bank ready: added {added} variants across {len(self.routing_tree.leaves)} leaves")

    def _get_heap_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch using heap sampling - FIXED to return clean, adv, labels"""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        leaf_k = int(batch_size * self.cfg.heap_ratio)
        global_k = int(batch_size * self.cfg.global_heap_ratio)
        fresh_k = batch_size - leaf_k - global_k
        if fresh_k < 0:
            # scale down the two heap draws proportionally to make room for fresh
            total_heap_k = leaf_k + global_k
            if total_heap_k > 0:
                scale = (batch_size - 1) / total_heap_k  # leave at least 1 fresh
                leaf_k = max(0, int(leaf_k * scale))
                global_k = max(0, int(global_k * scale))
            fresh_k = batch_size - leaf_k - global_k

        variants: List[Variant] = []

        if self.cfg.focus_mode == "burst":
            if self._focus_leaf is None or self.global_step >= self._focus_until:
                # pick the leaf whose heap currently has the highest top score
                if self.routing_tree.leaves:
                    def leaf_top_score(lh):
                        # lh.heap is a min-heap by score; get max score robustly
                        return max((s for (s, _, _) in lh.heap), default=float("-inf"))
                    hardest_leaf_id, _ = max(
                        ((lid, leaf_top_score(lh)) for lid, lh in self.routing_tree.leaves.items()),
                        key=lambda t: t[1],
                        default=(None, float("-inf"))
                    )
                    self._focus_leaf = hardest_leaf_id
                    self._focus_until = self.global_step + self.cfg.burst_len

        # Sample from leaf heaps
        if self.routing_tree.leaves and leaf_k > 0:
            if self.cfg.focus_mode == "burst" and self._focus_leaf in self.routing_tree.leaves:
                variants.extend(self.routing_tree.leaves[self._focus_leaf].sample(leaf_k))
            else:
                per_leaf = max(1, leaf_k // max(1, len(self.routing_tree.leaves)))
                picked = 0
                for lh in self.routing_tree.leaves.values():
                    if picked >= leaf_k:
                        break
                    take = min(per_leaf, leaf_k - picked)
                    variants.extend(lh.sample(take))
                    picked += take

        # Sample from global heap
        if global_k > 0:
            variants.extend(self.routing_tree.global_heap.sample(global_k))

        # Generate fresh adversaries
        if fresh_k > 0:
            fresh_x, fresh_y = x[:fresh_k], y[:fresh_k]

            # Curriculum on attack strength (FIXED)
            pgd_k = min(self.cfg.pgd_steps + (self.current_epoch // 5 if self.cfg.curriculum_enabled else 0), 10)

            x_adv_fresh = Attacks.pgd(
                self.model, fresh_x, fresh_y,
                self.cfg.epsilon, self.cfg.pgd_step_size, pgd_k,
                str(self.device), normalized=self.cfg.use_normalization, set_eval=True
            )
            self.metrics.pgd_calls_train += pgd_k * fresh_x.size(0)

            feats = self.feature_extractor.extract_features(fresh_x, x_adv_fresh, fresh_y)
            for i in range(fresh_x.size(0)):
                vf = {k: feats[k][i] for k in feats}
                v = Variant(
                    fresh_x[i], x_adv_fresh[i], fresh_y[i], vf,
                    generation_step=self.global_step,
                    attack_meta={'type': f'PGD{pgd_k}', 'k': pgd_k}
                )
                leaf_id = self.routing_tree.route(v)
                v.score = self._compute_hardness_score(v)
                v.last_scored_step = self.global_step
                self.routing_tree.leaves[leaf_id].insert(v, v.score)
                self.routing_tree.global_heap.insert(v, v.score)
                variants.append(v)
                self.metrics.unique_variants_seen.add(v.id)

        if not variants:
            # Fallback: use clean batch
            return x, x, y

        # Build aligned tensors from variants (FIXED)
        x_clean_batch = torch.stack([v.x_clean.float().to(self.device) for v in variants], dim=0)
        x_adv_batch = torch.stack([v.reconstruct_adv(str(self.device)) for v in variants], dim=0)
        y_batch = torch.tensor([v.y for v in variants], device=self.device, dtype=torch.long)

        # Track mix
        heap_count = len(variants) - fresh_k
        self.metrics.heap_vs_fresh_ratio.append(
            (heap_count/len(variants), fresh_k/len(variants)) if len(variants) > 0 else (0.0, 0.0)
        )

        # Apply age decay periodically
        if self.global_step % 100 == 0:
            for leaf_heap in self.routing_tree.leaves.values():
                leaf_heap.age_decay(self.cfg.gamma, self.global_step)
            self.routing_tree.global_heap.age_decay(self.cfg.gamma, self.global_step)

        return x_clean_batch, x_adv_batch, y_batch

    def _get_random_buffer_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random buffer baseline - FIXED label handling"""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        buffer_k = min(batch_size // 2, len(self.buffer))

        # Sample from buffer
        buffer_variants = AlternativeBuffers.random_buffer_sample(self.buffer, buffer_k)

        # Generate fresh
        fresh_k = batch_size - len(buffer_variants)
        fresh_x, fresh_y = x[:fresh_k], y[:fresh_k]

        x_adv_fresh = Attacks.pgd(
            self.model, fresh_x, fresh_y,
            self.cfg.epsilon, self.cfg.pgd_step_size, self.cfg.pgd_steps,
            str(self.device), normalized=self.cfg.use_normalization, set_eval=True
        )
        self.metrics.pgd_calls_train += self.cfg.pgd_steps * fresh_x.size(0)

        # Add to buffer
        for i in range(fresh_x.size(0)):
            self.buffer.append((fresh_x[i], x_adv_fresh[i], fresh_y[i]))

        # Combine (FIXED: proper label handling)
        all_x_clean = []
        all_x_adv = []
        all_y = []

        for x_c, x_a, y_i in buffer_variants:
            all_x_clean.append(x_c.to(self.device))
            all_x_adv.append(x_a.to(self.device))
            all_y.append(int(y_i.item()) if torch.is_tensor(y_i) else int(y_i))

        all_x_clean.extend([fresh_x[i] for i in range(fresh_k)])
        all_x_adv.extend([x_adv_fresh[i] for i in range(fresh_k)])
        all_y.extend([int(fresh_y[i].item()) for i in range(fresh_k)])

        x_clean_batch = torch.stack(all_x_clean)
        x_adv_batch = torch.stack(all_x_adv)
        y_batch = torch.tensor(all_y, device=self.device, dtype=torch.long)

        return x_clean_batch, x_adv_batch, y_batch

    def _get_hard_mining_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hard mining baseline - FIXED label handling"""
        batch_size = min(x.size(0), self.cfg.k_per_batch)

        # Mine hard examples
        if self.buffer:
            buffer_k = min(batch_size // 2, len(self.buffer))
            buffer_variants = AlternativeBuffers.hard_mining_sample(
                self.buffer, buffer_k,
                key_func=lambda v: v[3] if len(v) > 3 else 0  # Use hardness if stored
            )
        else:
            buffer_variants = []

        # Generate fresh
        fresh_k = batch_size - len(buffer_variants)
        fresh_x, fresh_y = x[:fresh_k], y[:fresh_k]

        x_adv_fresh = Attacks.pgd(
            self.model, fresh_x, fresh_y,
            self.cfg.epsilon, self.cfg.pgd_step_size, self.cfg.pgd_steps,
            str(self.device), normalized=self.cfg.use_normalization, set_eval=True
        )
        self.metrics.pgd_calls_train += self.cfg.pgd_steps * fresh_x.size(0)

        # Compute hardness for fresh examples
        with torch.no_grad():
            logits = self.model(Attacks.normalize(x_adv_fresh) if self.cfg.use_normalization else x_adv_fresh)
            probs = F.softmax(logits, dim=1)
            conf = probs.gather(1, fresh_y.unsqueeze(1)).squeeze()
            hardness = 1.0 - conf

        # Add to buffer with hardness
        for i in range(fresh_x.size(0)):
            self.buffer.append((fresh_x[i], x_adv_fresh[i], fresh_y[i], hardness[i].item()))

        # Combine (FIXED: proper label handling)
        all_x_clean = []
        all_x_adv = []
        all_y = []

        for item in buffer_variants:
            x_c, x_a, y_i = item[:3]
            all_x_clean.append(x_c.to(self.device))
            all_x_adv.append(x_a.to(self.device))
            all_y.append(int(y_i.item()) if torch.is_tensor(y_i) else int(y_i))

        all_x_clean.extend([fresh_x[i] for i in range(fresh_k)])
        all_x_adv.extend([x_adv_fresh[i] for i in range(fresh_k)])
        all_y.extend([int(fresh_y[i].item()) for i in range(fresh_k)])

        x_clean_batch = torch.stack(all_x_clean)
        x_adv_batch = torch.stack(all_x_adv)
        y_batch = torch.tensor(all_y, device=self.device, dtype=torch.long)

        return x_clean_batch, x_adv_batch, y_batch

    def _get_fifo_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FIFO queue baseline - FIXED label handling"""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        queue_k = min(batch_size // 2, len(self.queue))

        # Pop from queue
        queue_variants = AlternativeBuffers.fifo_sample(self.queue, queue_k)

        # Generate fresh
        fresh_k = batch_size - len(queue_variants)
        fresh_x, fresh_y = x[:fresh_k], y[:fresh_k]

        x_adv_fresh = Attacks.pgd(
            self.model, fresh_x, fresh_y,
            self.cfg.epsilon, self.cfg.pgd_step_size, self.cfg.pgd_steps,
            str(self.device), normalized=self.cfg.use_normalization, set_eval=True
        )
        self.metrics.pgd_calls_train += self.cfg.pgd_steps * fresh_x.size(0)

        # Add to queue
        for i in range(fresh_x.size(0)):
            self.queue.append((fresh_x[i], x_adv_fresh[i], fresh_y[i]))

        # Combine (FIXED: proper label handling)
        all_x_clean = []
        all_x_adv = []
        all_y = []

        for x_c, x_a, y_i in queue_variants:
            all_x_clean.append(x_c.to(self.device))
            all_x_adv.append(x_a.to(self.device))
            all_y.append(int(y_i.item()) if torch.is_tensor(y_i) else int(y_i))

        all_x_clean.extend([fresh_x[i] for i in range(fresh_k)])
        all_x_adv.extend([x_adv_fresh[i] for i in range(fresh_k)])
        all_y.extend([int(fresh_y[i].item()) for i in range(fresh_k)])

        x_clean_batch = torch.stack(all_x_clean)
        x_adv_batch = torch.stack(all_x_adv)
        y_batch = torch.tensor(all_y, device=self.device, dtype=torch.long)

        return x_clean_batch, x_adv_batch, y_batch

    def _get_standard_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Get standard PGD batch (baseline)"""
        x_adv = Attacks.pgd(
            self.model, x, y,
            self.cfg.epsilon, self.cfg.pgd_step_size, self.cfg.pgd_steps,
            str(self.device), normalized=self.cfg.use_normalization,
            set_eval=True
        )
        self.metrics.pgd_calls_train += self.cfg.pgd_steps * x.size(0)
        return x_adv, {}

    def _train_step(self, x_clean: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step - FIXED AMP and MART"""
        if self.cfg.use_normalization:
            x_clean = Attacks.normalize(x_clean)
            x_adv = Attacks.normalize(x_adv)

        self.optimizer.zero_grad()

        # Use AMP correctly with proper MART handling
        if self.scaler:
            with amp_autocast():
                if self.cfg.trades_beta > 0:
                    logits_clean = self.model(x_clean)
                    logits_adv = self.model(x_adv)

                    if self.cfg.mart_enabled:
                        # TRADES with MART: compute per-sample CE then weight
                        ce_vec = F.cross_entropy(logits_adv, y, reduction='none')
                        weights = self.robustness_plugins.mart_reweight(
                            logits_adv, y, self.cfg.mart_weight
                        )
                        ce_loss = (ce_vec * weights).mean()

                        # Add KL term
                        kl_loss = F.kl_div(
                            F.log_softmax(logits_adv / 1.0, dim=1),
                            F.softmax(logits_clean / 1.0, dim=1),
                            reduction='batchmean'
                        )
                        loss = ce_loss + self.cfg.trades_beta * kl_loss
                    else:
                        # Standard TRADES
                        loss = self.robustness_plugins.trades_loss(
                            logits_clean, logits_adv, y, beta=self.cfg.trades_beta
                        )
                    logits_for_margin = logits_adv
                else:
                    logits_adv = self.model(x_adv)

                    if self.cfg.mart_enabled:
                        # MART: per-sample CE with weights
                        ce_vec = F.cross_entropy(logits_adv, y, reduction='none')
                        weights = self.robustness_plugins.mart_reweight(
                            logits_adv, y, self.cfg.mart_weight
                        )
                        loss = (ce_vec * weights).mean()
                    else:
                        # Standard CE
                        loss = F.cross_entropy(logits_adv, y)
                    logits_for_margin = logits_adv

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Non-AMP path
            if self.cfg.trades_beta > 0:
                logits_clean = self.model(x_clean)
                logits_adv = self.model(x_adv)

                if self.cfg.mart_enabled:
                    # TRADES with MART
                    ce_vec = F.cross_entropy(logits_adv, y, reduction='none')
                    weights = self.robustness_plugins.mart_reweight(
                        logits_adv, y, self.cfg.mart_weight
                    )
                    ce_loss = (ce_vec * weights).mean()

                    kl_loss = F.kl_div(
                        F.log_softmax(logits_adv / 1.0, dim=1),
                        F.softmax(logits_clean / 1.0, dim=1),
                        reduction='batchmean'
                    )
                    loss = ce_loss + self.cfg.trades_beta * kl_loss
                else:
                    loss = self.robustness_plugins.trades_loss(
                        logits_clean, logits_adv, y, beta=self.cfg.trades_beta
                    )
                logits_for_margin = logits_adv
            else:
                logits_adv = self.model(x_adv)

                if self.cfg.mart_enabled:
                    # MART: per-sample CE with weights
                    ce_vec = F.cross_entropy(logits_adv, y, reduction='none')
                    weights = self.robustness_plugins.mart_reweight(
                        logits_adv, y, self.cfg.mart_weight
                    )
                    loss = (ce_vec * weights).mean()
                else:
                    loss = F.cross_entropy(logits_adv, y)
                logits_for_margin = logits_adv

            loss.backward()
            self.optimizer.step()

        # Compute margin metric
        with torch.no_grad():
            tl = logits_for_margin.gather(1, y.unsqueeze(1)).squeeze()
            other = logits_for_margin.clone()
            other.scatter_(1, y.unsqueeze(1), float('-inf'))
            mo = other.max(dim=1)[0]
            margin = (tl - mo).mean().item()
            self.metrics.batch_margin_mean.append(margin)

        self.metrics.total_forward_passes += x_adv.size(0)
        self.metrics.batch_loss_mean.append(float(loss.item()))

        return float(loss.item())

    def _apply_chi_actions(self):
        """Apply CHI nudges to training - with proper clamping"""
        if not self.chi_active or self.cfg.mode != "heap":
            return

        # Check for plateau or low entropy
        entropy = self.routing_tree.compute_entropy() if self.routing_tree else 1.0

        if entropy < 0.5:  # Low diversity
            # Reduce heap ratios temporarily
            self.cfg.heap_ratio *= 0.8
            self.cfg.global_heap_ratio *= 0.8

            # Clamp and renormalize to keep sane
            self.cfg.heap_ratio = max(0.0, min(0.9, self.cfg.heap_ratio))
            self.cfg.global_heap_ratio = max(0.0, min(0.9, self.cfg.global_heap_ratio))

            # Ensure sum doesn't exceed 0.9
            total = self.cfg.heap_ratio + self.cfg.global_heap_ratio
            if total > 0.9:
                scale = 0.9 / total
                self.cfg.heap_ratio *= scale
                self.cfg.global_heap_ratio *= scale

            logger.info(f"CHI: Low entropy {entropy:.3f}, adjusted heap ratios to "
                       f"{self.cfg.heap_ratio:.2f}/{self.cfg.global_heap_ratio:.2f}")

        # Check for learning plateau
        if len(self.metrics.batch_loss_mean) > 100:
            recent_losses = self.metrics.batch_loss_mean[-100:]
            if np.std(recent_losses) < 0.01:
                # Boost feature weights
                for key in self.feature_weights:
                    self.feature_weights[key] = min(2.0, self.feature_weights[key] * 1.1)
                logger.info("CHI: Plateau detected, boosting feature weights")

    def _log_metrics(self):
        """Log efficiency metrics"""
        if self.cfg.mode == "heap" and self.routing_tree:
            entropy = self.routing_tree.compute_entropy()
            self.metrics.bucket_entropy.append(entropy)

            # Compute heap stats
            total_heap_size = sum(len(lh.heap) for lh in self.routing_tree.leaves.values())
            total_heap_size += len(self.routing_tree.global_heap.heap)

            if total_heap_size > 0:
                ages = []
                for lh in self.routing_tree.leaves.values():
                    ages.extend([self.global_step - v.generation_step for _, _, v in lh.heap])
                ages.extend([self.global_step - v.generation_step for _, _, v in self.routing_tree.global_heap.heap])
                if ages:
                    self.metrics.heap_age_mean.append(float(np.mean(ages)))
                    self.metrics.heap_age_median.append(float(np.median(ages)))
        else:
            # keep series lengths consistent across modes
            self.metrics.bucket_entropy.append(0.0)

    # --- end of class HeapCHITrainer ---

# ---------- evaluation helpers ----------
def _evaluate_model(model, dataloader, device, normalized, eps, step_size, steps, metrics):
    model.eval()
    clean_correct = 0
    robust_correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            logits = model(Attacks.normalize(x) if normalized else x)
            clean_correct += (logits.argmax(1) == y).sum().item()

        x_adv = Attacks.pgd(
            model, x, y,
            eps, step_size, steps,
            str(device), normalized=normalized, set_eval=True
        )
        metrics.pgd_calls_eval += steps * x.size(0)

        with torch.no_grad():
            logits_adv = model(Attacks.normalize(x_adv) if normalized else x_adv)
            robust_correct += (logits_adv.argmax(1) == y).sum().item()

        total += y.size(0)

    return {"clean_acc": clean_correct/total, "robust_acc": robust_correct/total}


def _comprehensive_evaluate_model(model, testloader, device, normalized, eps, step_size,
                                  steps, restarts, metrics):
    model.eval()
    # Clean
    clean_correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = model(Attacks.normalize(x) if normalized else x)
            clean_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    clean = clean_correct / total

    # Robust (multi-restart)
    robust_correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        batch_correct = torch.ones(x.size(0), dtype=torch.bool, device=device)
        for _ in range(restarts):
            x_adv = Attacks.pgd(
                model, x, y,
                eps, step_size, steps,
                str(device), random_start=True, normalized=normalized, set_eval=True
            )
            metrics.pgd_calls_eval += steps * x.size(0)
            with torch.no_grad():
                logits = model(Attacks.normalize(x_adv) if normalized else x_adv)
                batch_correct &= (logits.argmax(1) == y)
        robust_correct += batch_correct.sum().item()
        total += y.size(0)
    robust = robust_correct / total
    return {"clean_acc": clean, "robust_acc": robust}


# Patch the trainer to use the helpers above
def _trainer_eval(self, dataloader):
    return _evaluate_model(
        self.model, dataloader, self.device, self.cfg.use_normalization,
        self.cfg.epsilon, self.cfg.pgd_step_size, self.cfg.eval_pgd_steps, self.metrics
    )

def _trainer_comp_eval(self, testloader, valloader):
    return _comprehensive_evaluate_model(
        self.model, testloader, self.device, self.cfg.use_normalization,
        self.cfg.epsilon, self.cfg.pgd_step_size, self.cfg.eval_pgd_steps,
        self.cfg.eval_pgd_restarts, self.metrics
    )

HeapCHITrainer._evaluate = _trainer_eval
HeapCHITrainer._comprehensive_evaluate = _trainer_comp_eval


# ---------- checkpointing & results ----------
def _save_checkpoint(self, epoch: int, is_best: bool = False):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.scheduler.state_dict(),
        "best_val_robust": self.best_val_robust,
        "config": asdict(self.cfg),
    }
    path = os.path.join(self.cfg.experiment_dir, "best_model.pth" if is_best
                        else f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(ckpt, path)

def _save_results(self, test_metrics: Dict[str, float]):
    pgd = max(self.metrics.pgd_calls_train, 1)
    efficiency_score = test_metrics["robust_acc"] * (10000.0 / pgd)
    out = {
        "config": asdict(self.cfg),
        "test_metrics": test_metrics,
        "history": dict(self.history),
        "best_val_robust": self.best_val_robust,
        "efficiency_metrics": self.metrics.to_dict(),
        "efficiency_score": efficiency_score,
        "total_wall_time": time.time() - self.metrics.start_time,
    }
    with open(os.path.join(self.cfg.experiment_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Results saved to {self.cfg.experiment_dir}")

HeapCHITrainer._save_checkpoint = _save_checkpoint
HeapCHITrainer._save_results = _save_results


# ---------- data loaders ----------
def get_cifar10_loaders(cfg: Config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    full_train = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform_train)
    testset    = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=transform_test)

    # Subset semantics: 0/None -> full 50k
    if cfg.subset_size is not None and cfg.subset_size > 0:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.permutation(len(full_train))[:cfg.subset_size]
        subset = Subset(full_train, idx.tolist())
        train_size = int((1.0 - cfg.val_split) * len(subset))
        val_size = len(subset) - train_size
        g = torch.Generator().manual_seed(cfg.seed)
        trainset, valset = torch.utils.data.random_split(subset, [train_size, val_size], generator=g)
    else:
        # full dataset split
        train_size = int((1.0 - cfg.val_split) * len(full_train))
        val_size = len(full_train) - train_size
        g = torch.Generator().manual_seed(cfg.seed)
        trainset, valset = torch.utils.data.random_split(full_train, [train_size, val_size], generator=g)

    kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    trainloader = DataLoader(trainset, shuffle=True, drop_last=True, **kwargs)
    valloader   = DataLoader(valset, shuffle=False, **kwargs)
    testloader  = DataLoader(testset, shuffle=False, **kwargs)
    return trainloader, valloader, testloader


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="heap_chi_corrected")
    parser.add_argument("--mode", default="heap", choices=["heap","no_heap","random_buffer","hard_mining","fifo"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=0, help="0 => full 50k")
    parser.add_argument("--val_split", type=float, default=0.1)

    # attacks & eval
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--pgd_step_size", type=float, default=2/255)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--eval_pgd_steps", type=int, default=20)
    parser.add_argument("--eval_pgd_restarts", type=int, default=2)

    # heap
    parser.add_argument("--heap_max_size", type=int, default=2000)
    parser.add_argument("--heap_ratio", type=float, default=0.5)
    parser.add_argument("--global_heap_ratio", type=float, default=0.3)
    parser.add_argument("--k_per_batch", type=int, default=128)
    parser.add_argument("--leaf_heap_size", type=int, default=512)
    parser.add_argument("--global_heap_size", type=int, default=128)

    # misc
    parser.add_argument("--output_dir", default="./experiments")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--model_name", default="resnet18",
                        choices=["resnet18","resnet34","mobilenet_v2"])
    parser.add_argument("--pgd_budget_train", type=int, default=0, help="0 => no budget")
    parser.add_argument("--focus_mode", default="none", choices=["none","burst"])
    parser.add_argument("--burst_len", type=int, default=200)
    parser.add_argument("--trades_beta", type=float, default=0.0)
    parser.add_argument("--initial_bank_size", type=int, default=1000,
                    help="Number of FGSM variants to seed the heap after warmup")



    args = parser.parse_args()

    cfg = Config(
        experiment_name=args.experiment_name,
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=None if args.subset_size == 0 else int(args.subset_size),
        val_split=args.val_split,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        pgd_step_size=args.pgd_step_size,
        eval_pgd_steps=args.eval_pgd_steps,
        eval_pgd_restarts=args.eval_pgd_restarts,
        heap_max_size=args.heap_max_size,
        heap_ratio=args.heap_ratio,
        global_heap_ratio=args.global_heap_ratio,
        k_per_batch=args.k_per_batch,
        leaf_heap_size=args.leaf_heap_size,
        global_heap_size=args.global_heap_size,
        human_in_loop=False,  # CHI OFF
        output_dir=args.output_dir,
        model_name=args.model_name,
        pgd_budget_train=None if args.pgd_budget_train == 0 else args.pgd_budget_train,
        focus_mode=args.focus_mode,        
        burst_len=args.burst_len,
        initial_bank_size=args.initial_bank_size,
        trades_beta=args.trades_beta
    )

    trainloader, valloader, testloader = get_cifar10_loaders(cfg)
    trainer = HeapCHITrainer(cfg)
    model, history, test_metrics = trainer.train(trainloader, valloader, testloader)

    # Pretty print
    title = "HEAP" if cfg.mode == "heap" else cfg.mode.upper()
    total_time = time.time() - trainer.metrics.start_time
    eff = test_metrics["robust_acc"] * (10000.0 / max(1, trainer.metrics.pgd_calls_train))
    print("\nFINAL RESULTS (" + title + ")")
    print("="*60)
    print(f"Test Clean Accuracy:  {test_metrics['clean_acc']:.4f}")
    print(f"Test Robust Accuracy: {test_metrics['robust_acc']:.4f}")
    print(f"Best Val Robust:      {trainer.best_val_robust:.4f}")
    print(f"Total Time:           {total_time:.1f}s")
    print(f"PGD Calls (train):    {trainer.metrics.pgd_calls_train}")
    print(f"Unique Variants Seen: {len(trainer.metrics.unique_variants_seen)}")
    print(f"Efficiency Score:     {eff:.4f}")


if __name__ == "__main__":
    main()
