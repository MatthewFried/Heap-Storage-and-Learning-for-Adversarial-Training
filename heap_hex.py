#!/usr/bin/env python3
"""
Heap-Guided Adversarial Robustness - Enhanced for Efficiency Demonstration
Focus: Compute-aware scheduling that maximizes robustness per unit compute
"""

import os
import json
import math
import time
import heapq
import random
import copy
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Any, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================= Utility Functions =========================
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# CIFAR-10 normalization constants
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# ========================= Enhanced Configuration =========================
@dataclass
class Config:
    """Configuration with efficiency-focused parameters."""
    
    # Experiment settings
    experiment_name: str = "heap_efficient"
    mode: str = "heap"  # "heap", "no_heap", "random_buffer", "hard_mining", "fifo"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    
    # Time budget mode
    time_budget_minutes: Optional[float] = None  # Stop after X minutes
    
    # Data
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    subset_size: Optional[int] = None
    val_split: float = 0.1
    use_normalization: bool = True  # Following advice: usually more stable
    
    # Training
    epochs: int = 50
    warmup_epochs: int = 1  # Clean training warmup
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: List[int] = None
    
    # Attack parameters
    epsilon: float = 8/255
    pgd_steps: int = 10
    pgd_step_size: float = 2/255
    eval_pgd_steps: int = 20
    eval_pgd_restarts: int = 2
    
    # Heap parameters - UPDATED
    heap_max_size: int = 2000  # Smaller for faster turnover on small datasets
    initial_bank_size: int = 1000  # 1-2k as suggested
    heap_ratio: float = 0.5  # Proportion from heap (vs fresh PGD) - RENAMED for clarity
    k_per_batch: int = 128
    diversity_k_factor: int = 3
    
    # Delta refresh
    delta_refresh_rate: float = 0.2
    delta_refresh_interval: int = 100
    
    # Bucket strategy
    bucket_strategy: str = "class_hardness"  # "class", "class_hardness", "kmeans"
    num_buckets: int = 20  # For kmeans
    tree_refit_interval: int = 390  # Refit every ~epoch on 10k subset
    tree_max_depth: int = 3
    tree_min_samples_leaf: int = 20
    
    # Scoring weights
    weight_hardness: float = 0.5  # Increased for hardness focus
    weight_bucket: float = 0.2
    weight_novelty: float = 0.15
    weight_features: float = 0.1
    weight_age_penalty: float = 0.05
    
    # Saturation parameters
    saturation_enabled: bool = True
    saturation_threshold: float = 0.7
    saturation_rate: float = 0.05
    saturation_neighbors: int = 50
    saturation_reheap_interval: int = 500
    
    # CHI/HITL parameters
    human_in_loop: bool = False
    chi_cooldown: int = 200
    chi_decay_rate: float = 0.95
    chi_budget: int = 5
    chi_web_port: int = 5000  # Web server port
    
    # Evaluation and logging
    eval_interval: int = 2
    log_interval: int = 50
    save_interval: int = 10
    metrics_log_interval: int = 20  # Log efficiency metrics frequently
    early_stopping_patience: int = 10
    
    # Output
    output_dir: str = "./experiments"
    
    def __post_init__(self):
        if self.lr_milestones is None:
            self.lr_milestones = [int(self.epochs * 0.5), int(self.epochs * 0.75)]
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.experiment_dir = os.path.join(
            self.output_dir, 
            f"{self.experiment_name}_{self.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

# ========================= Efficiency Metrics Tracker =========================
@dataclass
class EfficiencyMetrics:
    """Track all compute and efficiency metrics."""
    
    # Compute tracking
    pgd_calls_train: int = 0
    fgsm_calls_train: int = 0
    pgd_calls_eval: int = 0
    total_forward_passes: int = 0
    
    # Time tracking
    start_time: float = field(default_factory=time.time)
    wall_clock_seconds: List[float] = field(default_factory=list)
    
    # Heap dynamics
    heap_turnover_rate: List[float] = field(default_factory=list)
    heap_age_mean: List[float] = field(default_factory=list)
    heap_age_median: List[float] = field(default_factory=list)
    unique_variants_seen: Set[int] = field(default_factory=set)
    
    # Diversity metrics
    bucket_entropy: List[float] = field(default_factory=list)
    per_class_coverage: List[Dict[int, float]] = field(default_factory=list)
    novelty_mean: List[float] = field(default_factory=list)
    feature_diversity: List[Dict[str, float]] = field(default_factory=list)
    
    # Batch quality
    batch_loss_mean: List[float] = field(default_factory=list)
    batch_margin_mean: List[float] = field(default_factory=list)
    heap_vs_fresh_ratio: List[Tuple[float, float]] = field(default_factory=list)
    
    # Anytime performance
    robust_acc_timeline: List[Tuple[float, float]] = field(default_factory=list)  # (time, acc)
    robust_acc_vs_pgd: List[Tuple[int, float]] = field(default_factory=list)  # (pgd_calls, acc)
    
    def log_snapshot(self, robust_acc: float = None):
        """Record current state."""
        elapsed = time.time() - self.start_time
        self.wall_clock_seconds.append(elapsed)
        
        if robust_acc is not None:
            self.robust_acc_timeline.append((elapsed, robust_acc))
            self.robust_acc_vs_pgd.append((self.pgd_calls_train, robust_acc))
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
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
            'per_class_coverage': self.per_class_coverage,  # FIX: Include class coverage
        }

# ========================= Attack Implementations (unchanged) =========================
class Attacks:
    """Attack implementations with proper gradient handling and normalization support."""
    
    @staticmethod
    def normalize(x: Tensor) -> Tensor:
        """Normalize images to CIFAR-10 statistics."""
        device = x.device
        mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        return (x - mean) / std
    
    @staticmethod
    def denormalize(x: Tensor) -> Tensor:
        """Denormalize images from CIFAR-10 statistics."""
        device = x.device
        mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        return x * std + mean
    
    @staticmethod
    def fgsm(model: nn.Module, x: Tensor, y: Tensor, eps: float, device: str, 
             normalized: bool = False, set_eval: bool = True) -> Tensor:
        """FGSM attack with normalization support."""
        was_training = model.training
        if set_eval:
            model.eval()
        
        x = x.clone().detach().to(device).requires_grad_(True)
        y = y.to(device)
        
        if normalized:
            x_norm = Attacks.normalize(x)
            logits = model(x_norm)
        else:
            logits = model(x)
        
        loss = F.cross_entropy(logits, y)
        
        model.zero_grad()
        loss.backward()
        
        x_adv = x + eps * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        if was_training and set_eval:
            model.train()
        
        return x_adv.detach()
    
    @staticmethod
    def pgd(model: nn.Module, x: Tensor, y: Tensor, eps: float, 
            step_size: float, steps: int, device: str, random_start: bool = True,
            normalized: bool = False, set_eval: bool = True) -> Tensor:
        """PGD attack with normalization support."""
        was_training = model.training
        if set_eval:
            model.eval()
        
        x = x.to(device)
        y = y.to(device)
        
        if random_start:
            x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = x.clone()
        
        for _ in range(steps):
            x_adv.requires_grad_(True)
            
            if normalized:
                x_norm = Attacks.normalize(x_adv)
                logits = model(x_norm)
            else:
                logits = model(x_adv)
            
            loss = F.cross_entropy(logits, y)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            x_adv = x_adv.detach() + step_size * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
        
        if was_training and set_eval:
            model.train()
        
        return x_adv.detach()

# ========================= Feature Extraction (unchanged) =========================
class FastFeatureExtractor:
    """Efficient feature extraction with normalization support."""
    
    def __init__(self, model: nn.Module, device: str, use_normalization: bool = False):
        self.model = model
        self.device = device
        self.use_normalization = use_normalization
    
    def extract_features(self, x_clean: Tensor, x_adv: Tensor, y: Tensor) -> Dict[str, Tensor]:
        """Extract features with model in eval mode."""
        features = {}
        
        # Perturbation features
        delta = x_adv - x_clean
        batch_size = delta.size(0)
        
        delta_flat = delta.view(batch_size, -1)
        features['l2_norm'] = delta_flat.norm(p=2, dim=1) / math.sqrt(delta_flat.size(1))
        features['linf_norm'] = delta_flat.abs().max(dim=1)[0]
        
        delta_spatial = delta.abs().mean(dim=1, keepdim=True)
        total = delta_spatial.view(batch_size, -1).sum(dim=1) + 1e-8
        maximum = delta_spatial.view(batch_size, -1).max(dim=1)[0]
        features['spatial_concentration'] = maximum / total
        
        tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().sum(dim=(1, 2, 3))
        tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().sum(dim=(1, 2, 3))
        features['total_variation'] = (tv_h + tv_w) / delta[0].numel()
        
        # Model features
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            if self.use_normalization:
                x_clean_norm = Attacks.normalize(x_clean)
                x_adv_norm = Attacks.normalize(x_adv)
                logits_clean = self.model(x_clean_norm)
                logits_adv = self.model(x_adv_norm)
            else:
                logits_clean = self.model(x_clean)
                logits_adv = self.model(x_adv)
            
            pred_adv = logits_adv.argmax(dim=1)
            features['attack_success'] = (pred_adv != y).float()
            
            probs_clean = F.softmax(logits_clean, dim=1)
            probs_adv = F.softmax(logits_adv, dim=1)
            clean_conf = probs_clean.gather(1, y.unsqueeze(1)).squeeze()
            adv_conf = probs_adv.gather(1, y.unsqueeze(1)).squeeze()
            features['confidence_drop'] = clean_conf - adv_conf
            
            true_logits = logits_adv.gather(1, y.unsqueeze(1)).squeeze()
            other_logits = logits_adv.clone()
            other_logits.scatter_(1, y.unsqueeze(1), float('-inf'))
            max_other = other_logits.max(dim=1)[0]
            features['margin_logit'] = true_logits - max_other
            features['normalized_margin'] = torch.sigmoid(features['margin_logit'])
            
            # Hardness score (1 - normalized_margin)
            features['hardness'] = 1.0 - features['normalized_margin']
            
            features['entropy'] = -(probs_adv * torch.log(probs_adv + 1e-8)).sum(dim=1)
        
        if was_training:
            self.model.train()
        
        # Frequency features
        if batch_size > 0:
            gray_delta = delta.mean(dim=1, keepdim=True)
            fft = torch.fft.rfft2(gray_delta, s=(16, 16))
            fft_mag = torch.abs(fft)
            
            low_freq = fft_mag[:, :, :4, :4].mean(dim=(1, 2, 3))
            high_freq = fft_mag[:, :, 4:, 4:].mean(dim=(1, 2, 3))
            total_freq = fft_mag.mean(dim=(1, 2, 3)) + 1e-8
            
            features['low_freq_ratio'] = low_freq / total_freq
            features['high_freq_ratio'] = high_freq / total_freq
        
        return features

# ========================= Enhanced Variant Class =========================
class Variant:
    """Variant with unique ID for tracking."""
    
    _id_counter = 0
    
    __slots__ = ('id', 'x_clean', 'delta', 'x_clean_idx', 'y', 'features', 'bucket_id', 
                 'bucket_score', 'novelty', 'age', 'score', 'last_scored_step',
                 'generation_step', 'refresh_count', '_feature_vector')
    
    def __init__(self, x_clean: Tensor, x_adv: Tensor, y: Tensor, 
                 features: Dict[str, float], idx: int = -1, generation_step: int = 0):
        self.id = Variant._id_counter
        Variant._id_counter += 1
        
        self.x_clean = x_clean.cpu().half()
        self.delta = (x_adv - x_clean).cpu().half()
        self.x_clean_idx = idx
        self.y = y.item() if isinstance(y, Tensor) else y
        
        self.features = {k: v.item() if isinstance(v, Tensor) else v 
                        for k, v in features.items()}
        
        self.bucket_id = 0
        self.bucket_score = 0.5
        self.novelty = 0.5
        self.age = 0
        self.score = 0.0
        self.last_scored_step = 0
        self.generation_step = generation_step
        self.refresh_count = 0
        self._feature_vector = None
    
    def get_feature_vector(self) -> np.ndarray:
        """Get normalized feature vector."""
        if self._feature_vector is None:
            feat_names = ['hardness', 'l2_norm', 'linf_norm', 
                         'spatial_concentration', 'high_freq_ratio']
            self._feature_vector = np.array([
                self.features.get(k, 0.0) for k in feat_names
            ])
        return self._feature_vector
    
    def reconstruct_adv(self, device: str) -> Tensor:
        """Reconstruct adversarial from stored clean and delta."""
        x_clean = self.x_clean.float().to(device)
        delta = self.delta.float().to(device)
        return torch.clamp(x_clean + delta, 0.0, 1.0)
    
    def refresh_delta(self, new_delta: Tensor):
        """Refresh the perturbation with a new one."""
        self.delta = new_delta.cpu().half()
        self.refresh_count += 1
        self._feature_vector = None

# ========================= Enhanced Heap with Tracking =========================
class FixedVariantHeap:
    """Min-heap with comprehensive tracking."""
    
    def __init__(self, max_size: int, cfg: Config):
        self.max_size = max_size
        self.cfg = cfg
        self.heap = []
        self.counter = 0
        
        self.bucket_counts = defaultdict(int)
        self.class_counts = defaultdict(int)
        self.recent_features = deque(maxlen=512)
        self.metrics = defaultdict(float)
        self.last_reheap_step = 0
        
        # Tracking for turnover
        self.variants_popped_this_epoch = set()
        self.all_variant_ids = set()
    
    def push(self, variant: Variant) -> bool:
        """Add variant to heap."""
        item = (variant.score, self.counter, variant)
        self.counter += 1
        
        self.all_variant_ids.add(variant.id)
        
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
            self._update_on_add(variant)
            return True
        else:
            if variant.score > self.heap[0][0]:
                removed = heapq.heapreplace(self.heap, item)
                self._update_on_remove(removed[2])
                self._update_on_add(variant)
                return True
            return False
    
    def pop_diverse_batch(self, k: int) -> List[Variant]:
        """Pop k diverse high-scoring variants with tracking."""
        if not self.heap:
            return []
        
        num_candidates = min(self.cfg.diversity_k_factor * k, len(self.heap))
        candidates = heapq.nlargest(num_candidates, self.heap)
        
        candidate_set = set(candidates)
        self.heap = [item for item in self.heap if item not in candidate_set]
        heapq.heapify(self.heap)
        
        candidate_variants = [item[2] for item in candidates]
        
        selected = self._select_diverse(candidate_variants, k)
        
        # Track popped variants
        for v in selected:
            self.variants_popped_this_epoch.add(v.id)
        
        # Re-add non-selected
        for v in candidate_variants:
            if v not in selected:
                heapq.heappush(self.heap, (v.score, self.counter, v))
                self.counter += 1
        
        for v in selected:
            self._update_on_remove(v)
            self.recent_features.append(v.get_feature_vector())
        
        return selected
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive heap metrics."""
        metrics = {}
        
        if not self.heap:
            return metrics
        
        # Age statistics
        ages = [v.age for _, _, v in self.heap]
        metrics['age_mean'] = np.mean(ages)
        metrics['age_median'] = np.median(ages)
        metrics['age_std'] = np.std(ages)
        
        # Hardness distribution
        hardnesses = [v.features.get('hardness', 0.5) for _, _, v in self.heap]
        metrics['hardness_mean'] = np.mean(hardnesses)
        metrics['hardness_std'] = np.std(hardnesses)
        
        # Bucket entropy
        if self.bucket_counts:
            total = sum(self.bucket_counts.values())
            probs = [count / total for count in self.bucket_counts.values()]
            metrics['bucket_entropy'] = -sum(p * math.log(p + 1e-8) for p in probs)
        else:
            metrics['bucket_entropy'] = 0.0
        
        # Class coverage
        metrics['num_classes_covered'] = len(self.class_counts)
        if self.class_counts:
            total = sum(self.class_counts.values())
            class_probs = [count / total for count in self.class_counts.values()]
            metrics['class_entropy'] = -sum(p * math.log(p + 1e-8) for p in class_probs)
        
        # Novelty
        novelties = [v.novelty for _, _, v in self.heap]
        metrics['novelty_mean'] = np.mean(novelties)
        
        # Turnover rate (FIX: use max_size as denominator)
        metrics['turnover_rate'] = len(self.variants_popped_this_epoch) / max(self.max_size, 1)
        
        return metrics
    
    def reset_epoch_tracking(self):
        """Reset per-epoch tracking."""
        self.variants_popped_this_epoch.clear()
    
    def get_class_distribution(self) -> Dict[int, float]:
        """Get percentage of each class in heap."""
        if not self.heap:
            return {}
        
        total = len(self.heap)
        return {cls: count/total for cls, count in self.class_counts.items()}
    
    def _select_diverse(self, candidates: List[Variant], k: int) -> List[Variant]:
        """Select diverse variants with class and bucket balance."""
        if len(candidates) <= k:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # First, pick best from each class (FIX: sort by score first)
        class_groups = defaultdict(list)
        for v in remaining:
            class_groups[v.y].append(v)
        
        for cls in sorted(class_groups.keys()):
            if len(selected) >= k:
                break
            if class_groups[cls]:
                # Sort by score and take best
                class_groups[cls].sort(key=lambda v: v.score, reverse=True)
                selected.append(class_groups[cls][0])
                remaining.remove(class_groups[cls][0])
        
        # Then fill by feature diversity
        while len(selected) < k and remaining:
            max_min_dist = -1
            best_idx = 0
            
            for i, cand in enumerate(remaining):
                cand_feat = cand.get_feature_vector()
                min_dist = float('inf')
                
                for sel in selected:
                    sel_feat = sel.get_feature_vector()
                    dist = np.linalg.norm(cand_feat - sel_feat)
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _update_on_add(self, variant: Variant):
        self.bucket_counts[variant.bucket_id] += 1
        self.class_counts[variant.y] += 1
        self.metrics['total_pushed'] += 1
    
    def _update_on_remove(self, variant: Variant):
        self.bucket_counts[variant.bucket_id] -= 1
        if self.bucket_counts[variant.bucket_id] <= 0:
            del self.bucket_counts[variant.bucket_id]
        
        self.class_counts[variant.y] -= 1
        if self.class_counts[variant.y] <= 0:
            del self.class_counts[variant.y]
        
        self.metrics['total_popped'] += 1
    
    def reheapify(self, step: int, recompute_score=None):
        """Reheapify after score changes."""
        if step - self.last_reheap_step < self.cfg.saturation_reheap_interval:
            return
        
        self.last_reheap_step = step
        
        new_heap = []
        for _, _, v in self.heap:
            if recompute_score is not None:
                v.score = recompute_score(v)
            new_heap.append((v.score, self.counter, v))
            self.counter += 1
        
        self.heap = new_heap
        heapq.heapify(self.heap)
    
    def size(self) -> int:
        return len(self.heap)

# ========================= CHI/HITL Interface =========================
class CHIInterface:
    """Human-in-the-loop interface for optional guidance."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.budget = cfg.chi_budget
        self.last_query_step = -cfg.chi_cooldown
        self.query_history = []
        self.active_boosts = {}
        self.web_server = None
        self.pending_response = None
        
        # Start web server if in web mode
        if cfg.human_in_loop == "web":
            self._start_web_server()
    
    def _start_web_server(self):
        """Start Flask web server for interactive CHI."""
        try:
            # Check if server already running
            if getattr(self.__class__, "_server_running", False):
                logger.info("CHI Web Server already running; reusing.")
                return
            
            from flask import Flask, render_template_string, request, jsonify
            import threading
            import base64
            from io import BytesIO
            
            self.__class__._server_running = True
            self.app = Flask(__name__)
            self.current_data = {}
            self.response_ready = threading.Event()
            self.web_server = True  # Mark server as available
            
            port = getattr(self.cfg, "chi_web_port", 5000)
            
            @self.app.route('/')
            def index():
                return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>CHI: Heap-Guided Adversarial Training</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
                        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                        .metric { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }
                        .metric h3 { margin: 0 0 5px 0; color: #666; font-size: 14px; }
                        .metric p { margin: 0; font-size: 24px; font-weight: bold; color: #333; }
                        .plot { margin: 20px 0; text-align: center; }
                        .plot img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
                        .actions { margin: 30px 0; }
                        .action-btn { 
                            background: #4CAF50; color: white; padding: 12px 24px; 
                            border: none; border-radius: 5px; cursor: pointer; 
                            margin: 5px; font-size: 16px; transition: all 0.3s;
                        }
                        .action-btn:hover { background: #45a049; transform: translateY(-2px); }
                        .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 20px 0; }
                        .feature-item { 
                            background: #e8f5e9; padding: 10px; border-radius: 5px; 
                            cursor: pointer; transition: all 0.3s; text-align: center;
                        }
                        .feature-item:hover { background: #c8e6c9; }
                        .feature-item.selected { background: #4CAF50; color: white; }
                        input[type="number"] { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 3px; }
                        .status { 
                            background: #fff3cd; border: 1px solid #ffc107; 
                            color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0;
                        }
                    </style>
                    <script>
                        let selectedFeature = null;
                        let selectedBucket = null;
                        
                        function selectFeature(evt, name) {
                            document.querySelectorAll('.feature-item').forEach(el => el.classList.remove('selected'));
                            evt.currentTarget.classList.add('selected');
                            selectedFeature = name;
                        }
                        
                        function sendAction(action) {
                            let data = {action: action};
                            
                            if (action === 'boost_feature' && selectedFeature) {
                                data.feature = selectedFeature;
                                data.boost = parseFloat(document.getElementById('boost-value').value) || 1.5;
                            } else if (action === 'penalize_bucket') {
                                data.bucket_id = parseInt(document.getElementById('bucket-id').value) || 0;
                                data.mult = parseFloat(document.getElementById('penalty-mult').value) || 0.85;
                            } else if (action === 'diversity_burst') {
                                data.steps = parseInt(document.getElementById('burst-steps').value) || 200;
                            }
                            
                            fetch('/respond', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify(data)
                            }).then(() => {
                                document.getElementById('status').innerHTML = 'Response sent! Training will continue...';
                            });
                        }
                        
                        function refreshData() {
                            fetch('/data')
                                .then(response => response.json())
                                .then(data => {
                                    if (data.waiting) {
                                        document.getElementById('content').style.display = 'block';
                                        document.getElementById('step').textContent = data.step;
                                        document.getElementById('entropy').textContent = data.entropy.toFixed(3);
                                        document.getElementById('turnover').textContent = data.turnover.toFixed(3);
                                        document.getElementById('hardness').textContent = data.hardness.toFixed(3);
                                        document.getElementById('heap-size').textContent = data.heap_size;
                                        
                                        if (data.plot) {
                                            document.getElementById('plot').innerHTML = 
                                                '<img src="data:image/png;base64,' + data.plot + '">';
                                        }
                                        
                                        // Update features
                                        let featuresHtml = '';
                                        for (let [name, weight] of Object.entries(data.features)) {
                                            featuresHtml += `
                                                <div class="feature-item" onclick="selectFeature(event, '${name}')">
                                                    <strong>${name}</strong><br>
                                                    ${weight.toFixed(3)}
                                                </div>
                                            `;
                                        }
                                        document.getElementById('features').innerHTML = featuresHtml;
                                        
                                        // Update top buckets
                                        if (data.top_buckets) {
                                            document.getElementById('top-buckets').innerHTML =
                                                data.top_buckets.map(([id,c]) => 
                                                    `<span style="cursor:pointer; margin: 0 5px; color: #4CAF50;" 
                                                           onclick="document.getElementById('bucket-id').value=${id}">
                                                        ${id}:${c}
                                                    </span>`
                                                ).join(' · ');
                                        }
                                        
                                        // Update status
                                        let status = 'CHI Query: ';
                                        if (data.plateau) status += 'PLATEAU DETECTED ';
                                        if (data.low_entropy) status += 'LOW ENTROPY ';
                                        document.getElementById('trigger').textContent = status;
                                    }
                                });
                        }
                        
                        setInterval(refreshData, 2000);
                        refreshData();
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h1>🤖 CHI: Heap-Guided Adversarial Training</h1>
                        
                        <div id="content" style="display:none;">
                            <div class="status">
                                <strong id="trigger">Waiting for CHI trigger...</strong>
                            </div>
                            
                            <div class="metrics">
                                <div class="metric">
                                    <h3>Training Step</h3>
                                    <p id="step">-</p>
                                </div>
                                <div class="metric">
                                    <h3>Bucket Entropy</h3>
                                    <p id="entropy">-</p>
                                </div>
                                <div class="metric">
                                    <h3>Turnover Rate</h3>
                                    <p id="turnover">-</p>
                                </div>
                                <div class="metric">
                                    <h3>Mean Hardness</h3>
                                    <p id="hardness">-</p>
                                </div>
                                <div class="metric">
                                    <h3>Heap Size</h3>
                                    <p id="heap-size">-</p>
                                </div>
                                <div class="metric">
                                    <h3>Top Buckets (id:count)</h3>
                                    <p id="top-buckets" style="font-size: 14px;">-</p>
                                </div>
                            </div>
                            
                            <div class="plot" id="plot">
                                <!-- Visualization will appear here -->
                            </div>
                            
                            <h2>Feature Weights</h2>
                            <div class="feature-grid" id="features">
                                <!-- Features will be populated here -->
                            </div>
                            
                            <div class="actions">
                                <h2>Actions</h2>
                                
                                <div style="margin: 20px 0;">
                                    <button class="action-btn" onclick="sendAction('diversity_burst')">
                                        🎲 Diversity Burst
                                    </button>
                                    Steps: <input type="number" id="burst-steps" value="200" min="50" max="500">
                                </div>
                                
                                <div style="margin: 20px 0;">
                                    <button class="action-btn" onclick="sendAction('boost_feature')">
                                        ⬆️ Boost Selected Feature
                                    </button>
                                    Multiplier: <input type="number" id="boost-value" value="1.5" min="1.1" max="3.0" step="0.1">
                                </div>
                                
                                <div style="margin: 20px 0;">
                                    <button class="action-btn" onclick="sendAction('penalize_bucket')">
                                        ⬇️ Penalize Bucket
                                    </button>
                                    Bucket ID: <input type="number" id="bucket-id" value="0" min="0">
                                    Penalty: <input type="number" id="penalty-mult" value="0.85" min="0.1" max="1.0" step="0.05">
                                </div>
                                
                                <div style="margin: 20px 0;">
                                    <button class="action-btn" onclick="sendAction('skip')" style="background: #9e9e9e;">
                                        ⏭️ Skip
                                    </button>
                                    <button class="action-btn" onclick="sendAction('early_accept_best')" style="background: #ff9800;">
                                        ✅ Accept Best Model
                                    </button>
                                    <button class="action-btn" onclick="sendAction('reset_weights')" style="background: #607d8b;">
                                        🔄 Reset Feature Weights
                                    </button>
                                </div>
                            </div>
                            
                            <div id="status" style="color: green; font-weight: bold; margin-top: 20px;"></div>
                        </div>
                    </div>
                </body>
                </html>
                ''')
            
            @self.app.route('/data')
            def get_data():
                return jsonify(self.current_data)
            
            @self.app.route('/respond', methods=['POST'])
            def respond():
                self.pending_response = request.json
                self.response_ready.set()
                return jsonify({'status': 'ok'})
            
            # Start server in background thread
            self.server_thread = threading.Thread(
                target=lambda: self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info(f"CHI Web Server started at http://localhost:{port}")
            
        except ImportError:
            logger.warning("Flask not installed. Falling back to interactive CLI mode. Run: pip install flask")
            self.web_server = False  # Mark server as unavailable
    
    def should_query(self, step: int, *, plateau: bool, low_entropy: bool) -> bool:
        """Check if CHI should be invoked."""
        if self.budget <= 0:
            return False
        if step - self.last_query_step < self.cfg.chi_cooldown:
            return False
        return plateau or low_entropy
    
    def apply_action(self, trainer, action: Dict[str, Any]):
        """Apply human-guided action to trainer."""
        a = action.get("action")
        
        if a == "boost_feature":
            # Boost a specific feature weight
            f = action["feature"]
            mult = action.get("boost", 1.5)
            if f in trainer.feature_weights:
                trainer.feature_weights[f] *= mult
                logger.info(f"CHI: Boosted {f} by {mult}x")
        
        elif a == "penalize_bucket":
            # Penalize a specific bucket
            bid = int(action["bucket_id"])
            mult = action.get("mult", 0.85)
            if hasattr(trainer, "bucket_manager") and bid in trainer.bucket_manager.bucket_scores:
                trainer.bucket_manager.bucket_scores[bid] *= mult
                trainer.heap.reheapify(trainer.global_step, recompute_score=trainer._compute_score)
                logger.info(f"CHI: Penalized bucket {bid} by {mult}x")
        
        elif a == "diversity_burst":
            # Trigger diversity injection
            trainer._diversity_burst_steps = action.get("steps", 200)
            logger.info(f"CHI: Triggered diversity burst for {trainer._diversity_burst_steps} steps")
        
        elif a == "early_accept_best":
            # Accept current best model
            trainer._load_best_checkpoint()
            logger.info("CHI: Accepted best checkpoint")
        
        elif a == "reset_weights" or a == "reset_feature_weights":
            # Reset feature weights to defaults
            trainer.feature_weights = {
                'hardness': 0.3,
                'confidence_drop': 0.2,
                'high_freq_ratio': 0.15,
                'spatial_concentration': 0.15,
                'entropy': 0.2
            }
            logger.info("CHI: Reset feature weights to defaults")
        
        elif a == "adjust_heap_ratio":
            # Adjust heap:fresh ratio dynamically
            new_ratio = action.get("ratio", 0.5)
            trainer.cfg.heap_ratio = max(0.1, min(0.9, new_ratio))
            logger.info(f"CHI: Adjusted heap ratio to {trainer.cfg.heap_ratio} (heap:{trainer.cfg.heap_ratio:.0%}, fresh:{1-trainer.cfg.heap_ratio:.0%})")
        
        elif a == "skip":
            # Do nothing
            logger.info("CHI: Skipped action")
        
        # Bookkeeping
        self.query_history.append({
            "step": trainer.global_step,
            "action": action,
            "plateau": trainer._last_plateau,
            "entropy": trainer._last_entropy
        })
        self.last_query_step = trainer.global_step
        self.budget -= 1
        
        if a == "boost_feature":
            self.active_boosts[action["feature"]] = (action.get("boost", 1.5), trainer.global_step)
    
    def decay_boosts(self, step: int, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply exponential decay to active boosts."""
        new_weights = weights.copy()
        
        for feature, (boost, start_step) in list(self.active_boosts.items()):
            steps_elapsed = step - start_step
            decay = self.cfg.chi_decay_rate ** (steps_elapsed / self.cfg.chi_cooldown)
            
            if decay < 0.01:
                del self.active_boosts[feature]
            else:
                if feature in new_weights:
                    new_weights[feature] *= (1 + (boost - 1) * decay)
        
        return new_weights
    
    def simulate_human_response(self, trainer, plateau: bool, low_entropy: bool) -> Dict[str, Any]:
        """Simulate deterministic human response for testing."""
        if plateau:
            # On plateau, inject diversity
            return {"action": "diversity_burst", "steps": 200}
        elif low_entropy:
            # On low entropy, penalize dominant bucket
            if hasattr(trainer, 'heap') and trainer.heap.bucket_counts:
                dominant_bucket = max(trainer.heap.bucket_counts, key=trainer.heap.bucket_counts.get)
                return {"action": "penalize_bucket", "bucket_id": dominant_bucket, "mult": 0.85}
        
        # Default: boost least-weighted feature
        min_feature = min(trainer.feature_weights, key=trainer.feature_weights.get)
        return {"action": "boost_feature", "feature": min_feature, "boost": 1.5}
    
    def web_human_response(self, trainer, plateau: bool, low_entropy: bool) -> Dict[str, Any]:
        """Get human input via web interface."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import base64
        from io import BytesIO
        
        # Prepare data for web interface
        heap_metrics = {}
        top_buckets = []
        
        if hasattr(trainer, 'heap'):
            heap_metrics = trainer.heap.compute_metrics()
            if trainer.heap.bucket_counts:
                top_buckets = sorted(trainer.heap.bucket_counts.items(), 
                                   key=lambda kv: kv[1], reverse=True)[:8]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Loss history
        ax = axes[0, 0]
        if hasattr(trainer, '_loss_hist') and len(trainer._loss_hist) > 0:
            ax.plot(list(trainer._loss_hist), color='steelblue', linewidth=2)
            ax.axhline(y=np.mean(trainer._loss_hist), color='red', linestyle='--', alpha=0.5, label='Mean')
        ax.set_xlabel('Recent Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (Plateau Detection)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Heap metrics over time
        ax = axes[0, 1]
        if trainer.metrics.bucket_entropy:
            ax.plot(trainer.metrics.bucket_entropy[-100:], label='Bucket Entropy', color='green')
        if trainer.metrics.novelty_mean:
            ax2 = ax.twinx()
            ax2.plot(trainer.metrics.novelty_mean[-100:], label='Novelty', color='orange', alpha=0.7)
            ax2.set_ylabel('Novelty', color='orange')
        ax.set_xlabel('Logging Steps')
        ax.set_ylabel('Entropy', color='green')
        ax.set_title('Heap Diversity Metrics')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Class distribution in heap
        ax = axes[1, 0]
        if hasattr(trainer, 'heap'):
            class_dist = trainer.heap.get_class_distribution()
            if class_dist:
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                ax.bar(classes, counts, color='coral')
                ax.set_xlabel('Class')
                ax.set_ylabel('Proportion in Heap')
                ax.set_title('Class Coverage')
        
        # Plot 4: Feature weights
        ax = axes[1, 1]
        features = list(trainer.feature_weights.keys())
        weights = list(trainer.feature_weights.values())
        colors = ['red' if w > 0.25 else 'yellow' if w > 0.15 else 'green' for w in weights]
        ax.barh(features, weights, color=colors)
        ax.set_xlabel('Weight')
        ax.set_title('Feature Importance')
        ax.set_xlim(0, max(weights) * 1.2 if weights else 1)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Update web interface data
        self.current_data = {
            'waiting': True,
            'step': trainer.global_step,
            'plateau': plateau,
            'low_entropy': low_entropy,
            'entropy': heap_metrics.get('bucket_entropy', 0),
            'turnover': heap_metrics.get('turnover_rate', 0),
            'hardness': heap_metrics.get('hardness_mean', 0),
            'heap_size': trainer.heap.size() if hasattr(trainer, 'heap') else 0,
            'features': trainer.feature_weights,
            'top_buckets': top_buckets,  # Added top buckets
            'plot': plot_base64
        }
        
        # Wait for response
        logger.info(f"CHI: Waiting for web response at http://localhost:{self.cfg.chi_web_port}")
        self.response_ready.clear()
        self.pending_response = None
        
        # Wait with timeout
        if self.response_ready.wait(timeout=300):  # 5 minute timeout
            response = self.pending_response
            self.current_data = {'waiting': False}
            return response if response else {"action": "skip"}
        else:
            logger.warning("CHI: Timeout waiting for response, skipping")
            return {"action": "skip"}

# ========================= Bucket Manager =========================
# ========================= Bucket Manager =========================
class BucketManager:
    """Manage buckets using different strategies (with stable scaling for k-means)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.strategy = cfg.bucket_strategy

        # separate models per strategy to avoid clashes
        self.tree_model = None
        self.kmeans_model = None
        self.scaler = None          # for kmeans only

        self.bucket_scores = {}
        self.last_refit_step = 0

    def fit(self, variants: List[Variant], step: int = 0):
        """Fit / refit bucket model (scales features for k-means)."""
        if not variants:
            return

        # raw feature matrix
        X = np.array([v.get_feature_vector() for v in variants], dtype=np.float32)
        # make sure no NaNs/Infs get in
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.strategy == "class":
            # class buckets
            groups = defaultdict(list)
            for v in variants:
                v.bucket_id = v.y
                groups[v.bucket_id].append(v)
            for bid, group in groups.items():
                self._update_bucket_score(bid, group)

        elif self.strategy == "class_hardness":
            # class × hardness (3 bins)
            groups = defaultdict(list)
            for v in variants:
                hardness_bin = int(v.features.get('hardness', 0.5) * 3)
                v.bucket_id = v.y * 10 + hardness_bin
                groups[v.bucket_id].append(v)
            for bid, group in groups.items():
                self._update_bucket_score(bid, group)

        elif self.strategy == "kmeans":
            # ---- key stability fixes: Standardize + keep model/scaler ----
            n_clusters = int(min(self.cfg.num_buckets, len(variants)))
            if n_clusters < 2:
                # degenerate case: leave everything in one bucket
                for v in variants:
                    v.bucket_id = 0
                self.bucket_scores = {0: float(np.mean([vv.features.get('hardness', 0.5) for vv in variants]))}
            else:
                # fit scaler on current pool
                self.scaler = StandardScaler()
                Xs = self.scaler.fit_transform(X)

                # a modest k helps stability (you already default to 20)
                self.kmeans_model = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=self.cfg.seed,
                    batch_size=256,
                    n_init=10,                # more robust init
                    max_no_improvement=20,    # early-stop churn
                    reassignment_ratio=0.01
                )
                self.kmeans_model.fit(Xs)
                labels = self.kmeans_model.labels_

                # assign cluster labels to variants
                for i, v in enumerate(variants):
                    v.bucket_id = int(labels[i])

                # compute bucket scores
                for bucket_id in np.unique(labels):
                    bucket_variants = [v for v in variants if v.bucket_id == int(bucket_id)]
                    self._update_bucket_score(int(bucket_id), bucket_variants)

        else:  # "tree" (decision tree on features)
            from sklearn.tree import DecisionTreeClassifier

            y = np.array([v.y for v in variants], dtype=np.int64)

            self.tree_model = DecisionTreeClassifier(
                max_depth=self.cfg.tree_max_depth,
                min_samples_leaf=self.cfg.tree_min_samples_leaf,
                random_state=self.cfg.seed
            )
            self.tree_model.fit(X, y)
            leaf_ids = self.tree_model.apply(X)

            for i, v in enumerate(variants):
                v.bucket_id = int(leaf_ids[i])

            for leaf_id in np.unique(leaf_ids):
                bucket_variants = [v for v in variants if v.bucket_id == int(leaf_id)]
                self._update_bucket_score(int(leaf_id), bucket_variants)

        # set bucket_score on each variant (fallback 0.5)
        for v in variants:
            v.bucket_score = self.bucket_scores.get(v.bucket_id, 0.5)

        self.last_refit_step = step

    def should_refit(self, step: int) -> bool:
        """Refit buckets about once per epoch (uses tree_refit_interval for all strategies)."""
        return step - self.last_refit_step >= self.cfg.tree_refit_interval

    def assign_bucket(self, variant: Variant):
        """Assign bucket to a *new* variant using the fitted model."""
        if self.strategy == "class":
            variant.bucket_id = variant.y

        elif self.strategy == "class_hardness":
            hardness_bin = int(variant.features.get('hardness', 0.5) * 3)
            variant.bucket_id = variant.y * 10 + hardness_bin

        elif self.strategy == "kmeans" and (self.kmeans_model is not None) and (self.scaler is not None):
            x = variant.get_feature_vector().reshape(1, -1).astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            xs = self.scaler.transform(x)
            variant.bucket_id = int(self.kmeans_model.predict(xs)[0])

        elif self.tree_model is not None:  # "tree"
            x = variant.get_feature_vector().reshape(1, -1).astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            variant.bucket_id = int(self.tree_model.apply(x)[0])

        # assign bucket score
        variant.bucket_score = self.bucket_scores.get(variant.bucket_id, 0.5)

    def _update_bucket_score(self, bucket_id: int, variants: List[Variant]):
        """Bucket score = mean hardness (works well per-compute)."""
        if not variants:
            return
        hardnesses = [v.features.get('hardness', 0.5) for v in variants]
        self.bucket_scores[bucket_id] = float(np.mean(hardnesses))


# ========================= Main Trainer with Efficiency Focus =========================
class HeapRobustTrainer:
    """Trainer focused on demonstrating efficiency gains."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        set_seed(cfg.seed)
        
        # Model and optimizer
        self.model = self._build_model()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=cfg.lr, 
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=cfg.lr_milestones
        )
        
        # AMP
        self.scaler = GradScaler('cuda') if cfg.use_amp and cfg.device == "cuda" else None
        
        # Components
        self.feature_extractor = FastFeatureExtractor(
            self.model, str(self.device), cfg.use_normalization
        )
        
        # Mode-specific setup
        if cfg.mode == "heap":
            self.heap = FixedVariantHeap(cfg.heap_max_size, cfg)
            self.bucket_manager = BucketManager(cfg)
        elif cfg.mode in ["random_buffer", "hard_mining"]:
            self.buffer = deque(maxlen=cfg.heap_max_size)
        elif cfg.mode == "fifo":
            self.queue = deque(maxlen=cfg.heap_max_size)
        
        # Metrics tracking
        self.metrics = EfficiencyMetrics()
        self.history = defaultdict(list)
        self.recent_phi_cache = deque(maxlen=512)
        
        # Feature weights
        self.feature_weights = {
            'hardness': 0.3,
            'confidence_drop': 0.2,
            'high_freq_ratio': 0.15,
            'spatial_concentration': 0.15,
            'entropy': 0.2
        }
        
        # Early stopping
        self.best_val_robust = 0
        self.patience_counter = 0
        
        # CHI/HITL
        self.chi = CHIInterface(cfg) if cfg.human_in_loop else None
        self._loss_hist = deque(maxlen=200)
        self._diversity_burst_steps = 0
        self._last_plateau = False
        self._last_entropy = 1.0
        
        self.global_step = 0
    
    def _build_model(self) -> nn.Module:
        """Build ResNet18 for CIFAR-10."""
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)
        return model.to(self.device)
    
    def train(self, trainloader: DataLoader, valloader: DataLoader, testloader: DataLoader):
        """Main training loop with efficiency tracking."""
        logger.info(f"Starting {self.cfg.mode} training for {self.cfg.epochs} epochs")
        logger.info(f"Mode: {self.cfg.mode}, Heap size: {self.cfg.heap_max_size if self.cfg.mode == 'heap' else 'N/A'}")
        
        # Time budget check
        if self.cfg.time_budget_minutes:
            time_limit = self.cfg.time_budget_minutes * 60
            logger.info(f"Time budget: {self.cfg.time_budget_minutes} minutes")
        else:
            time_limit = float('inf')
        
        # Training loop
        for epoch in range(self.cfg.epochs):
            # Check time budget
            if time.time() - self.metrics.start_time > time_limit:
                logger.info(f"Time budget exceeded at epoch {epoch}")
                break
            
            self.model.train()
            epoch_loss = 0
            epoch_steps = 0
            epoch_start = time.time()
            
            # Reset epoch tracking
            if self.cfg.mode == "heap":
                self.heap.reset_epoch_tracking()
            
            # Warmup with clean training
            if epoch < self.cfg.warmup_epochs:
                logger.info(f"Warmup epoch {epoch+1}/{self.cfg.warmup_epochs} - Clean training")
                for batch_idx, (x, y) in enumerate(trainloader):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    if self.cfg.use_normalization:
                        x_norm = Attacks.normalize(x)
                        logits = self.model(x_norm)
                    else:
                        logits = self.model(x)
                    
                    loss = F.cross_entropy(logits, y)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_steps += 1
                    self.metrics.total_forward_passes += x.size(0)
                
                self.scheduler.step()
                continue
            
            # Build initial bank after warmup (heap mode only)
            if epoch == self.cfg.warmup_epochs and self.cfg.mode == "heap" and self.heap.size() == 0:
                self.model.eval()
                self._build_initial_bank(trainloader)
                self.model.train()
            
            # Main training
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Get adversarial variants based on mode
                if self.cfg.mode == "heap":
                    variants, heap_count, fresh_count = self._get_heap_variants(x, y)
                    self.metrics.heap_vs_fresh_ratio.append((heap_count/len(variants), fresh_count/len(variants)))
                elif self.cfg.mode == "no_heap":
                    variants = self._get_standard_variants(x, y)
                elif self.cfg.mode == "random_buffer":
                    variants = self._get_random_buffer_variants(x, y)
                elif self.cfg.mode == "hard_mining":
                    variants = self._get_hard_mining_variants(x, y)
                elif self.cfg.mode == "fifo":
                    variants = self._get_fifo_variants(x, y)
                else:
                    raise ValueError(f"Unknown mode: {self.cfg.mode}")
                
                # Train step
                loss, batch_metrics = self._train_step_with_metrics(variants)
                epoch_loss += loss
                epoch_steps += 1
                self._loss_hist.append(loss)
                
                # Record batch quality
                self.metrics.batch_loss_mean.append(batch_metrics['loss'])
                self.metrics.batch_margin_mean.append(batch_metrics['margin'])
                
                # Detect plateau and low entropy
                plateau = self._detect_plateau()
                low_entropy = False
                if self.cfg.mode == "heap":
                    hm = self.heap.compute_metrics()
                    self._last_entropy = hm.get('bucket_entropy', 1.0)
                    low_entropy = self._last_entropy < 0.2
                self._last_plateau = plateau
                
                # CHI prompt if enabled
                if self.chi and self.chi.should_query(self.global_step, plateau=plateau, low_entropy=low_entropy):
                    # Use web, interactive, or simulated response based on mode
                    if self.cfg.human_in_loop == "web":
                        action = self.chi.web_human_response(self, plateau, low_entropy)
                    elif self.cfg.human_in_loop == "interactive":
                        action = self.chi.interactive_human_response(self, plateau, low_entropy)
                    else:
                        action = self.chi.simulate_human_response(self, plateau, low_entropy)
                    self.chi.apply_action(self, action)
                
                # Apply CHI decay
                if self.chi:
                    self.feature_weights = self.chi.decay_boosts(self.global_step, self.feature_weights)
                
                # Update variants (heap mode)
                if self.cfg.mode == "heap":
                    self._update_variants(variants)
                    
                    # Refit buckets periodically (FIX: use nlargest for better sampling)
                    if self.bucket_manager.should_refit(self.global_step):
                        heap_sample = [item[2] for item in heapq.nlargest(500, self.heap.heap)]
                        self.bucket_manager.fit(heap_sample, self.global_step)
                        logger.debug(f"Refit buckets at step {self.global_step}")
                
                # Log efficiency metrics
                if self.global_step % self.cfg.metrics_log_interval == 0:
                    self._log_efficiency_metrics()
                
                self.global_step += 1
                
                # Check time budget
                if time.time() - self.metrics.start_time > time_limit:
                    logger.info(f"Time budget exceeded at step {self.global_step}")
                    break
            
            # Epoch complete
            self.scheduler.step()
            
            # Validation
            if (epoch + 1) % self.cfg.eval_interval == 0:
                val_metrics = self._evaluate(valloader)
                
                # Record anytime performance
                self.metrics.log_snapshot(val_metrics['robust_acc'])
                
                self.history['val_clean_acc'].append(val_metrics['clean_acc'])
                self.history['val_robust_acc'].append(val_metrics['robust_acc'])
                
                logger.info(f"Epoch {epoch+1}/{self.cfg.epochs}:")
                logger.info(f"  Train Loss: {epoch_loss/max(epoch_steps,1):.4f}")
                logger.info(f"  Val Clean: {val_metrics['clean_acc']:.4f}")
                logger.info(f"  Val Robust: {val_metrics['robust_acc']:.4f}")
                logger.info(f"  Wall time: {time.time() - self.metrics.start_time:.1f}s")
                logger.info(f"  PGD calls: {self.metrics.pgd_calls_train}")
                
                # Log batch quality trend
                if self.metrics.batch_margin_mean:
                    recent_margin = np.mean(self.metrics.batch_margin_mean[-100:])
                    logger.info(f"  Recent batch margin: {recent_margin:.4f}")
                
                if self.cfg.mode == "heap":
                    heap_metrics = self.heap.compute_metrics()
                    logger.info(f"  Heap: size={self.heap.size()}, entropy={heap_metrics['bucket_entropy']:.3f}, "
                              f"turnover={heap_metrics['turnover_rate']:.3f}")
                
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
        
        # Final evaluation
        self._load_best_checkpoint()
        test_metrics = self._comprehensive_evaluate(testloader)
        self._save_results(test_metrics)
        
        return self.model, self.history, test_metrics
    
    def _build_initial_bank(self, trainloader: DataLoader):
        """Build initial bank efficiently."""
        logger.info(f"Building initial bank ({self.cfg.initial_bank_size} variants)")
        
        initial_variants = []
        
        for batch_idx, (x, y) in enumerate(trainloader):
            if len(initial_variants) >= self.cfg.initial_bank_size:
                break
            
            x, y = x.to(self.device), y.to(self.device)
            
            # Use FGSM for speed
            x_adv = Attacks.fgsm(self.model, x, y, self.cfg.epsilon, 
                               str(self.device), self.cfg.use_normalization, set_eval=True)
            self.metrics.fgsm_calls_train += x.size(0)
            
            features = self.feature_extractor.extract_features(x, x_adv, y)
            
            for i in range(min(x.size(0), self.cfg.initial_bank_size - len(initial_variants))):
                variant_features = {k: v[i] for k, v in features.items()}
                variant = Variant(x[i], x_adv[i], y[i], variant_features, 
                                idx=batch_idx * self.cfg.batch_size + i,
                                generation_step=0)
                initial_variants.append(variant)
                self.metrics.unique_variants_seen.add(variant.id)
        
        # Initialize buckets
        self.bucket_manager.fit(initial_variants)
        
        # Compute scores and add to heap
        for v in initial_variants:
            self.bucket_manager.assign_bucket(v)
            v.novelty = self._compute_novelty(v)
            v.score = self._compute_score(v)
            v.last_scored_step = 0
            self.heap.push(v)
        
        logger.info(f"Initial bank ready: {self.heap.size()} variants")
    
    def _detect_plateau(self) -> bool:
        """Detect training plateau."""
        if len(self._loss_hist) < self._loss_hist.maxlen:
            return False
        arr = np.array(self._loss_hist)
        return arr.std() < 1e-3
    
    def _get_heap_variants(self, x: Tensor, y: Tensor) -> Tuple[List[Variant], int, int]:
        """Get variants using heap strategy with configurable mix."""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        
        # Use heap_ratio to determine split (heap_ratio = proportion from heap)
        heap_k = int(batch_size * self.cfg.heap_ratio)
        heap_variants = self.heap.pop_diverse_batch(min(heap_k, self.heap.size()))
        
        fresh_k = batch_size - len(heap_variants)
        fresh_variants = self._generate_fresh_variants(x[:fresh_k], y[:fresh_k])
        
        # Apply diversity burst if active
        all_variants = heap_variants + fresh_variants
        if self._diversity_burst_steps > 0:
            self._diversity_burst_steps -= 1
            # Re-select for maximum diversity
            all_variants = self.heap._select_diverse(all_variants, min(len(all_variants), self.cfg.k_per_batch))
        
        return all_variants, len(heap_variants), len(fresh_variants)
    
    def _get_standard_variants(self, x: Tensor, y: Tensor) -> List[Variant]:
        """Standard adversarial training (no heap)."""
        return self._generate_fresh_variants(x, y)
    
    def _get_random_buffer_variants(self, x: Tensor, y: Tensor) -> List[Variant]:
        """Random buffer strategy."""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        
        # Sample from buffer
        buffer_k = min(batch_size // 2, len(self.buffer))
        if buffer_k > 0:
            buffer_variants = random.sample(list(self.buffer), buffer_k)
        else:
            buffer_variants = []
        
        # Generate fresh
        fresh_k = batch_size - len(buffer_variants)
        fresh_variants = self._generate_fresh_variants(x[:fresh_k], y[:fresh_k])
        
        # Add fresh to buffer
        for v in fresh_variants:
            self.buffer.append(v)
        
        return buffer_variants + fresh_variants
    
    def _get_hard_mining_variants(self, x: Tensor, y: Tensor) -> List[Variant]:
        """Hard mining strategy (buffer sorted by loss)."""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        
        # Get hardest from buffer
        if len(self.buffer) > 0:
            sorted_buffer = sorted(self.buffer, 
                                 key=lambda v: v.features.get('hardness', 0), 
                                 reverse=True)
            buffer_k = min(batch_size // 2, len(self.buffer))
            buffer_variants = sorted_buffer[:buffer_k]
        else:
            buffer_variants = []
        
        # Generate fresh
        fresh_k = batch_size - len(buffer_variants)
        fresh_variants = self._generate_fresh_variants(x[:fresh_k], y[:fresh_k])
        
        # Update buffer
        for v in fresh_variants:
            self.buffer.append(v)
        
        return buffer_variants + fresh_variants
    
    def _get_fifo_variants(self, x: Tensor, y: Tensor) -> List[Variant]:
        """FIFO queue strategy."""
        batch_size = min(x.size(0), self.cfg.k_per_batch)
        
        # Pop from queue
        queue_k = min(batch_size // 2, len(self.queue))
        queue_variants = []
        for _ in range(queue_k):
            if self.queue:
                queue_variants.append(self.queue.popleft())
        
        # Generate fresh
        fresh_k = batch_size - len(queue_variants)
        fresh_variants = self._generate_fresh_variants(x[:fresh_k], y[:fresh_k])
        
        # Add fresh to queue
        for v in fresh_variants:
            self.queue.append(v)
        
        return queue_variants + fresh_variants
    
    def _generate_fresh_variants(self, x: Tensor, y: Tensor) -> List[Variant]:
        """Generate fresh PGD variants (FIX: use set_eval=True for BN stability)."""
        x_adv = Attacks.pgd(
            self.model, x, y, self.cfg.epsilon, 
            self.cfg.pgd_step_size, self.cfg.pgd_steps, 
            str(self.device), normalized=self.cfg.use_normalization,
            set_eval=True  # FIX: Freeze BN during PGD for stability
        )
        self.metrics.pgd_calls_train += x.size(0)
        
        features = self.feature_extractor.extract_features(x, x_adv, y)
        
        variants = []
        for i in range(x.size(0)):
            variant_features = {k: v[i] for k, v in features.items()}
            variant = Variant(x[i], x_adv[i], y[i], variant_features,
                            generation_step=self.global_step)
            
            if self.cfg.mode == "heap":
                self.bucket_manager.assign_bucket(variant)
                variant.novelty = self._compute_novelty(variant)
                variant.score = self._compute_score(variant)
            
            variant.last_scored_step = self.global_step
            variants.append(variant)
            self.metrics.unique_variants_seen.add(variant.id)
        
        return variants
    
    def _train_step_with_metrics(self, variants: List[Variant]) -> Tuple[float, Dict]:
        """Training step that returns metrics."""
        x_adv_list = [v.reconstruct_adv(str(self.device)) for v in variants]
        x_adv = torch.stack(x_adv_list, dim=0)
        y = torch.tensor([v.y for v in variants], device=self.device, dtype=torch.long)
        
        if self.cfg.use_normalization:
            x_adv = Attacks.normalize(x_adv)
        
        # Forward pass and loss
        if self.scaler:
            with autocast('cuda'):
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.metrics.total_forward_passes += x_adv.size(0)
        
        # Compute batch metrics
        with torch.no_grad():
            true_logits = logits.gather(1, y.unsqueeze(1)).squeeze()
            other_logits = logits.clone()
            other_logits.scatter_(1, y.unsqueeze(1), float('-inf'))
            max_other = other_logits.max(dim=1)[0]
            margins = true_logits - max_other
            avg_margin = margins.mean().item()
        
        return loss.item(), {'loss': loss.item(), 'margin': avg_margin}
    
    def _update_variants(self, variants: List[Variant]):
        """Update variant features and push to heap."""
        with torch.no_grad():
            x_adv = torch.stack([v.reconstruct_adv(str(self.device)) for v in variants], dim=0)
            x_clean = torch.stack([v.x_clean.float().to(self.device) for v in variants], dim=0)
            y = torch.tensor([v.y for v in variants], device=self.device, dtype=torch.long)
            
            if self.cfg.use_normalization:
                x_adv_norm = Attacks.normalize(x_adv)
                x_clean_norm = Attacks.normalize(x_clean)
                logits = self.model(x_adv_norm)
                logits_clean = self.model(x_clean_norm)
            else:
                logits = self.model(x_adv)
                logits_clean = self.model(x_clean)
            
            true_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y.unsqueeze(1), float('-inf'))
            max_other = masked.max(1).values
            norm_margin = torch.sigmoid(true_logits - max_other)
            
            probs = F.softmax(logits, dim=1)
            probs_clean = F.softmax(logits_clean, dim=1)
            confidence = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            clean_conf = probs_clean.gather(1, y.unsqueeze(1)).squeeze(1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        for i, v in enumerate(variants):
            v.features['normalized_margin'] = float(norm_margin[i].item())
            v.features['hardness'] = float(1.0 - norm_margin[i].item())
            v.features['confidence'] = float(confidence[i].item())
            v.features['confidence_drop'] = float((clean_conf[i] - confidence[i]).item())
            v.features['entropy'] = float(entropy[i].item())
            
            v._feature_vector = None
            
            v.age = self.global_step - v.last_scored_step
            
            self.bucket_manager.assign_bucket(v)
            v.novelty = self._compute_novelty(v)
            v.score = self._compute_score(v)
            v.last_scored_step = self.global_step
            
            self.heap.push(v)
            self.recent_phi_cache.append(v.get_feature_vector())
    
    def _compute_score(self, variant: Variant) -> float:
        """Compute variant priority score with hardness focus."""
        hardness = variant.features.get('hardness', 0.5)
        
        feat_score = 0.0
        for feat_name, weight in self.feature_weights.items():
            if feat_name in variant.features:
                feat_score += weight * variant.features[feat_name]
        
        age = self.global_step - variant.last_scored_step
        age_penalty = min(age / 1000, 1.0)
        
        score = (
            self.cfg.weight_hardness * hardness +
            self.cfg.weight_bucket * variant.bucket_score +
            self.cfg.weight_novelty * variant.novelty +
            self.cfg.weight_features * feat_score -
            self.cfg.weight_age_penalty * age_penalty
        )
        
        return float(score)
    
    def _compute_novelty(self, variant: Variant) -> float:
        """Compute novelty against recent variants."""
        if not self.recent_phi_cache:
            return 1.0
        
        v_feat = variant.get_feature_vector()
        
        sample_size = min(50, len(self.recent_phi_cache))
        sample_indices = np.random.choice(len(self.recent_phi_cache), sample_size, replace=False)
        
        distances = []
        for idx in sample_indices:
            other_feat = self.recent_phi_cache[idx]
            dist = np.linalg.norm(v_feat - other_feat)
            distances.append(dist)
        
        avg_dist = np.mean(distances)
        novelty = 1 - np.exp(-avg_dist)
        
        return float(novelty)
    
    def _log_efficiency_metrics(self):
        """Log detailed efficiency metrics."""
        if self.cfg.mode == "heap":
            heap_metrics = self.heap.compute_metrics()
            self.metrics.heap_turnover_rate.append(heap_metrics['turnover_rate'])
            self.metrics.heap_age_mean.append(heap_metrics['age_mean'])
            self.metrics.heap_age_median.append(heap_metrics['age_median'])
            self.metrics.bucket_entropy.append(heap_metrics['bucket_entropy'])
            self.metrics.novelty_mean.append(heap_metrics['novelty_mean'])
            
            # Per-class coverage
            class_dist = self.heap.get_class_distribution()
            self.metrics.per_class_coverage.append(class_dist)
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Standard evaluation."""
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            
            with torch.no_grad():
                if self.cfg.use_normalization:
                    x_norm = Attacks.normalize(x)
                    logits = self.model(x_norm)
                else:
                    logits = self.model(x)
                clean_correct += (logits.argmax(1) == y).sum().item()
            
            x_adv = Attacks.pgd(
                self.model, x, y,
                self.cfg.epsilon, self.cfg.pgd_step_size,
                self.cfg.eval_pgd_steps, str(self.device),
                normalized=self.cfg.use_normalization,
                set_eval=True
            )
            self.metrics.pgd_calls_eval += x.size(0)
            
            with torch.no_grad():
                if self.cfg.use_normalization:
                    x_adv_norm = Attacks.normalize(x_adv)
                    logits_adv = self.model(x_adv_norm)
                else:
                    logits_adv = self.model(x_adv)
                robust_correct += (logits_adv.argmax(1) == y).sum().item()
            
            total += y.size(0)
        
        return {
            'clean_acc': clean_correct / total,
            'robust_acc': robust_correct / total
        }
    
    def _comprehensive_evaluate(self, testloader: DataLoader) -> Dict[str, float]:
        """Comprehensive evaluation."""
        logger.info("Running comprehensive evaluation on test set...")
        
        self.model.eval()
        results = {}
        
        # Clean accuracy
        clean_correct = 0
        total = 0
        
        for x, y in testloader:
            x, y = x.to(self.device), y.to(self.device)
            
            with torch.no_grad():
                if self.cfg.use_normalization:
                    x_norm = Attacks.normalize(x)
                    logits = self.model(x_norm)
                else:
                    logits = self.model(x)
                clean_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        
        results['clean_acc'] = clean_correct / total
        
        # Multi-restart PGD
        pgd_correct = 0
        total = 0
        
        for x, y in testloader:
            x, y = x.to(self.device), y.to(self.device)
            
            batch_correct = torch.ones(x.size(0), dtype=torch.bool, device=self.device)
            
            for restart in range(self.cfg.eval_pgd_restarts):
                x_adv = Attacks.pgd(
                    self.model, x, y,
                    self.cfg.epsilon, self.cfg.pgd_step_size,
                    self.cfg.eval_pgd_steps, str(self.device),
                    random_start=True,
                    normalized=self.cfg.use_normalization,
                    set_eval=True
                )
                
                with torch.no_grad():
                    if self.cfg.use_normalization:
                        x_adv_norm = Attacks.normalize(x_adv)
                        logits = self.model(x_adv_norm)
                    else:
                        logits = self.model(x_adv)
                    batch_correct &= (logits.argmax(1) == y)
            
            pgd_correct += batch_correct.sum().item()
            total += y.size(0)
        
        results['robust_acc'] = pgd_correct / total
        
        logger.info(f"Test Results:")
        logger.info(f"  Clean:  {results['clean_acc']:.4f}")
        logger.info(f"  Robust: {results['robust_acc']:.4f}")
        
        return results
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_robust': self.best_val_robust,
            'config': asdict(self.cfg)
        }
        
        if is_best:
            path = os.path.join(self.cfg.experiment_dir, 'best_model.pth')
        else:
            path = os.path.join(self.cfg.experiment_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        path = os.path.join(self.cfg.experiment_dir, 'best_model.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    def _save_results(self, test_metrics: Dict):
        """Save comprehensive results with efficiency metrics."""
        # Compute efficiency score (consistent definition)
        pgd = max(self.metrics.pgd_calls_train, 1)
        efficiency_score = test_metrics['robust_acc'] * (10000 / pgd)
        
        results = {
            'config': asdict(self.cfg),
            'test_metrics': test_metrics,
            'history': dict(self.history),
            'best_val_robust': self.best_val_robust,
            'efficiency_metrics': self.metrics.to_dict(),
            'efficiency_score': efficiency_score,  # Robust acc per 10k PGD calls
            'total_wall_time': time.time() - self.metrics.start_time,
            'chi_history': self.chi.query_history if self.chi else []
        }
        
        json_path = os.path.join(self.cfg.experiment_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.cfg.experiment_dir}")

# ========================= Data Loading =========================
def get_cifar10_loaders(cfg: Config):
    """Get CIFAR-10 loaders with proper train/val/test splits."""
    
    # Transforms WITHOUT normalization (keep data in [0,1])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load full datasets
    trainset_full = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=False, download=True, transform=transform_test
    )
    
    # Create train/val split
    train_size = int((1 - cfg.val_split) * len(trainset_full))
    val_size = len(trainset_full) - train_size
    
    # Use deterministic split
    generator = torch.Generator().manual_seed(cfg.seed)
    trainset, valset = random_split(trainset_full, [train_size, val_size], generator=generator)
    
    # Apply subset if specified
    if cfg.subset_size:
        train_subset_size = min(int(cfg.subset_size * (1 - cfg.val_split)), len(trainset))
        val_subset_size = min(int(cfg.subset_size * cfg.val_split), len(valset))
        test_subset_size = min(max(cfg.subset_size // 5, 100), len(testset))
        
        trainset = Subset(trainset, range(train_subset_size))
        valset = Subset(valset, range(val_subset_size))
        testset = Subset(testset, range(test_subset_size))
    
    # Create loaders
    persistent = cfg.num_workers > 0
    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=persistent
    )
    valloader = DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=persistent
    )
    testloader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=persistent
    )
    
    return trainloader, valloader, testloader

# ========================= Efficiency Comparison =========================
def run_efficiency_comparison(cfg: Config, modes: List[str], time_budget: float = None):
    """Run efficiency comparison across different modes."""
    results = {}
    
    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {mode.upper()} mode")
        logger.info(f"{'='*60}")
        
        cfg_mode = copy.deepcopy(cfg)
        cfg_mode.mode = mode
        cfg_mode.time_budget_minutes = time_budget
        
        trainloader, valloader, testloader = get_cifar10_loaders(cfg_mode)
        trainer = HeapRobustTrainer(cfg_mode)
        model, history, metrics = trainer.train(trainloader, valloader, testloader)
        
        results[mode] = {
            'test_metrics': metrics,
            'efficiency_metrics': trainer.metrics.to_dict(),
            'best_val_robust': trainer.best_val_robust,
            'total_time': time.time() - trainer.metrics.start_time,
            'history': history
        }
        
        # Log key efficiency metrics
        logger.info(f"\n{mode.upper()} Summary:")
        logger.info(f"  Test Robust: {metrics['robust_acc']:.4f}")
        logger.info(f"  Total Time: {results[mode]['total_time']:.1f}s")
        logger.info(f"  PGD Calls (train): {trainer.metrics.pgd_calls_train}")
        logger.info(f"  Unique Variants: {len(trainer.metrics.unique_variants_seen)}")
    
    return results

def plot_efficiency_results(results: Dict, output_dir: str):
    """Generate efficiency comparison plots."""
    try:
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("Seaborn not available, using matplotlib defaults")
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Robust Acc vs Time
        ax = axes[0, 0]
        for mode, data in results.items():
            timeline = data['efficiency_metrics']['robust_acc_timeline']
            if timeline:
                times, accs = zip(*timeline)
                ax.plot(times, accs, label=mode, marker='o')
        ax.set_xlabel('Wall Clock Time (s)')
        ax.set_ylabel('Robust Accuracy')
        ax.set_title('Anytime Performance: Robust Acc vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Robust Acc vs PGD Calls
        ax = axes[0, 1]
        for mode, data in results.items():
            pgd_timeline = data['efficiency_metrics']['robust_acc_vs_pgd']
            if pgd_timeline:
                calls, accs = zip(*pgd_timeline)
                ax.plot(calls, accs, label=mode, marker='s')
        ax.set_xlabel('PGD Calls (Training)')
        ax.set_ylabel('Robust Accuracy')
        ax.set_title('Compute Efficiency: Robust Acc vs PGD Calls')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Heap Dynamics (if heap mode exists)
        if 'heap' in results:
            ax = axes[1, 0]
            heap_data = results['heap']['efficiency_metrics']
            
            # Bucket entropy over time
            if heap_data['bucket_entropy']:
                ax.plot(heap_data['bucket_entropy'], label='Bucket Entropy', color='blue')
            ax2 = ax.twinx()
            if heap_data['heap_turnover_rate']:
                ax2.plot(heap_data['heap_turnover_rate'], label='Turnover Rate', 
                        color='orange', linestyle='--')
            
            ax.set_xlabel('Logging Steps')
            ax.set_ylabel('Bucket Entropy', color='blue')
            ax2.set_ylabel('Turnover Rate', color='orange')
            ax.set_title('Heap Health Metrics')
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Final Performance Comparison
        ax = axes[1, 1]
        modes_list = list(results.keys())
        test_robust = [results[m]['test_metrics']['robust_acc'] for m in modes_list]
        pgd_calls = [results[m]['efficiency_metrics']['pgd_calls_train'] for m in modes_list]
        
        x = np.arange(len(modes_list))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, test_robust, width, label='Test Robust Acc', color='steelblue')
        bars2 = ax2.bar(x + width/2, pgd_calls, width, label='PGD Calls', color='coral')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Test Robust Accuracy', color='steelblue')
        ax2.set_ylabel('PGD Calls (Training)', color='coral')
        ax.set_title('Final Performance vs Compute Cost')
        ax.set_xticks(x)
        ax.set_xticklabels(modes_list)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Add value labels on bars
        for bar, val in zip(bars1, test_robust):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'efficiency_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_path}")
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")

def export_heap_bank(trainer: HeapRobustTrainer, output_path: str, format: str = "json"):
    """Export the heap bank for reuse (JSON or PyTorch format)."""
    if not hasattr(trainer, 'heap'):
        logger.warning("No heap to export")
        return
    
    if format == "pt":
        # PyTorch format (faster for reloading)
        bank_data = {
            'variants': [],
            'bucket_scores': trainer.bucket_manager.bucket_scores if hasattr(trainer, 'bucket_manager') else {},
            'config': trainer.cfg
        }
        
        for _, _, variant in heapq.nlargest(1000, trainer.heap.heap):
            bank_data['variants'].append({
                'x_clean': variant.x_clean,
                'delta': variant.delta,
                'y': variant.y,
                'features': variant.features,
                'bucket_id': variant.bucket_id,
                'score': variant.score
            })
        
        torch.save(bank_data, output_path)
        logger.info(f"Exported {len(bank_data['variants'])} variants to {output_path} (PyTorch format)")
    
    else:  # JSON format
        bank_data = {
            'variants': [],
            'bucket_scores': trainer.bucket_manager.bucket_scores if hasattr(trainer, 'bucket_manager') else {},
            'config': asdict(trainer.cfg)
        }
        
        for _, _, variant in heapq.nlargest(1000, trainer.heap.heap):
            bank_data['variants'].append({
                'x_clean': variant.x_clean.numpy().tolist(),
                'delta': variant.delta.numpy().tolist(),
                'y': variant.y,
                'features': variant.features,
                'bucket_id': variant.bucket_id,
                'score': variant.score
            })
        
        with open(output_path, 'w') as f:
            json.dump(bank_data, f)
        
        logger.info(f"Exported {len(bank_data['variants'])} variants to {output_path} (JSON format)")

# ========================= Main Execution =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heap-Guided Adversarial Training - Efficiency Focus")
    parser.add_argument("--experiment", default="efficiency",
                        choices=["efficiency", "single", "ablation", "time_budget"])
    parser.add_argument("--mode", default="heap",
                        choices=["heap", "no_heap", "random_buffer", "hard_mining", "fifo"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--time_budget", type=float, default=None,
                        help="Time budget in minutes")
    parser.add_argument("--heap_size", type=int, default=2000)
    parser.add_argument("--bucket_strategy", default="class_hardness",
                        choices=["class", "class_hardness", "kmeans", "tree"])
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--chi", action="store_true", help="Enable HITL prompts (simulation mode)")
    parser.add_argument("--chi_interactive", action="store_true", help="Enable interactive HITL (asks for input)")
    parser.add_argument("--chi_web", action="store_true", help="Enable web-based HITL (http://localhost:5000)")
    parser.add_argument("--chi_web_port", type=int, default=5000, help="Port for CHI web server")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Clean training warmup epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--export_bank", action="store_true")
    parser.add_argument("--export_format", default="json", choices=["json", "pt"],
                        help="Export format for heap bank")
    parser.add_argument("--heap_ratio", type=float, default=0.5)
    parser.add_argument("--diversity_k_factor", type=int, default=3)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--eval_pgd_steps", type=int, default=20)
    parser.add_argument("--eval_pgd_restarts", type=int, default=2)
    parser.add_argument("--experiment_name", default="heap_efficient")

    args = parser.parse_args()

    # Build configuration (fixes: apply --lr, use --warmup_epochs, remove duplicate metrics_log_interval)
    cfg = Config(
        experiment_name=args.experiment_name,
        mode=args.mode,                              # <-- important fix
        epochs=args.epochs,
        subset_size=args.subset if args.subset > 0 else None,
        heap_max_size=args.heap_size,
        initial_bank_size=min(1000, args.heap_size // 2),
        bucket_strategy=args.bucket_strategy,
        use_normalization=not args.no_normalize,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        eval_interval=2,
        metrics_log_interval=20,
        early_stopping_patience=10,
        time_budget_minutes=args.time_budget,
        heap_ratio=args.heap_ratio,                  # new
        diversity_k_factor=args.diversity_k_factor,  # new
        pgd_steps=args.pgd_steps,                    # new
        eval_pgd_steps=args.eval_pgd_steps,          # new
        eval_pgd_restarts=args.eval_pgd_restarts,    # new
        human_in_loop=("web" if args.chi_web else ("interactive" if args.chi_interactive else (True if args.chi else False))),
        chi_web_port=args.chi_web_port
    )


    print(f"\nConfiguration:")
    print(f"  Device: {cfg.device}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Subset: {cfg.subset_size if cfg.subset_size else 'Full dataset'}")
    print(f"  Heap size: {cfg.heap_max_size}")
    print(f"  Bucket strategy: {cfg.bucket_strategy}")
    print(f"  Normalization: {cfg.use_normalization}")
    print(f"  Warmup epochs: {cfg.warmup_epochs}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  LR milestones: epochs {cfg.lr_milestones}")
    print(f"  CHI/HITL: {cfg.human_in_loop}")
    if cfg.time_budget_minutes:
        print(f"  Time budget: {cfg.time_budget_minutes} minutes")

    if args.experiment == "efficiency":
        # Run full efficiency comparison
        modes = ["heap", "no_heap", "random_buffer", "hard_mining"]
        results = run_efficiency_comparison(cfg, modes, args.time_budget)

        # Generate plots
        plot_efficiency_results(results, cfg.output_dir)

        # Save comparison results
        comparison_path = os.path.join(
            cfg.output_dir, f"efficiency_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(comparison_path, 'w') as f:
            # Convert numpy values for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, set):
                    return list(obj)
                return obj

            results_serializable = {
                mode: {k: convert_numpy(v) for k, v in data.items()}
                for mode, data in results.items()
            }
            json.dump(results_serializable, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("EFFICIENCY COMPARISON SUMMARY")
        print(f"{'='*60}")
        for mode in modes:
            r = results[mode]
            print(f"\n{mode.upper()}:")
            print(f"  Test Robust Acc: {r['test_metrics']['robust_acc']:.4f}")
            print(f"  Total Time: {r['total_time']:.1f}s")
            print(f"  PGD Calls: {r['efficiency_metrics']['pgd_calls_train']}")
            print(f"  Unique Variants: {r['efficiency_metrics']['total_unique_variants']}")
            pgd = max(r['efficiency_metrics']['pgd_calls_train'], 1)
            efficiency = r['test_metrics']['robust_acc'] * (10000 / pgd)
            print(f"  Efficiency (RobAcc per 10k PGD): {efficiency:.4f}")
            results[mode]['efficiency_score'] = efficiency  # store it, too

        print(f"\nResults saved to {comparison_path}")

    elif args.experiment == "time_budget":
        # Anytime performance experiment
        time_budgets = [5, 10, 20, 30]  # minutes
        modes = ["heap", "no_heap"]

        budget_results = {}
        for budget in time_budgets:
            print(f"\n{'='*60}")
            print(f"TIME BUDGET: {budget} minutes")
            print(f"{'='*60}")
            budget_results[budget] = run_efficiency_comparison(cfg, modes, budget)

        # Summarize
        print(f"\n{'='*60}")
        print("TIME BUDGET ANALYSIS")
        print(f"{'='*60}")
        for budget in time_budgets:
            print(f"\n{budget} minutes:")
            for mode in modes:
                acc = budget_results[budget][mode]['test_metrics']['robust_acc']
                print(f"  {mode}: {acc:.4f}")

    else:
        # "single" (or "ablation") – just run one mode
        cfg.mode = args.mode
        trainloader, valloader, testloader = get_cifar10_loaders(cfg)

        trainer = HeapRobustTrainer(cfg)
        model, history, metrics = trainer.train(trainloader, valloader, testloader)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ({cfg.mode.upper()})")
        print(f"{'='*60}")
        print(f"Test Clean Accuracy:  {metrics['clean_acc']:.4f}")
        print(f"Test Robust Accuracy: {metrics['robust_acc']:.4f}")
        print(f"Best Val Robust:      {trainer.best_val_robust:.4f}")
        print(f"Total Time:           {time.time() - trainer.metrics.start_time:.1f}s")
        print(f"PGD Calls (train):    {trainer.metrics.pgd_calls_train}")
        print(f"Unique Variants Seen: {len(trainer.metrics.unique_variants_seen)}")

        pgd = max(trainer.metrics.pgd_calls_train, 1)
        efficiency = metrics['robust_acc'] * (10000 / pgd)
        print(f"Efficiency Score:     {efficiency:.4f}")

        if trainer.chi:
            print(f"CHI Queries Used:     {len(trainer.chi.query_history)}")

        if args.export_bank and cfg.mode == "heap":
            ext = "pt" if args.export_format == "pt" else "json"
            bank_path = os.path.join(cfg.experiment_dir, f"heap_bank.{ext}")
            export_heap_bank(trainer, bank_path, format=args.export_format)
            print(f"Heap bank exported to {bank_path}")
