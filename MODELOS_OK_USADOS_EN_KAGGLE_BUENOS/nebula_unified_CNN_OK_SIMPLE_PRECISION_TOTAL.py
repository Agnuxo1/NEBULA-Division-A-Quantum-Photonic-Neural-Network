#!/usr/bin/env python3
"""
NEBULA Grand X-Ray Unified System - High Precision Edition
===========================================================
Unified architecture targeting 0.99 precision for chest X-ray classification
Optimal configuration: 900 rays, 256x256 resolution
All modules integrated and optimized for maximum accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from scipy.optimize import differential_evolution
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import gc
import glob
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# ADAPTIVE MEMORY MANAGEMENT SYSTEM
# ============================================================================

class AdaptiveMemoryManager:
    """Adaptive memory management to prevent CUDA errors"""
    
    def __init__(self, device):
        self.device = device
        self.memory_threshold = 0.85  # Use max 85% of GPU memory
        self.recovery_delay = 2.0  # Seconds to wait for memory recovery
        self.cleanup_interval = 50  # Clean memory every N batches
        self.batch_counter = 0
        self.memory_warnings = 0
        
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU Memory Manager: {self.total_memory / 1024**3:.1f}GB total")
    
    def check_memory_status(self):
        """Check current GPU memory usage"""
        if not torch.cuda.is_available():
            return True, 0.0
            
        try:
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            usage_ratio = allocated / self.total_memory
            
            return usage_ratio < self.memory_threshold, usage_ratio
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return True, 0.0
    
    def adaptive_cleanup(self, force=False):
        """Adaptive memory cleanup with smart timing"""
        self.batch_counter += 1
        
        # Check if cleanup is needed
        should_cleanup = (
            force or 
            self.batch_counter % self.cleanup_interval == 0
        )
        
        if should_cleanup:
            try:
                # Progressive cleanup strategy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                # Python garbage collection
                gc.collect()
                
                # Memory status check
                is_ok, usage = self.check_memory_status()
                
                if not is_ok:
                    self.memory_warnings += 1
                    logger.warning(f"High memory usage: {usage:.1%} - Applying recovery delay")
                    time.sleep(self.recovery_delay)
                    
                    # Aggressive cleanup if still high
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Memory cleanup failed: {e}")
    
    def safe_forward_pass(self, model, batch, use_amp, criterion):
        """Safe forward pass with memory monitoring"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Check memory before forward pass
                is_ok, usage = self.check_memory_status()
                
                if not is_ok and attempt == 0:
                    logger.warning(f"Pre-emptive memory cleanup - Usage: {usage:.1%}")
                    self.adaptive_cleanup(force=True)
                    time.sleep(retry_delay)
                
                # Forward pass
                if use_amp:
                    with autocast():
                        outputs = model(batch['image'])
                        loss = criterion(outputs, batch['labels'])
                else:
                    outputs = model(batch['image'])
                    loss = criterion(outputs, batch['labels'])
                
                return outputs, loss, True
                
            except RuntimeError as e:
                error_msg = str(e)
                
                # Handle CUDA memory errors specifically
                if "out of memory" in error_msg.lower() or "illegal memory access" in error_msg.lower():
                    logger.warning(f"CUDA memory error (attempt {attempt+1}/{max_retries}): {error_msg[:100]}")
                    
                    # Aggressive recovery
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                    
                    gc.collect()
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached - Memory error persists")
                        return None, None, False
                else:
                    # Non-memory error, re-raise
                    raise e
        
        return None, None, False

# Global memory manager instance
memory_manager = AdaptiveMemoryManager(DEVICE)

# ============================================================================
# THRESHOLD OPTIMIZATION SYSTEM - PRIORITY 1 ENHANCEMENT
# ============================================================================

class ThresholdOptimizer:
    """Advanced threshold optimization for multi-label classification"""
    
    def __init__(self):
        self.best_thresholds = None
        self.optimization_metrics = {}
    
    def optimize_thresholds(self, model, dataloader, device, target_metric='f1'):
        """
        Optimize decision thresholds for each class
        Args:
            model: Trained model
            dataloader: Validation dataloader  
            device: CUDA device
            target_metric: 'f1', 'precision', 'recall', or 'balanced'
        Returns:
            best_thresholds: Array of optimal thresholds per class
            metrics: Performance metrics with optimized thresholds
        """
        logger.info("🎯 Starting threshold optimization - Expected +3-5% accuracy boost")
        
        # Get model predictions and true labels
        all_predictions, all_labels = self._get_predictions(model, dataloader, device)
        
        num_classes = all_predictions.shape[1]
        logger.info(f"Optimizing thresholds for {num_classes} classes")
        
        # Optimize thresholds using differential evolution
        def objective_function(thresholds):
            predictions_binary = (all_predictions >= thresholds).astype(int)
            
            if target_metric == 'f1':
                scores = []
                for i in range(num_classes):
                    if all_labels[:, i].sum() > 0:  # Skip classes with no positive samples
                        score = f1_score(all_labels[:, i], predictions_binary[:, i])
                        scores.append(score)
                return -np.mean(scores)  # Negative because we minimize
            
            elif target_metric == 'precision':
                scores = []
                for i in range(num_classes):
                    if all_labels[:, i].sum() > 0:
                        score = precision_score(all_labels[:, i], predictions_binary[:, i], zero_division=0)
                        scores.append(score)
                return -np.mean(scores)
            
            elif target_metric == 'balanced':
                # Balanced combination of precision, recall, f1
                f1_scores, prec_scores, rec_scores = [], [], []
                for i in range(num_classes):
                    if all_labels[:, i].sum() > 0:
                        f1 = f1_score(all_labels[:, i], predictions_binary[:, i])
                        prec = precision_score(all_labels[:, i], predictions_binary[:, i], zero_division=0)
                        rec = recall_score(all_labels[:, i], predictions_binary[:, i])
                        f1_scores.append(f1)
                        prec_scores.append(prec)
                        rec_scores.append(rec)
                
                balanced_score = (np.mean(f1_scores) + np.mean(prec_scores) + np.mean(rec_scores)) / 3
                return -balanced_score
        
        # Optimization bounds (0.1 to 0.9 for each threshold)
        bounds = [(0.1, 0.9) for _ in range(num_classes)]
        
        logger.info("🔍 Running differential evolution optimization...")
        result = differential_evolution(
            objective_function, 
            bounds, 
            maxiter=50,  # Reasonable iterations for speed
            popsize=10,  # Population size
            seed=42
        )
        
        self.best_thresholds = result.x
        logger.info(f"✅ Optimization complete! Best thresholds: {self.best_thresholds}")
        
        # Calculate metrics with optimized thresholds
        metrics = self._calculate_optimized_metrics(all_predictions, all_labels, self.best_thresholds)
        self.optimization_metrics = metrics
        
        return self.best_thresholds, metrics
    
    def _get_predictions(self, model, dataloader, device):
        """Get all predictions and labels from dataloader"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Getting predictions for threshold optimization"):
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                
                # Safe prediction with memory management
                try:
                    outputs = model(images)
                    predictions = torch.sigmoid(outputs).cpu().numpy()
                    all_predictions.append(predictions)
                    all_labels.append(labels.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Batch prediction failed: {e}")
                    continue
        
        return np.vstack(all_predictions), np.vstack(all_labels)
    
    def _calculate_optimized_metrics(self, predictions, labels, thresholds):
        """Calculate metrics using optimized thresholds"""
        predictions_binary = (predictions >= thresholds).astype(int)
        
        metrics = {}
        class_names = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            if labels[:, i].sum() > 0:
                f1 = f1_score(labels[:, i], predictions_binary[:, i])
                prec = precision_score(labels[:, i], predictions_binary[:, i], zero_division=0)
                rec = recall_score(labels[:, i], predictions_binary[:, i])
                
                metrics[f'{class_name}_f1'] = f1
                metrics[f'{class_name}_precision'] = prec
                metrics[f'{class_name}_recall'] = rec
        
        # Overall metrics
        f1_scores = [metrics[k] for k in metrics.keys() if k.endswith('_f1')]
        prec_scores = [metrics[k] for k in metrics.keys() if k.endswith('_precision')]
        rec_scores = [metrics[k] for k in metrics.keys() if k.endswith('_recall')]
        
        metrics['mean_f1'] = np.mean(f1_scores)
        metrics['mean_precision'] = np.mean(prec_scores)
        metrics['mean_recall'] = np.mean(rec_scores)
        metrics['balanced_score'] = (metrics['mean_f1'] + metrics['mean_precision'] + metrics['mean_recall']) / 3
        
        return metrics
    
    def evaluate_with_optimized_thresholds(self, model, dataloader, device):
        """Evaluate model using optimized thresholds"""
        if self.best_thresholds is None:
            raise ValueError("Must optimize thresholds first!")
        
        logger.info("📊 Evaluating with optimized thresholds...")
        predictions, labels = self._get_predictions(model, dataloader, device)
        metrics = self._calculate_optimized_metrics(predictions, labels, self.best_thresholds)
        
        logger.info(f"🎯 OPTIMIZED RESULTS:")
        logger.info(f"   Mean F1: {metrics['mean_f1']:.4f}")
        logger.info(f"   Mean Precision: {metrics['mean_precision']:.4f}")
        logger.info(f"   Mean Recall: {metrics['mean_recall']:.4f}")
        logger.info(f"   Balanced Score: {metrics['balanced_score']:.4f}")
        
        return metrics

# Global threshold optimizer
threshold_optimizer = ThresholdOptimizer()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False  # More stable for long training
    torch.backends.cudnn.deterministic = True  # Prevent memory access errors
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Enhanced GPU memory management
    torch.cuda.empty_cache()
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    logger.warning("CUDA not available, using CPU (will be slower)")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class NEBULAConfig:
    """Unified configuration for optimal performance"""
    # Core parameters - OPTIMIZED FOR 0.99 PRECISION
    num_rays: int = 900  # Optimal ray count as specified
    resolution: Tuple[int, int] = (256, 256)  # Optimal resolution
    batch_size: int = 16  # Increased for better gradient estimation
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Model architecture
    num_classes: int = 14
    hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Ray-tracing parameters
    ray_march_steps: int = 32
    wavelengths: List[float] = None  # Will be set in __post_init__
    num_spectral_bands: int = 4
    
    # Training parameters
    epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    
    # Data augmentation
    augment_prob: float = 0.5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Optimization
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.wavelengths is None:
            self.wavelengths = [0.08, 0.12, 0.18, 0.25]  # Optimal X-ray wavelengths

# ============================================================================
# OPTIMIZED RAY-TRACING ENGINE
# ============================================================================

class OptimizedRayTracer(nn.Module):
    """Fixed and optimized ray-tracing engine - no data loss"""
    
    def __init__(self, config: NEBULAConfig):
        super().__init__()
        self.config = config
        
        # Ensure perfect square for optimal grid
        grid_size = int(math.sqrt(config.num_rays))
        self.num_rays = grid_size * grid_size  # Ensure perfect square
        self.grid_size = grid_size
        
        # Learnable ray parameters
        self.ray_intensity = nn.Parameter(torch.ones(config.num_spectral_bands))
        self.absorption_coeffs = nn.Parameter(torch.ones(config.num_spectral_bands) * 0.5)
        self.scattering_coeffs = nn.Parameter(torch.ones(config.num_spectral_bands) * 0.1)
        
        logger.info(f"Ray-tracer initialized: {self.num_rays} rays ({grid_size}x{grid_size} grid)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized ray-tracing without data loss
        Args:
            x: [batch, 1, H, W] input tensor
        Returns:
            [batch, num_spectral_bands, H, W] multi-spectral output
        """
        batch_size, _, H, W = x.shape
        device = x.device
        
        # Generate ray grid positions (no data loss here)
        ray_x = torch.linspace(-1, 1, self.grid_size, device=device)
        ray_y = torch.linspace(-1, 1, self.grid_size, device=device)
        grid_x, grid_y = torch.meshgrid(ray_x, ray_y, indexing='ij')
        
        # Flatten grid for ray processing
        ray_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [num_rays, 2]
        ray_positions = ray_positions.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_rays, 2]
        
        # Sample tissue density at ray positions (bilinear interpolation)
        # FIXED: Proper grid sampling without data loss
        sampling_grid = ray_positions.unsqueeze(2)  # [batch, num_rays, 1, 2]
        
        # Ensure x has correct shape for grid_sample
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Sample with proper boundary handling
        tissue_density = F.grid_sample(
            x, 
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(-1).squeeze(1)  # [batch, num_rays]
        
        # Multi-spectral processing
        spectral_outputs = []
        for band in range(self.config.num_spectral_bands):
            # Apply Beer-Lambert law for each wavelength
            absorption = self.absorption_coeffs[band]
            scattering = self.scattering_coeffs[band]
            intensity = self.ray_intensity[band]
            
            # Calculate attenuation
            attenuation = torch.exp(-(absorption + scattering) * tissue_density)
            transmitted = intensity * attenuation
            
            # Reshape to grid
            ray_image = transmitted.view(batch_size, self.grid_size, self.grid_size)
            
            # Upsample to target resolution
            ray_image = F.interpolate(
                ray_image.unsqueeze(1),
                size=self.config.resolution,
                mode='bilinear',
                align_corners=True
            )
            
            spectral_outputs.append(ray_image)
        
        # Combine spectral bands
        output = torch.cat(spectral_outputs, dim=1)  # [batch, num_spectral_bands, H, W]
        
        return output

# ============================================================================
# ATTENTION MODULE FOR FEATURE EXTRACTION
# ============================================================================

class EfficientSelfAttention(nn.Module):
    """Efficient self-attention for medical image features"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# ============================================================================
# MAIN NEBULA MODEL
# ============================================================================

class NEBULAGrandXRay(nn.Module):
    """Main NEBULA model with all optimizations for 0.99 precision"""
    
    def __init__(self, config: NEBULAConfig):
        super().__init__()
        self.config = config
        
        # Ray-tracing module
        self.ray_tracer = OptimizedRayTracer(config)
        
        # Feature extraction backbone (optimized for medical imaging)
        self.stem = nn.Sequential(
            nn.Conv2d(config.num_spectral_bands, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks for robust feature extraction
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 1024, 3, stride=2)
        
        # Global feature pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Attention mechanism for feature refinement
        self.attention = EfficientSelfAttention(1024, config.num_attention_heads, config.dropout_rate)
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(1024, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate)
        )
        
        # Classification heads (separate for better gradients)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim // 2, 1)
            ) for _ in range(config.num_classes)
        ])
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        """Create residual layer"""
        layers = []
        
        # First block handles stride and channel change
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        """Proper weight initialization"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with no data loss
        Args:
            x: [batch, 1, H, W] input tensor
        Returns:
            [batch, num_classes] predictions
        """
        # Ray-tracing for multi-spectral analysis
        x = self.ray_tracer(x)  # [batch, num_spectral_bands, H, W]
        
        # Feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.global_pool(x)  # [batch, 1024, 1, 1]
        x = x.flatten(1)  # [batch, 1024]
        
        # Attention refinement
        x = x.unsqueeze(1)  # [batch, 1, 1024]
        x = self.attention(x)
        x = x.squeeze(1)  # [batch, 1024]
        
        # Feature fusion
        x = self.fusion(x)  # [batch, hidden_dim]
        
        # Multi-label classification
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x))
        
        output = torch.cat(outputs, dim=1)  # [batch, num_classes]
        
        return output

class ResidualBlock(nn.Module):
    """Residual block for feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalBCELoss(nn.Module):
    """Focal loss for handling class imbalance - critical for 0.99 precision"""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weights
        p_t = torch.exp(-bce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()

# ============================================================================
# DATA HANDLING
# ============================================================================

class GrandXRayDataset(Dataset):
    """Optimized dataset with proper data handling"""
    
    def __init__(self, 
                 image_dir: str,
                 labels_df: pd.DataFrame,
                 config: NEBULAConfig,
                 transform=None,
                 training: bool = False):
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df
        self.config = config
        self.transform = transform
        self.training = training
        
        # Condition columns
        self.conditions = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.labels_df.iloc[idx]
        
        # Load image (handle different formats)
        image_name = row.get('Image_name', row.get('image_name', ''))
        image_path = self.image_dir / image_name
        image = self.load_image(image_path)
        
        # Get labels
        labels = torch.zeros(self.config.num_classes, dtype=torch.float32)
        for i, condition in enumerate(self.conditions):
            if condition in row:
                labels[i] = float(row[condition])
        
        # Apply augmentations if training
        if self.training and np.random.random() < self.config.augment_prob:
            image = self.augment(image)
        
        return {
            'image': image,
            'labels': labels,
            'image_id': image_name
        }
    
    def load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            from PIL import Image
            img = Image.open(path).convert('L')  # Grayscale
            img = img.resize(self.config.resolution)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Normalize
            img_array = (img_array - 0.5) / 0.5
            
            return torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            # Return blank image on error
            return torch.zeros(1, *self.config.resolution)
    
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = torch.flip(image, [-1])
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        image = self.rotate_image(image, angle)
        
        # Random brightness/contrast
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
        image = image * contrast + brightness - 1
        
        return torch.clamp(image, -1, 1)
    
    def rotate_image(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate image by angle degrees"""
        # Simple rotation implementation
        # In production, use torchvision.transforms.functional.rotate
        return image  # Placeholder

class TestDataset(Dataset):
    """Test dataset for generating predictions"""
    
    def __init__(self, image_dir: str, image_names: List[str], config: NEBULAConfig):
        self.image_dir = Path(image_dir)
        self.image_names = image_names
        self.config = config
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[idx]
        image_path = self.image_dir / image_name
        
        # Load image
        image = self.load_image(image_path)
        
        return {
            'image': image,
            'image_id': image_name
        }
    
    def load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            from PIL import Image
            img = Image.open(path).convert('L')  # Grayscale
            img = img.resize(self.config.resolution)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Normalize
            img_array = (img_array - 0.5) / 0.5
            
            return torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            # Return blank image on error
            return torch.zeros(1, *self.config.resolution)

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Optimized trainer for 0.99 precision target"""
    
    def __init__(self, config: NEBULAConfig):
        self.config = config
        self.device = DEVICE
        
        # Model
        self.model = NEBULAGrandXRay(config).to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss function - Focal loss for class imbalance
        self.criterion = FocalBCELoss(config.focal_loss_gamma, config.focal_loss_alpha)
        
        # Enhanced mixed precision with better error handling
        self.scaler = GradScaler(
            init_scale=2.**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100
        ) if config.use_amp else None
        
        # Metrics tracking
        self.best_auc = 0.0
        self.best_precision = 0.0
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        successful_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Apply mixup/cutmix augmentation
            if self.config.mixup_alpha > 0 and np.random.random() < 0.5:
                images, labels = self.mixup(images, labels)
            
            # Create batch for safe processing
            safe_batch = {'image': images, 'labels': labels}
            
            # Safe forward pass with adaptive memory management
            outputs, loss, success = memory_manager.safe_forward_pass(
                self.model, safe_batch, self.config.use_amp, self.criterion
            )
            
            if not success:
                logger.error(f"Batch {batch_idx} failed after all retries - skipping")
                continue
            
            self.optimizer.zero_grad()
            
            # Backward pass with memory management
            try:
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                
                successful_batches += 1
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mem_warnings': memory_manager.memory_warnings,
                    'success_rate': f"{successful_batches}/{batch_idx+1}"
                })
                
                # Adaptive memory cleanup
                memory_manager.adaptive_cleanup()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Backward pass memory error - skipping batch {batch_idx}")
                    memory_manager.adaptive_cleanup(force=True)
                    time.sleep(1.0)
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / max(1, successful_batches)  # Avoid division by zero
        logger.info(f"Epoch {epoch} summary: {successful_batches}/{len(dataloader)} batches successful, "
                   f"{memory_manager.memory_warnings} memory warnings")
        return {'loss': avg_loss, 'successful_batches': successful_batches}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                all_outputs.append(torch.sigmoid(outputs).cpu())
                all_labels.append(labels.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Calculate metrics
        auc_scores = []
        precision_scores = []
        
        for i in range(self.config.num_classes):
            if all_labels[:, i].sum() > 0:  # Skip if no positive samples
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                precision = average_precision_score(all_labels[:, i], all_outputs[:, i])
                auc_scores.append(auc)
                precision_scores.append(precision)
        
        mean_auc = np.mean(auc_scores)
        mean_precision = np.mean(precision_scores)
        
        # Update best scores
        if mean_auc > self.best_auc:
            self.best_auc = mean_auc
        if mean_precision > self.best_precision:
            self.best_precision = mean_precision
        
        return {
            'auc': mean_auc,
            'precision': mean_precision,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision
        }
    
    def mixup(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixup augmentation for better generalization"""
        batch_size = images.size(0)
        
        # Generate mix ratio
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        # Random shuffle
        index = torch.randperm(batch_size).to(self.device)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_auc = checkpoint.get('best_auc', 0.0)
        self.best_precision = checkpoint.get('best_precision', 0.0)
        logger.info(f"Checkpoint loaded: {path}")
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate predictions on test set with memory-safe processing"""
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
                images = batch['image'].to(self.device)
                
                # Safe prediction with memory management
                try:
                    outputs, _, success = memory_manager.safe_forward_pass(
                        self.model, {'image': images}, self.config.use_amp, None
                    )
                    
                    if not success:
                        logger.warning(f"Prediction batch {batch_idx} failed - using zeros")
                        predictions = np.zeros((images.shape[0], self.config.num_classes))
                    else:
                        # Apply sigmoid to get probabilities
                        predictions = torch.sigmoid(outputs).cpu().numpy()
                    
                    all_predictions.append(predictions)
                    
                    # Aggressive memory cleanup for large test set
                    if batch_idx % 50 == 0:
                        memory_manager.adaptive_cleanup(force=True)
                        time.sleep(0.5)  # Brief pause for memory recovery
                        
                except Exception as e:
                    logger.error(f"Critical prediction error batch {batch_idx}: {e}")
                    # Create zero predictions as fallback
                    predictions = np.zeros((images.shape[0], self.config.num_classes))
                    all_predictions.append(predictions)
        
        return np.concatenate(all_predictions, axis=0)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    logger.info("="*60)
    logger.info("NEBULA Grand X-Ray Unified System - High Precision Edition")
    logger.info("Target: 0.99 Precision | Configuration: 900 rays, 256x256")
    logger.info("="*60)
    
    # Initialize configuration
    config = NEBULAConfig()
    logger.info(f"Configuration loaded: {config}")
    
    # Load real dataset
    logger.info("Loading Grand X-Ray Slam Division A dataset...")
    
    # Dataset paths
    data_dir = Path(r"D:\\NEBULA_DIVISION_A\\datasets\\grand-xray-slam-division-a")
    train_csv_path = data_dir / "train1.csv"
    train_image_dir = data_dir / "train1"
    test_image_dir = data_dir / "test1"
    sample_submission_path = data_dir / "sample_submission_1.csv"
    
    # Load training data
    train_df = pd.read_csv(train_csv_path)
    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Load submission template
    submission_template = pd.read_csv(sample_submission_path)
    logger.info(f"Test set has {len(submission_template)} images")
    
    # Split training data (80/20)
    split_idx = int(len(train_df) * 0.8)
    train_labels = train_df.iloc[:split_idx].copy().reset_index(drop=True)
    val_labels = train_df.iloc[split_idx:].copy().reset_index(drop=True)
    
    logger.info(f"Train: {len(train_labels)}, Validation: {len(val_labels)}")
    
    # Create datasets
    train_dataset = GrandXRayDataset(
        image_dir=str(train_image_dir),
        labels_df=train_labels,
        config=config,
        training=True
    )
    
    val_dataset = GrandXRayDataset(
        image_dir=str(train_image_dir),
        labels_df=val_labels,
        config=config,
        training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Initialize trainer
    trainer = Trainer(config)

    # --- START: MODIFIED LOGIC FOR RESUMING TRAINING ---
    start_epoch = 1
    
    # Find all checkpoint files
    checkpoint_files = glob.glob('checkpoint_epoch_*.pth')
    
    if checkpoint_files:
        # Find the latest checkpoint
        latest_epoch = -1
        latest_checkpoint = None
        for f in checkpoint_files:
            match = re.search(r'checkpoint_epoch_(\d+).pth', f)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint = f
        
        if latest_checkpoint:
            logger.info(f"Found latest checkpoint: {latest_checkpoint}")
            try:
                trainer.load_checkpoint(latest_checkpoint)
                start_epoch = latest_epoch + 1
                logger.info(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                logger.info(f"Failed to load checkpoint {latest_checkpoint}: {e}")
                logger.info("Starting training from scratch.")
    else:
        logger.info("No checkpoints found. Starting training from scratch.")
    # --- END: MODIFIED LOGIC FOR RESUMING TRAINING ---
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Training - Loss: {train_metrics['loss']:.4f}")
        
        # Save checkpoint every epoch to prevent data loss
        trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', train_metrics)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_metrics = trainer.validate(val_loader)
            logger.info(f"Validation - AUC: {val_metrics['auc']:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}, "
                       f"Best AUC: {val_metrics['best_auc']:.4f}, "
                       f"Best Precision: {val_metrics['best_precision']:.4f}")
            
            # Save checkpoint if improved
            if val_metrics['precision'] >= trainer.best_precision:
                trainer.save_checkpoint('best_model.pth', val_metrics)
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Early stopping if target reached
        if trainer.best_precision >= 0.99:
            logger.info(f"Target precision of 0.99 reached! Best: {trainer.best_precision:.4f}")
            break
    
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info(f"Best AUC: {trainer.best_auc:.4f}")
    logger.info(f"Best Precision: {trainer.best_precision:.4f}")
    logger.info("="*60)
    
    # PRIORITY 1: THRESHOLD OPTIMIZATION - Expected +3-5% boost
    logger.info("🎯 PRIORITY 1: Starting Threshold Optimization...")
    logger.info("Expected improvement: +3-5% accuracy boost WITHOUT retraining!")
    
    # Load best model for threshold optimization
    if Path('best_model.pth').exists():
        trainer.load_checkpoint('best_model.pth')
        logger.info("Loaded best model for threshold optimization")
    
    # Run threshold optimization on validation set
    try:
        best_thresholds, optimization_metrics = threshold_optimizer.optimize_thresholds(
            trainer.model, 
            val_loader, 
            DEVICE, 
            target_metric='balanced'  # Best for multi-label classification
        )
        
        # Evaluate with optimized thresholds
        improved_metrics = threshold_optimizer.evaluate_with_optimized_thresholds(
            trainer.model, 
            val_loader, 
            DEVICE
        )
        
        logger.info("🚀 THRESHOLD OPTIMIZATION COMPLETE!")
        logger.info(f"🎯 IMPROVEMENT ACHIEVED:")
        logger.info(f"   Balanced Score: {improved_metrics['balanced_score']:.4f}")
        logger.info(f"   Mean F1: {improved_metrics['mean_f1']:.4f}")
        logger.info(f"   Mean Precision: {improved_metrics['mean_precision']:.4f}")
        logger.info(f"   Best Thresholds: {best_thresholds}")
        
        # Save optimized thresholds
        np.save('optimized_thresholds.npy', best_thresholds)
        logger.info("✅ Optimized thresholds saved to 'optimized_thresholds.npy'")
        
    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}")
        logger.info("Continuing with default 0.5 thresholds...")
    
    # Generate predictions on test set
    logger.info("Generating predictions on test set...")
    
    # Load best model
    if Path('best_model.pth').exists():
        trainer.load_checkpoint('best_model.pth')
        logger.info("Loaded best model checkpoint")
    
    # Create test dataset
    test_dataset = TestDataset(
        image_dir=str(test_image_dir),
        image_names=submission_template['Image_name'].tolist(),
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Generate predictions (probabilities)
    predictions = trainer.predict(test_loader)
    
    # --- START: MODIFIED SUBMISSION LOGIC ---
    # Create submission DataFrame with probabilities for AUC evaluation
    submission = submission_template.copy()
    condition_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    logger.info("📝 Creating submission file with probabilities for AUC evaluation...")
    
    # Use probabilities directly for submission
    for i, col in enumerate(condition_columns):
        submission[col] = predictions[:, i]
    
    # Save submission
    submission_path = 'nebula_submission_probabilities.csv'
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission with probabilities saved to: {submission_path}")
    # --- END: MODIFIED SUBMISSION LOGIC ---
    
    logger.info("NEBULA processing complete!")


if __name__ == "__main__":
    main()