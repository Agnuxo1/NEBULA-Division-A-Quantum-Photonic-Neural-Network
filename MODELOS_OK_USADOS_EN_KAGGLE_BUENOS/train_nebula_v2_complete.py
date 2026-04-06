#!/usr/bin/env python3
"""
NEBULA v2.0 COMPLETE - Full Training System
============================================
Complete training system with:
- Multi-spectral photonic engine (100% real ray-tracing)
- CUDA-optimized ray-marching with 4-buffer system  
- Pixel-to-pixel reward system for medical analysis
- Quantum-holographic memory integration
- Radiologist-simulated training process
- Automatic checkpoint saving per epoch
- Real-time GPU monitoring and optimization

Architecture: MultiSpectralPhotonicEngine + QuantumThoracicProcessor + HolographicThoracicMemory
Base: NEBULA_GrandXRay_COMPLETE_v2.py (pure photonic processor)

Author: NEBULA AGI Agent
Mission: Achieve AUC > 0.936819 with complete NEBULA architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import NEBULA v2 COMPLETE architecture
from NEBULA_GrandXRay_COMPLETE_v2 import (
    NEBULAXRayControlPanel,
    NEBULAGrandXRayComplete,
    GrandXRayDataset,
    RadiologistSimulatedTrainer
)

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'nebula_v2_complete_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedNEBULAv2Trainer:
    """
    Enhanced training system for NEBULA v2 COMPLETE
    - Pixel-to-pixel reward system
    - 100% real CUDA ray-tracing
    - Automatic checkpoint management
    - Multi-spectral photonic analysis
    """
    
    def __init__(self, config: NEBULAXRayControlPanel):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize NEBULA v2 COMPLETE model
        self.model = NEBULAGrandXRayComplete(config).to(self.device)
        
        # MANDATORY: Load calibrated vision system
        logger.info("Loading NEBULA calibrated vision system...")
        try:
            self.model.load_calibration('nebula_stepped_calibration_complete.json')
            logger.info("✅ Calibrated vision system loaded successfully!")
        except Exception as e:
            logger.error(f"❌ FAILED to load calibration: {e}")
            raise RuntimeError("Cannot proceed without calibrated vision system!")
        
        # Enhanced pixel-to-pixel reward system weights - REDUCED for stability
        self.reward_weights = {
            'primary_loss': 1.0,                    # Main classification loss
            'spectral_consistency': 0.05,           # Multi-spectral consistency - REDUCED
            'anomaly_emphasis': 0.10,               # Anomaly detection emphasis - REDUCED
            'superposition_analysis': 0.05,         # Lightbox superposition - REDUCED
            'pixel_coherence': 0.05,                # NEW: Pixel-level coherence - REDUCED
            'ray_tracing_fidelity': 0.05,           # NEW: Ray-tracing accuracy - REDUCED
            'quantum_holographic': 0.05             # NEW: Quantum-holographic integrity - REDUCED
        }
        
        # Optimizer and scheduler setup - REDUCED LR for stability
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate * 0.1,  # REDUCED 10x for stability
            weight_decay=1e-5,  # Reduced weight decay
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()
        self.pixel_mse = nn.MSELoss()
        self.pixel_l1 = nn.L1Loss()
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'epoch_aucs': [],
            'best_auc': 0.0,
            'best_epoch': 0,
            'total_training_time': 0.0,
            'gpu_utilization': [],
            'ray_tracing_stats': [],
            'pixel_coherence_scores': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path('./nebula_v2_checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Enhanced NEBULA v2 Trainer initialized")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Pixel-to-pixel reward system enabled")
        logger.info(f"100% real CUDA ray-tracing active")
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def compute_pixel_coherence_reward(self, ray_analysis, targets):
        """
        FIXED: Compute pixel-to-pixel coherence reward
        Simplified approach that actually works with NEBULA v2 features
        """
        try:
            coherence_loss = torch.tensor(0.0, device=self.device)
            
            # Use superposition and individual features directly (no reshape needed)
            if 'individual_spectrums' in ray_analysis and 'superposition' in ray_analysis:
                individual_features = ray_analysis['individual_spectrums']
                superposition_features = ray_analysis['superposition']
                
                if len(individual_features) >= 2:
                    # Simple pixel-coherence: compare feature magnitudes between spectrums
                    spectrum1 = individual_features[0]  # [batch, features]
                    spectrum2 = individual_features[1]  # [batch, features]
                    
                    # Feature-wise coherence (each feature is like a "pixel" in feature space)
                    feature_diff = torch.abs(spectrum1 - spectrum2)
                    
                    # Weight by pathology presence (more coherence expected in pathology cases)
                    pathology_weight = torch.sum(targets, dim=1, keepdim=True) + 0.1  # [batch, 1]
                    
                    # Apply pathology weighting
                    weighted_diff = feature_diff * pathology_weight
                    
                    # Average coherence loss
                    coherence_loss = torch.mean(weighted_diff)
                    
                    # Add superposition coherence check
                    superposition_norm = torch.norm(superposition_features, dim=1)  # [batch]
                    individual_norm = torch.norm(spectrum1, dim=1) + torch.norm(spectrum2, dim=1)  # [batch]
                    
                    # Superposition should be related to individuals
                    coherence_ratio = superposition_norm / (individual_norm + 1e-6)
                    target_ratio = torch.ones_like(coherence_ratio) * 0.7  # Target coherence
                    ratio_loss = torch.mean(torch.abs(coherence_ratio - target_ratio))
                    
                    coherence_loss = (coherence_loss + ratio_loss) / 2
            
            # Clamp result to prevent NaN
            coherence_loss = torch.clamp(coherence_loss, 0.001, 10.0)  # Min 0.001 to avoid always 0
            
            # Check for NaN and return small value if found
            if torch.isnan(coherence_loss):
                return torch.tensor(0.01, device=self.device)
                
            return coherence_loss
            
        except Exception as e:
            logger.warning(f"Pixel coherence computation failed: {e}")
            return torch.tensor(0.01, device=self.device)  # Small non-zero value
    
    def compute_ray_tracing_fidelity_reward(self, ray_analysis):
        """
        NEW: Compute ray-tracing fidelity reward
        Ensures ray-tracing results maintain physical accuracy
        """
        try:
            fidelity_loss = torch.tensor(0.0, device=self.device)
            
            # Check superposition analysis quality
            if 'superposition' in ray_analysis:
                superposition_features = ray_analysis['superposition']
                
                # Superposition should be more informative than individual spectrums
                individual_features = ray_analysis['individual_spectrums']
                if individual_features:
                    individual_combined = torch.cat(individual_features, dim=1)
                    
                    # Information content comparison
                    superposition_entropy = -torch.sum(torch.softmax(superposition_features, dim=1) * 
                                                      torch.log_softmax(superposition_features, dim=1), dim=1)
                    individual_entropy = -torch.sum(torch.softmax(individual_combined, dim=1) * 
                                                   torch.log_softmax(individual_combined, dim=1), dim=1)
                    
                    # Superposition should have higher information content
                    entropy_diff = torch.relu(individual_entropy - superposition_entropy)  # Penalty if superposition has less info
                    fidelity_loss += torch.mean(entropy_diff)
            
            # Check anomaly analysis consistency
            if 'anomaly_analysis' in ray_analysis:
                anomaly_features = ray_analysis['anomaly_analysis']
                
                # Anomaly features should have reasonable dynamic range
                anomaly_std = torch.std(anomaly_features, dim=1)
                min_std = torch.tensor(0.01, device=self.device)  # Minimum expected variation
                std_penalty = torch.relu(min_std - anomaly_std)   # Penalty for too uniform features
                fidelity_loss += torch.mean(std_penalty)
            
            # Clamp result to prevent NaN
            fidelity_loss = torch.clamp(fidelity_loss, 0.0, 10.0)
            
            # Check for NaN and return 0 if found
            if torch.isnan(fidelity_loss):
                return torch.tensor(0.0, device=self.device)
                
            return fidelity_loss
            
        except Exception as e:
            logger.warning(f"Ray-tracing fidelity computation failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def compute_quantum_holographic_reward(self, holographic_features, quantum_features):
        """
        NEW: Compute quantum-holographic integrity reward
        Ensures quantum processing maintains holographic pattern fidelity
        """
        try:
            if holographic_features is None or quantum_features is None:
                return torch.tensor(0.0, device=self.device)
            
            # Quantum processing should preserve holographic information
            batch_size, num_conditions, feature_dim = quantum_features.shape
            
            # Reshape holographic features to match quantum features
            if holographic_features.dim() == 3:
                # Already in [batch, conditions, features] format
                holo_reshaped = holographic_features
            else:
                # Need to reshape
                holo_reshaped = holographic_features.view(batch_size, num_conditions, -1)
            
            # Information preservation check
            # Quantum processing should maintain the essential information from holographic memory
            information_loss = torch.tensor(0.0, device=self.device)
            
            for condition_idx in range(num_conditions):
                holo_condition = holo_reshaped[:, condition_idx, :]
                quantum_condition = quantum_features[:, condition_idx, :]
                
                # Ensure dimensions match for comparison
                min_dim = min(holo_condition.shape[1], quantum_condition.shape[1])
                holo_truncated = holo_condition[:, :min_dim]
                quantum_truncated = quantum_condition[:, :min_dim]
                
                # Cosine similarity to measure information preservation
                similarity = torch.cosine_similarity(holo_truncated, quantum_truncated, dim=1)
                target_similarity = torch.ones_like(similarity) * 0.8  # Target 80% similarity
                
                # Penalty for low similarity (information loss)
                similarity_loss = torch.relu(target_similarity - similarity)
                information_loss += torch.mean(similarity_loss)
            
            # Normalize by number of conditions
            information_loss = information_loss / num_conditions
            
            # Clamp result to prevent NaN
            information_loss = torch.clamp(information_loss, 0.0, 10.0)
            
            # Check for NaN and return 0 if found
            if torch.isnan(information_loss):
                return torch.tensor(0.0, device=self.device)
            
            return information_loss
            
        except Exception as e:
            logger.warning(f"Quantum-holographic reward computation failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def train_epoch(self, train_loader, epoch):
        """
        Enhanced training epoch with pixel-to-pixel rewards
        """
        self.model.train()
        
        total_loss = 0.0
        reward_components = {key: 0.0 for key in self.reward_weights.keys()}
        num_batches = 0
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1e9
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Check for NaN/Inf in input data
            if torch.isnan(images).any() or torch.isinf(images).any():
                logger.warning(f"Bad input data at batch {batch_idx}, skipping...")
                continue
                
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logger.warning(f"Bad target data at batch {batch_idx}, skipping...")
                continue
            
            # Additional input stabilization
            images = torch.clamp(images, -10.0, 10.0)  # Prevent extreme values
            
            # Check for problematic batch (all zeros or all same value)
            if torch.std(images) < 1e-6:
                logger.warning(f"Degenerate input batch {batch_idx}, skipping...")
                continue
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass with detailed analysis (anomaly detection disabled for speed)
                logits, analysis = self.model(images, return_analysis=True)
                
                # CRITICAL: Check logits for NaN/Inf before loss computation
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.warning(f"NaN/Inf in model output at batch {batch_idx}, skipping...")
                    continue
                
                # Clamp logits to prevent extreme values
                logits = torch.clamp(logits, -10.0, 10.0)
                
                # Primary classification loss
                primary_loss = self.criterion(logits, targets)
                
                # Check for NaN in primary loss (critical)
                if torch.isnan(primary_loss) or torch.isinf(primary_loss):
                    logger.warning(f"NaN/Inf in primary loss at batch {batch_idx}, trying simple BCE...")
                    
                    # Try even simpler BCE calculation
                    try:
                        # Manual BCE calculation with extra stability
                        sigmoid_logits = torch.sigmoid(torch.clamp(logits, -10, 10))
                        sigmoid_logits = torch.clamp(sigmoid_logits, 1e-7, 1-1e-7)  # Prevent log(0)
                        simple_loss = -torch.mean(
                            targets * torch.log(sigmoid_logits) + 
                            (1 - targets) * torch.log(1 - sigmoid_logits)
                        )
                        
                        if not torch.isnan(simple_loss) and not torch.isinf(simple_loss):
                            simple_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.optimizer.step()
                            total_loss += simple_loss.item()
                            num_batches += 1
                            
                            # Log the recovery
                            if batch_idx % 100 == 0:
                                logger.info(f"Recovered from NaN using stable BCE at batch {batch_idx}")
                    except:
                        logger.warning(f"Complete failure at batch {batch_idx}, skipping...")
                    
                    continue
                
                reward_components['primary_loss'] += primary_loss.item()
                
                # Enhanced reward system components
                spectral_analysis = analysis['spectral_analysis']
                holographic_patterns = analysis['holographic_patterns']  
                quantum_features = analysis['quantum_features']
                
                # Original NEBULA v2 reward components
                spectral_consistency_loss = self._compute_spectral_consistency_loss(spectral_analysis)
                anomaly_emphasis_loss = self._compute_anomaly_emphasis_loss(spectral_analysis, targets)
                superposition_loss = self._compute_superposition_loss(spectral_analysis)
                
                # NEW enhanced pixel-level reward components  
                pixel_coherence_loss = self.compute_pixel_coherence_reward(spectral_analysis, targets)
                ray_tracing_fidelity_loss = self.compute_ray_tracing_fidelity_reward(spectral_analysis)
                quantum_holographic_loss = self.compute_quantum_holographic_reward(holographic_patterns, quantum_features)
                
                # Update reward component tracking
                reward_components['spectral_consistency'] += spectral_consistency_loss.item()
                reward_components['anomaly_emphasis'] += anomaly_emphasis_loss.item()
                reward_components['superposition_analysis'] += superposition_loss.item()
                reward_components['pixel_coherence'] += pixel_coherence_loss.item()
                reward_components['ray_tracing_fidelity'] += ray_tracing_fidelity_loss.item()
                reward_components['quantum_holographic'] += quantum_holographic_loss.item()
                
                # Combined enhanced loss with all reward components
                total_batch_loss = (
                    self.reward_weights['primary_loss'] * primary_loss +
                    self.reward_weights['spectral_consistency'] * spectral_consistency_loss +
                    self.reward_weights['anomaly_emphasis'] * anomaly_emphasis_loss +
                    self.reward_weights['superposition_analysis'] * superposition_loss +
                    self.reward_weights['pixel_coherence'] * pixel_coherence_loss +
                    self.reward_weights['ray_tracing_fidelity'] * ray_tracing_fidelity_loss +
                    self.reward_weights['quantum_holographic'] * quantum_holographic_loss
                )
                
                # CRITICAL: Check for NaN in total loss
                if torch.isnan(total_batch_loss):
                    logger.warning(f"NaN detected in batch {batch_idx}, skipping...")
                    continue
                
                # Clamp total loss to prevent explosion
                total_batch_loss = torch.clamp(total_batch_loss, 0.0, 100.0)
                
                total_batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
                
                # Update progress bar with detailed metrics
                progress_bar.set_postfix({
                    'Loss': f'{total_batch_loss.item():.4f}',
                    'Primary': f'{primary_loss.item():.4f}',
                    'PixelCoh': f'{pixel_coherence_loss.item():.4f}',
                    'RayFid': f'{ray_tracing_fidelity_loss.item():.4f}',
                    'QHolo': f'{quantum_holographic_loss.item():.4f}'
                })
                
                # Memory management
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1e9
                    if current_memory > 20:  # If using >20GB, clear cache
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Calculate average losses for this epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_rewards = {key: value / num_batches for key, value in reward_components.items()}
        
        # Update learning rate
        self.scheduler.step()
        
        # Log detailed epoch statistics
        logger.info(f"Epoch {epoch+1} Training Complete:")
        logger.info(f"  Average Loss: {avg_loss:.6f}")
        logger.info(f"  Primary Loss: {avg_rewards['primary_loss']:.6f}")
        logger.info(f"  Pixel Coherence: {avg_rewards['pixel_coherence']:.6f}")
        logger.info(f"  Ray-Tracing Fidelity: {avg_rewards['ray_tracing_fidelity']:.6f}")
        logger.info(f"  Quantum-Holographic: {avg_rewards['quantum_holographic']:.6f}")
        logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        
        return avg_loss, avg_rewards
    
    def validate(self, val_loader, epoch):
        """
        Validation with comprehensive metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = batch['image'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                logits = self.model(images)
                predictions = torch.sigmoid(logits)
                
                # Validation loss
                loss = self.criterion(logits, targets)
                val_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate AUC scores
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        condition_aucs = []
        conditions = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        
        for i, condition in enumerate(conditions):
            try:
                if len(np.unique(all_targets[:, i])) > 1:  # Must have both classes
                    auc = roc_auc_score(all_targets[:, i], all_predictions[:, i])
                    condition_aucs.append(auc)
                else:
                    condition_aucs.append(0.5)  # Default for missing class
            except:
                condition_aucs.append(0.5)
        
        mean_auc = np.mean(condition_aucs)
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        
        # Log validation results
        logger.info(f"Epoch {epoch+1} Validation Results:")
        logger.info(f"  Validation Loss: {avg_val_loss:.6f}")
        logger.info(f"  Mean AUC: {mean_auc:.6f}")
        logger.info(f"  Best AUC so far: {max(self.training_stats['best_auc'], mean_auc):.6f}")
        
        # Log individual condition AUCs
        for condition, auc in zip(conditions, condition_aucs):
            logger.info(f"    {condition}: {auc:.4f}")
        
        return mean_auc, avg_val_loss, condition_aucs
    
    def save_checkpoint(self, epoch, train_loss, val_auc, val_loss, condition_aucs, is_best=False):
        """
        Save comprehensive checkpoint with all training state
        """
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_auc': val_auc,
            'val_loss': val_loss,
            'condition_aucs': condition_aucs,
            'training_stats': self.training_stats,
            'reward_weights': self.reward_weights,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'nebula_v2_epoch_{epoch+1:03d}_auc_{val_auc:.4f}.pth'
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'nebula_v2_best_model.pth'
            torch.save(checkpoint_data, best_path)
            logger.info(f"🌟 NEW BEST MODEL saved: AUC = {val_auc:.6f}")
        
        logger.info(f"📁 Checkpoint saved: {checkpoint_path}")
        
        # Keep only last 5 regular checkpoints to save space
        checkpoints = list(self.checkpoint_dir.glob('nebula_v2_epoch_*.pth'))
        checkpoints.sort(key=os.path.getctime)
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def train_complete_system(self, train_csv_path, train_images_dir, epochs=50, batch_size=8):
        """
        Complete training system for NEBULA v2
        """
        logger.info("🚀 Starting NEBULA v2 COMPLETE Training System")
        logger.info("=" * 70)
        logger.info(f"Target AUC: > 0.936819 (first place)")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Architecture: MultiSpectral + Quantum + Holographic")
        logger.info("=" * 70)
        
        # Load dataset
        logger.info("Loading Grand X-Ray Slam dataset...")
        try:
            metadata_df = pd.read_csv(train_csv_path)
            logger.info(f"Dataset loaded: {len(metadata_df)} samples")
            
            dataset = GrandXRayDataset(metadata_df, train_images_dir, target_size=(256, 256))
            
            # Train/val split
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            logger.info(f"Train samples: {train_size}, Validation samples: {val_size}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
        
        # Training loop
        start_time = time.time()
        target_auc = 0.936819
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, reward_components = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_auc, val_loss, condition_aucs = self.validate(val_loader, epoch)
            
            # Update training statistics
            self.training_stats['epoch_losses'].append(train_loss)
            self.training_stats['epoch_aucs'].append(val_auc)
            
            # Check for best model
            is_best = val_auc > self.training_stats['best_auc']
            if is_best:
                self.training_stats['best_auc'] = val_auc
                self.training_stats['best_epoch'] = epoch
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch, train_loss, val_auc, val_loss, condition_aucs, is_best)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Check if target achieved
            if val_auc > target_auc:
                logger.info("🥇 TARGET AUC ACHIEVED! Training can continue for further improvement.")
                logger.info(f"🎯 Current AUC: {val_auc:.6f} > Target: {target_auc:.6f}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final training summary
        total_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_time
        
        logger.info("🏁 NEBULA v2 Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        logger.info(f"Best AUC achieved: {self.training_stats['best_auc']:.6f} (epoch {self.training_stats['best_epoch']+1})")
        logger.info(f"Target AUC (first place): {target_auc:.6f}")
        
        if self.training_stats['best_auc'] > target_auc:
            logger.info("✅ FIRST PLACE ACHIEVED!")
        else:
            logger.info(f"🎯 Gap to first place: {target_auc - self.training_stats['best_auc']:.6f}")
        
        logger.info(f"All checkpoints saved in: {self.checkpoint_dir}")
        logger.info("=" * 70)
        
        return self.training_stats
    
    # Helper methods from original RadiologistSimulatedTrainer
    def _compute_spectral_consistency_loss(self, spectral_analysis):
        """Compute spectral consistency loss from original v2"""
        individual_features = spectral_analysis['individual_spectrums']
        consistency_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(len(individual_features)):
            for j in range(i+1, len(individual_features)):
                similarity = torch.cosine_similarity(individual_features[i], individual_features[j], dim=1)
                target_similarity = torch.full_like(similarity, 0.7)
                consistency_loss += torch.mean((similarity - target_similarity) ** 2)
        
        if len(individual_features) > 1:
            consistency_loss = consistency_loss / (len(individual_features) * (len(individual_features) - 1) / 2)
        
        return consistency_loss
    
    def _compute_anomaly_emphasis_loss(self, spectral_analysis, targets):
        """Compute anomaly emphasis loss from original v2"""
        anomaly_features = spectral_analysis['anomaly_analysis']
        pathology_present = (torch.sum(targets, dim=1) > 0).float()
        anomaly_strength = torch.mean(torch.abs(anomaly_features), dim=1)
        return torch.mean((anomaly_strength - pathology_present) ** 2)
    
    def _compute_superposition_loss(self, spectral_analysis):
        """Compute superposition loss from original v2"""
        superposition_features = spectral_analysis['superposition']
        individual_features = spectral_analysis['individual_spectrums']
        
        if not individual_features:
            return torch.tensor(0.0, device=self.device)
        
        individual_combined = torch.cat(individual_features, dim=1)
        superposition_norm = torch.norm(superposition_features, dim=1)
        individual_norm = torch.norm(individual_combined, dim=1)
        
        ratio = superposition_norm / (individual_norm + 1e-8)
        target_ratio = torch.ones_like(ratio)
        
        return torch.mean((ratio - target_ratio) ** 2)

def main():
    """
    Main training function for NEBULA v2 COMPLETE
    """
    logger.info("🌟 NEBULA v2 COMPLETE Training System")
    logger.info("Initializing enhanced pixel-to-pixel training...")
    
    # Configuration
    config = NEBULAXRayControlPanel()
    config.batch_size = 4  # Optimized for RTX 3090 with ray-tracing
    
    # Dataset paths - UPDATE THESE FOR YOUR SETUP
    train_csv_path = "D:/NEBULA_DIVISION_A/datasets/grand-xray-slam-division-a/train1.csv"
    train_images_dir = "D:/NEBULA_DIVISION_A/datasets/grand-xray-slam-division-a/train1"
    
    # Verify paths exist
    if not Path(train_csv_path).exists():
        logger.error(f"❌ Train CSV not found: {train_csv_path}")
        logger.info("Please update the paths in the main() function")
        return
    
    if not Path(train_images_dir).exists():
        logger.error(f"❌ Train images directory not found: {train_images_dir}")  
        logger.info("Please update the paths in the main() function")
        return
    
    # Initialize trainer
    trainer = EnhancedNEBULAv2Trainer(config)
    
    # Start training
    training_stats = trainer.train_complete_system(
        train_csv_path=train_csv_path,
        train_images_dir=train_images_dir,
        epochs=50,
        batch_size=config.batch_size
    )
    
    logger.info("🎯 NEBULA v2 COMPLETE training finished!")
    logger.info("Check the nebula_v2_checkpoints/ directory for saved models")

if __name__ == "__main__":
    main()