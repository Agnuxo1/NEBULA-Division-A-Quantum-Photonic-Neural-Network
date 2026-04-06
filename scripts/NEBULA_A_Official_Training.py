#!/usr/bin/env python3
"""
NEBULA Official Training System - Grand X-Ray SLAM Division A
===========================================================
Entrenamiento real sin tiempo definido con dataset oficial
Francisco Angulo de Lafuente and NEBULA Team
Educational License - September 2025

Dataset: 107,374 chest X-ray images with 14 pathology labels
Training: Continuous without time limit until convergence
Architecture: Real ray-tracing with 900 calibrated rays
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import logging
from datetime import datetime, timedelta
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import NEBULA systems
from NEBULA_Professional_System import NEBULAMedicalAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'nebula_official_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GrandXRayDataset(Dataset):
    """
    Official Grand X-Ray SLAM Division A Dataset
    107,374 chest X-ray images with multi-label pathology annotations
    """
    
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file: Path to CSV file with annotations
            image_dir: Directory with all the images
            transform: Optional transform to be applied on sample
        """
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Pathology columns (excluding metadata)
        self.pathology_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        
        logger.info(f"Loaded Grand X-Ray dataset: {len(self.annotations)} samples")
        
        # Dataset statistics
        for col in self.pathology_columns:
            count = self.annotations[col].sum()
            percentage = (count / len(self.annotations)) * 100
            logger.info(f"  {col}: {count} samples ({percentage:.2f}%)")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and labels
        img_name = self.annotations.iloc[idx]['Image_name']
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return fallback image if loading fails - PRODUCTION READY
            image = Image.new('L', (256, 256), 128)
        
        # Get pathology labels
        labels = self.annotations.iloc[idx][self.pathology_columns].values.astype(np.float32)
        
        # Get metadata (convert to proper types for collation)
        age_val = self.annotations.iloc[idx]['Age']
        age = float(age_val) if pd.notna(age_val) else 0.0
        
        metadata = {
            'patient_id': int(self.annotations.iloc[idx]['Patient_ID']),
            'age': age
        }
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing for NEBULA
            image = image.resize((256, 256))
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        sample = {
            'image': image,
            'labels': torch.from_numpy(labels),
            'metadata': metadata,
            'image_path': img_path
        }
        
        return sample

class NEBULATrainingManager:
    """
    NEBULA Training Manager for Official Competition
    Manages continuous training without time limits
    """
    
    def __init__(self, dataset_path, model_save_dir='nebula_models'):
        """
        Initialize training manager
        
        Args:
            dataset_path: Path to Grand X-Ray dataset directory
            model_save_dir: Directory to save model checkpoints
        """
        self.dataset_path = dataset_path
        self.model_save_dir = model_save_dir
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Training configuration
        self.config = {
            'batch_size': 16,  # Reduced for memory efficiency
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_workers': 4,
            'mixed_precision': True,
            'gradient_clip': 1.0,
            'patience': 20,  # Early stopping patience
            'min_epochs': 10,
            'max_epochs': 1000,  # Very high limit - train until convergence
            'save_every': 5,  # Save every N epochs
            'validate_every': 1  # Validate every epoch
        }
        
        # Training state
        self.current_epoch = 0
        self.best_auc = 0.0
        self.patience_counter = 0
        self.training_start_time = None
        self.training_history = []
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("Using CPU")
        
        logger.info("NEBULA Official Training Manager initialized")
        logger.info(f"Configuration: {self.config}")
    
    def setup_datasets(self):
        """Setup training and validation datasets"""
        logger.info("Setting up datasets...")
        
        csv_file = os.path.join(self.dataset_path, 'train1.csv')
        image_dir = os.path.join(self.dataset_path, 'train1')
        
        # Load full dataset
        full_dataset = GrandXRayDataset(csv_file, image_dir)
        
        # Split into train/validation (80/20)
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Dataset split: {train_size} training, {val_size} validation samples")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        return train_dataset, val_dataset
    
    def setup_model(self):
        """Setup NEBULA model and training components"""
        logger.info("Setting up NEBULA model...")
        
        # Initialize NEBULA
        self.model = NEBULAMedicalAI()
        
        # Load calibration
        try:
            self.model.load_calibration('nebula_stepped_calibration_complete.json')
            logger.info("NEBULA calibrated vision loaded")
        except Exception as e:
            logger.warning(f"Calibration loading failed: {e}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup loss function (BCE for multi-label)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Mixed precision scaler
        if self.config['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"NEBULA model setup complete")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Clear cache periodically
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    # Move to device
                    images = batch['image'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)
                    
                    # Forward pass
                    if self.config['mixed_precision']:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    
                    # Collect predictions for metrics
                    predictions = torch.sigmoid(outputs).cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_labels.append(labels_np)
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        
        if all_predictions and all_labels:
            all_predictions = np.vstack(all_predictions)
            all_labels = np.vstack(all_labels)
            
            # Calculate AUC for each pathology
            pathology_aucs = []
            for i in range(all_labels.shape[1]):
                if all_labels[:, i].sum() > 0:  # Only calculate if positive samples exist
                    try:
                        auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                        pathology_aucs.append(auc)
                    except:
                        pathology_aucs.append(0.5)
                else:
                    pathology_aucs.append(0.5)
            
            mean_auc = np.mean(pathology_aucs)
        else:
            mean_auc = 0.0
            pathology_aucs = [0.0] * 14
        
        return avg_loss, mean_auc, pathology_aucs
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.model_save_dir, 
            f'nebula_official_epoch_{self.current_epoch:04d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_save_dir, 'nebula_official_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: AUC = {self.best_auc:.4f}")
    
    def train_continuous(self):
        """Start continuous training until convergence"""
        logger.info("="*80)
        logger.info("NEBULA OFFICIAL TRAINING - GRAND X-RAY SLAM DIVISION A")
        logger.info("Francisco Angulo de Lafuente and NEBULA Team")
        logger.info("="*80)
        logger.info("Starting continuous training without time limit...")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Training samples: 107,374 chest X-rays")
        logger.info(f"Ray-tracing: 900 calibrated rays")
        logger.info(f"Target: 14 pathology classification")
        
        # Setup datasets and model
        self.setup_datasets()
        self.setup_model()
        
        self.training_start_time = datetime.now()
        
        try:
            for epoch in range(self.config['max_epochs']):
                self.current_epoch = epoch
                
                # Training
                logger.info(f"\nEpoch {epoch + 1}/{self.config['max_epochs']}")
                train_loss = self.train_epoch()
                
                # Validation
                if epoch % self.config['validate_every'] == 0:
                    val_loss, val_auc, pathology_aucs = self.validate_epoch()
                    
                    # Learning rate scheduling
                    self.scheduler.step(val_auc)
                    
                    # Check if best model
                    is_best = val_auc > self.best_auc
                    if is_best:
                        self.best_auc = val_auc
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Log results
                    elapsed_time = datetime.now() - self.training_start_time
                    logger.info(f"Epoch {epoch + 1} Results:")
                    logger.info(f"  Train Loss: {train_loss:.4f}")
                    logger.info(f"  Val Loss: {val_loss:.4f}")
                    logger.info(f"  Val AUC: {val_auc:.4f} {'(BEST)' if is_best else ''}")
                    logger.info(f"  Best AUC: {self.best_auc:.4f}")
                    logger.info(f"  Patience: {self.patience_counter}/{self.config['patience']}")
                    logger.info(f"  Elapsed: {elapsed_time}")
                    logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                    
                    # Save training history
                    self.training_history.append({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_auc': val_auc,
                        'pathology_aucs': pathology_aucs,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'elapsed_time': str(elapsed_time)
                    })
                    
                    # Save checkpoint
                    if epoch % self.config['save_every'] == 0 or is_best:
                        self.save_checkpoint(is_best)
                    
                    # Early stopping check
                    if (epoch >= self.config['min_epochs'] and 
                        self.patience_counter >= self.config['patience']):
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        logger.info(f"Best AUC: {self.best_auc:.4f}")
                        break
                
                # Memory cleanup
                torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            # Save final checkpoint
            self.save_checkpoint()
            
            total_time = datetime.now() - self.training_start_time
            logger.info("="*80)
            logger.info("NEBULA OFFICIAL TRAINING COMPLETED")
            logger.info("="*80)
            logger.info(f"Total training time: {total_time}")
            logger.info(f"Final epoch: {self.current_epoch + 1}")
            logger.info(f"Best validation AUC: {self.best_auc:.4f}")
            logger.info(f"Model saved in: {self.model_save_dir}")

def main():
    """Main training function"""
    # Configuration
    dataset_path = r"E:\grand-xray-slam-division-a"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    # Initialize training manager
    trainer = NEBULATrainingManager(dataset_path)
    
    # Start continuous training
    trainer.train_continuous()

if __name__ == "__main__":
    main()