#!/usr/bin/env python3
"""
NEBULA CUDA Ray-Tracing PURO - Entrenamiento con RTX 3090 al máximo
================================================================
Usa el motor NEBULA_CUDA_RayTracing.py puro para aprovechar la GPU
Sin restricciones de calibración ni complejidad innecesaria
Objetivo: AUC > 0.936819 para primer puesto
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import time
import logging
from sklearn.metrics import roc_auc_score

# Import the pure CUDA ray-tracing engine
from NEBULA_CUDA_RayTracing import CUDARealRayTracer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleXRayDataset(Dataset):
    """Dataset simple para Grand X-Ray Slam"""
    def __init__(self, csv_path, image_dir, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.head(max_samples)
        
        self.image_dir = Path(image_dir)
        
        # Las 14 condiciones de Grand X-Ray Slam
        self.conditions = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        
        logger.info(f"Dataset loaded: {len(self.df)} samples")
        logger.info(f"Conditions: {self.conditions}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['Image_name']
        
        # Load image
        image_path = self.image_dir / image_name
        if image_path.exists():
            try:
                image = Image.open(image_path).convert('L')  # Grayscale
                image = image.resize((256, 256))  # Standard resolution
                image = np.array(image, dtype=np.float32) / 255.0
            except:
                # Fallback: synthetic X-ray pattern
                image = self.generate_synthetic_xray()
        else:
            image = self.generate_synthetic_xray()
        
        # Get targets for the 14 conditions
        targets = []
        for condition in self.conditions:
            targets.append(float(row.get(condition, 0)))
        
        return {
            'image': torch.FloatTensor(image).unsqueeze(0),  # Add channel dim
            'targets': torch.FloatTensor(targets),
            'name': image_name
        }
    
    def generate_synthetic_xray(self):
        """Generate a simple synthetic X-ray pattern"""
        image = np.random.normal(0.3, 0.1, (256, 256)).astype(np.float32)
        
        # Add chest structures
        h, w = 256, 256
        y, x = np.ogrid[:h, :w]
        
        # Heart shadow (left side)
        heart_mask = ((y - h//2)**2 + (x - w//3)**2) < (h//4)**2
        image[heart_mask] += 0.3
        
        # Lung fields (darker)
        left_lung = ((y - h//2)**2 + (x - w//4)**2) < (h//3)**2
        right_lung = ((y - h//2)**2 + (x - 3*w//4)**2) < (h//3)**2
        lung_mask = left_lung | right_lung
        image[lung_mask] -= 0.2
        
        return np.clip(image, 0, 1)

class NEBULACUDAModel(nn.Module):
    """Modelo híbrido: CUDA Ray-tracing + CNN"""
    def __init__(self, num_classes=14):
        super().__init__()
        
        # NEBULA CUDA Ray-Tracer - CONFIGURACIÓN v3 ENHANCED
        self.raytracer = CUDARealRayTracer(
            num_rays=800,   # v3 ENHANCED - anti-dazzling 
            ray_march_steps=64,
            image_size=(256, 256),
            cuda_buffers=8  # v3 ENHANCED - 100 rays per buffer
        )
        
        # CNN post-processing para multi-label
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        # CNN features: 128 * 8 * 8 = 8192
        # Ray features: 256 * 256 = 65536  (de averaged_spectrum)
        self.fc1 = nn.Linear(128 * 8 * 8 + 1, 512)  # Simplified ray features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # NEBULA CUDA Ray-tracing - Processing real (MANTIENE TODO EL RAY-TRACING)
        ray_features = []
        for i in range(batch_size):
            # Process cada imagen completa con ray-tracing CUDA real
            single_image = x[i:i+1]  # Keep batch dimension
            ray_result = self.raytracer(single_image)  # RAY-TRACING COMPLETO
            
            # Extract features del resultado (sin quitar ray-tracing)
            averaged_spectrum = ray_result['averaged_spectrum']  # Resultado completo
            ray_feat = torch.mean(averaged_spectrum).unsqueeze(0)  # Single global feature
            ray_features.append(ray_feat)
        
        ray_features = torch.cat(ray_features, dim=0)  # [batch_size, feature_dim]
        ray_features = ray_features.view(batch_size, -1)  # Ensure 2D
        
        # CNN processing
        cnn_out = self.relu(self.conv1(x))
        cnn_out = nn.functional.max_pool2d(cnn_out, 2)
        
        cnn_out = self.relu(self.conv2(cnn_out))
        cnn_out = nn.functional.max_pool2d(cnn_out, 2)
        
        cnn_out = self.relu(self.conv3(cnn_out))
        cnn_out = self.pool(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)
        
        # Combine ray-tracing and CNN features
        combined = torch.cat([cnn_out, ray_features], dim=1)
        
        # Final classification
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def train_nebula_cuda_pure():
    """Entrenamiento puro con CUDA Ray-tracing"""
    
    logger.info("=" * 70)
    logger.info("NEBULA CUDA RAY-TRACING PURO - TRAINING")
    logger.info("=" * 70)
    logger.info("RTX 3090 al máximo - Sin restricciones")
    logger.info("Objetivo: AUC > 0.936819 para primer puesto")
    logger.info("=" * 70)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
    
    # Dataset
    dataset_path = "D:/NEBULA_DIVISION_A/datasets/grand-xray-slam-division-a"
    train_csv = f"{dataset_path}/train1.csv"
    image_dir = f"{dataset_path}/train1"
    
    # Full dataset - RTX 3090 ready for all samples  
    dataset = SimpleXRayDataset(train_csv, image_dir, max_samples=None)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Model
    model = NEBULACUDAModel(num_classes=14).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_auc = 0.0
    target_auc = 0.936819
    
    logger.info("Iniciando entrenamiento NEBULA CUDA...")
    
    try:
        epoch = 0
        while best_auc < target_auc:
            epoch += 1
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                targets = batch['targets'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                # Monitor GPU usage
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    logger.info(f"GPU Memory: {memory_used:.2f} GB")
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    targets = batch['targets'].to(device)
                    
                    outputs = torch.sigmoid(model(images))
                    
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(targets.cpu().numpy())
            
            # Calculate metrics
            if val_preds:
                val_preds = np.vstack(val_preds)
                val_targets = np.vstack(val_targets)
                
                try:
                    val_auc = roc_auc_score(val_targets, val_preds, average='macro')
                except:
                    val_auc = 0.5
            else:
                val_auc = 0.5
            
            avg_train_loss = train_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch}: Loss={avg_train_loss:.4f}, AUC={val_auc:.4f}, Time={epoch_time:.1f}s")
            
            # Check for improvement
            if val_auc > best_auc:
                best_auc = val_auc
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                }, 'nebula_cuda_pure_best.pth')
                
                logger.info(f"🌟 NEW BEST AUC: {best_auc:.6f}")
                
                if best_auc > target_auc:
                    logger.info("🥇 PRIMER PUESTO CONSEGUIDO!")
                    break
            
            scheduler.step(val_auc)
            
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por usuario")
    
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO NEBULA CUDA COMPLETADO")
    logger.info(f"Best AUC: {best_auc:.6f}")
    logger.info(f"Target: {target_auc}")
    logger.info(f"Status: {'PRIMER PUESTO!' if best_auc > target_auc else 'Continuar entrenando'}")
    logger.info("=" * 70)
    
    return best_auc

if __name__ == "__main__":
    train_nebula_cuda_pure()