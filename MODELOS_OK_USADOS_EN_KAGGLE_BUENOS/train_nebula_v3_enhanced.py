#!/usr/bin/env python3
"""
NEBULA v3.0 ENHANCED - Training Script
======================================
Training the enhanced hybrid architecture with:
- Multi-frequency ray-tracing (5 spectral bands)
- Enhanced sensor photosensitivity  
- Auto-calibration system
- Advanced detector parameters

Base: v1.0 Hybrid (Ray-tracing + CNN) - PROVEN WORKING
Enhanced: Auto-Vision Calibration improvements

Author: NEBULA AGI Agent
Mission: Achieve AUC > 0.936819 for first place
"""

import torch
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path

# Import the enhanced v3 architecture
from NEBULA_GrandXRay_v3_ENHANCED import (
    NEBULAControlPanel,
    RSNAMasterTrainer,
    GrandXRayDataset
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nebula_v3_enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_nebula_v3_enhanced():
    """Train NEBULA v3 Enhanced with auto-calibration"""
    
    logger.info("=" * 70)
    logger.info("NEBULA v3.0 ENHANCED TRAINING - FIRST PLACE TARGET")
    logger.info("=" * 70)
    logger.info("Enhanced Features:")
    logger.info("  - Multi-frequency ray-tracing (5 bands)")
    logger.info("  - Enhanced sensor photosensitivity")
    logger.info("  - Auto-calibration system")
    logger.info("  - Advanced detector gain (1.3)")
    logger.info("  - Optimal light intensity (0.8)")
    logger.info("  - Anti-dazzling protection (800 rays total)")
    logger.info("=" * 70)
    
    # Configuration with enhanced parameters
    config = NEBULAControlPanel()
    config.checkpoint_filename = "nebula_v3_enhanced.pth"
    
    # Set dataset paths for Grand X-Ray Slam Division A
    dataset_path = Path("D:/NEBULA_DIVISION_A/datasets/grand-xray-slam-division-a")
    train_csv = dataset_path / "train1.csv"
    
    # Update config paths to match official dataset structure
    config.data_dir = str(dataset_path)
    config.train_csv = str(train_csv) 
    config.train_images_dir = str(dataset_path / "train1")
    config.test_images_dir = str(dataset_path / "test1")
    config.scan_data_dir = str(dataset_path)  # Use root directory with train1.csv
    config.output_dir = str(Path("C:/NEBULA_DIVISION_A/MODELOS_OK_USADOS_EN_KAGGLE/outputs"))
    
    if not train_csv.exists():
        logger.error(f"Dataset not found: {train_csv}")
        logger.info("Available paths:")
        if dataset_path.exists():
            for item in dataset_path.iterdir():
                logger.info(f"  - {item}")
        return
    
    logger.info(f"Dataset found: {train_csv}")
    logger.info("Enhanced Parameters:")
    logger.info(f"  - Ray frequencies: {config.ray_frequencies}")
    logger.info(f"  - Light intensity: {config.light_intensity}")  
    logger.info(f"  - Detector gain: {config.detector_gain}")
    logger.info(f"  - Max rays: {config.max_rays} (anti-dazzling)")
    logger.info(f"  - CUDA buffers: 8 (optimized)")
    logger.info(f"  - Rays per buffer: {config.max_rays // 8}")
    logger.info(f"  - Auto-calibration: {config.enable_auto_calibration}")
    
    try:
        # Initialize the NEBULA v3 ENHANCED trainer
        logger.info("Initializing NEBULA v3 ENHANCED Master Trainer...")
        trainer = RSNAMasterTrainer(config)
        
        logger.info("🚀 Starting NEBULA v3 ENHANCED training toward AUC > 0.936819...")
        logger.info("🎯 Target: First place in Grand X-Ray Slam Division A")
        logger.info("=" * 70)
        
        # Start training with the real trainer
        trainer.train()
        
        logger.info("=" * 70)
        logger.info("🏁 NEBULA v3 ENHANCED training session completed!")
        logger.info("Check outputs directory for results and checkpoints")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        logger.error("Please ensure Grand X-Ray Slam dataset is properly located")
    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.exception("Full error traceback:")
        raise

if __name__ == "__main__":
    train_nebula_v3_enhanced()