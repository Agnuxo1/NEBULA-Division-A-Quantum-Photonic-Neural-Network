#!/usr/bin/env python3
"""
Analyze Checkpoint Issues
========================
Investigate why epoch 28 checkpoint didn't load correctly
"""

import torch
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint file in detail"""
    
    logger.info(f"=== ANALYZING CHECKPOINT ===")
    logger.info(f"File: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # File stats
    stat = os.stat(checkpoint_path)
    size_mb = stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    
    logger.info(f"Size: {size_mb:.1f} MB")
    logger.info(f"Modified: {mod_time}")
    
    try:
        # Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        logger.info("Checkpoint contents:")
        for key, value in checkpoint.items():
            if key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
                logger.info(f"  {key}: <state_dict with {len(value)} keys>")
            else:
                logger.info(f"  {key}: {value}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None

def find_latest_checkpoint():
    """Find the actual latest checkpoint"""
    
    models_dir = r"D:\NEBULA_DIVISION_A\models"
    
    # Find all epoch checkpoints
    import glob
    pattern = os.path.join(models_dir, 'nebula_official_epoch_*.pth')
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        logger.error("No checkpoint files found!")
        return None
    
    # Sort by epoch number
    def get_epoch_num(filepath):
        filename = os.path.basename(filepath)
        epoch_part = filename.split('_')[-1].split('.')[0]
        return int(epoch_part)
    
    sorted_checkpoints = sorted(checkpoint_files, key=get_epoch_num)
    
    logger.info("Available checkpoints:")
    for cp in sorted_checkpoints:
        epoch_num = get_epoch_num(cp)
        size_mb = os.path.getsize(cp) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(cp))
        logger.info(f"  Epoch {epoch_num:2d}: {size_mb:6.1f} MB - {mod_time}")
    
    # Return the highest epoch (should be 28)
    latest = sorted_checkpoints[-1]
    latest_epoch = get_epoch_num(latest)
    
    logger.info(f"Latest checkpoint: Epoch {latest_epoch}")
    return latest, latest_epoch

def test_checkpoint_loading():
    """Test if checkpoint loading logic works correctly"""
    
    logger.info("=== TESTING CHECKPOINT LOADING LOGIC ===")
    
    # This mimics the logic in NEBULA_Official_Training.py
    models_dir = r"D:\NEBULA_DIVISION_A\models"
    
    import glob
    checkpoint_pattern = os.path.join(models_dir, 'nebula_official_epoch_*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    logger.info(f"Pattern: {checkpoint_pattern}")
    logger.info(f"Found files: {len(checkpoint_files)}")
    
    if not checkpoint_files:
        logger.info("No checkpoints found - would start from scratch ✗")
        return False
    
    # Get the latest checkpoint (this is the original logic)
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    logger.info(f"Selected checkpoint: {latest_checkpoint}")
    
    # Test loading
    checkpoint = analyze_checkpoint(latest_checkpoint)
    
    if checkpoint:
        epoch = checkpoint.get('epoch', 0)
        best_auc = checkpoint.get('best_auc', 0.0)
        
        logger.info(f"✓ Checkpoint loads successfully")
        logger.info(f"✓ Would resume from epoch {epoch + 1}")
        logger.info(f"✓ Best AUC: {best_auc:.4f}")
        
        return True
    else:
        logger.error("✗ Checkpoint loading failed")
        return False

def diagnose_restart_issue():
    """Diagnose why training restarted from epoch 0"""
    
    logger.info("=== DIAGNOSING RESTART ISSUE ===")
    
    # Check if epoch_0000.pth was created today
    epoch_0_file = r"D:\NEBULA_DIVISION_A\models\nebula_official_epoch_0000.pth"
    
    if os.path.exists(epoch_0_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(epoch_0_file))
        today = datetime.now().date()
        
        if mod_time.date() == today:
            logger.warning(f"⚠️  epoch_0000.pth was created TODAY at {mod_time}")
            logger.warning("This confirms training restarted from scratch instead of loading checkpoint")
            
            # Compare with epoch 28
            epoch_28_file = r"D:\NEBULA_DIVISION_A\models\nebula_official_epoch_0028.pth"
            if os.path.exists(epoch_28_file):
                logger.info("Comparing epoch 0 vs epoch 28:")
                
                cp_0 = analyze_checkpoint(epoch_0_file)
                cp_28 = analyze_checkpoint(epoch_28_file)
                
                if cp_0 and cp_28:
                    logger.info(f"Epoch 0 AUC: {cp_0.get('best_auc', 0.0):.4f}")
                    logger.info(f"Epoch 28 AUC: {cp_28.get('best_auc', 0.0):.4f}")
                    
                    auc_loss = cp_28.get('best_auc', 0.0) - cp_0.get('best_auc', 0.0)
                    logger.error(f"💥 AUC LOSS: {auc_loss:.4f} points lost!")
        
        else:
            logger.info("epoch_0000.pth is old, not from today's restart")
    
    # Check for the actual cause
    logger.info("Possible causes:")
    logger.info("1. Checkpoint path was wrong")
    logger.info("2. Checkpoint file was corrupted") 
    logger.info("3. Loading logic had an exception")
    logger.info("4. Model architecture mismatch")

if __name__ == "__main__":
    logger.info("NEBULA Checkpoint Analysis")
    logger.info("=" * 50)
    
    # Step 1: Find latest checkpoint
    result = find_latest_checkpoint()
    if result:
        latest_file, latest_epoch = result
        
        # Step 2: Analyze the checkpoint
        logger.info(f"\n=== ANALYZING EPOCH {latest_epoch} CHECKPOINT ===")
        checkpoint = analyze_checkpoint(latest_file)
        
        # Step 3: Test loading logic
        logger.info(f"\n=== TESTING LOADING LOGIC ===")
        loading_works = test_checkpoint_loading()
        
        # Step 4: Diagnose restart issue  
        logger.info(f"\n=== DIAGNOSING RESTART ===")
        diagnose_restart_issue()
        
        # Summary
        logger.info("=" * 50)
        logger.info("ANALYSIS SUMMARY:")
        if checkpoint:
            logger.info(f"✓ Latest checkpoint: Epoch {latest_epoch}")
            logger.info(f"✓ Best AUC: {checkpoint.get('best_auc', 0.0):.4f}")
            logger.info(f"✓ Checkpoint is valid and loadable")
        
        if loading_works:
            logger.info("✓ Loading logic works correctly")
        else:
            logger.error("✗ Loading logic has issues")
            
    else:
        logger.error("No checkpoints found for analysis")