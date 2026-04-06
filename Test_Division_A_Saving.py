#!/usr/bin/env python3
"""
Test Division A Model Saving
============================
Verify that Division A training manager saves models correctly

Francisco Angulo de Lafuente and NEBULA Team  
Educational License - September 2025
"""

import torch
import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(r'D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts')

from NEBULA_Official_Training import NEBULATrainingManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_division_a_saving():
    """Test Division A model saving"""
    logger.info("="*60)
    logger.info("TESTING DIVISION A MODEL SAVING")
    logger.info("="*60)
    
    # Create test directories
    dataset_path = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a"  
    test_save_dir = r"D:\NEBULA_DIVISION_A\test_models"
    os.makedirs(test_save_dir, exist_ok=True)
    
    try:
        # Initialize training manager
        logger.info("Creating NEBULATrainingManager...")
        trainer = NEBULATrainingManager(dataset_path, test_save_dir)
        
        # Setup datasets and model
        logger.info("Setting up datasets and model...")
        trainer.setup_datasets()
        trainer.setup_model()
        
        logger.info("Training manager initialized successfully")
        logger.info(f"Save directory: {trainer.model_save_dir}")
        logger.info(f"Save every: {trainer.config['save_every']} epochs")
        
        # Test direct save_checkpoint function
        logger.info("\nTesting save_checkpoint function...")
        
        # Set some test values
        trainer.current_epoch = 1
        trainer.best_auc = 0.8500
        
        # Test regular checkpoint save
        logger.info("Testing regular checkpoint save...")
        trainer.save_checkpoint(is_best=False)
        
        # Test best model save
        logger.info("Testing best model save...")  
        trainer.save_checkpoint(is_best=True)
        
        # List saved files
        logger.info("\nFiles in test directory:")
        test_files = os.listdir(test_save_dir)
        for file in test_files:
            file_path = os.path.join(test_save_dir, file)
            size = os.path.getsize(file_path)
            logger.info(f"  {file} ({size:,} bytes)")
        
        # Test loading
        logger.info("\nTesting model loading...")
        if test_files:
            test_file = os.path.join(test_save_dir, test_files[0])
            checkpoint = torch.load(test_file, map_location='cpu', weights_only=False)
            
            logger.info("Checkpoint contents:")
            for key, value in checkpoint.items():
                if key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
                    logger.info(f"  {key}: <state_dict>")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info("\n✓ Division A saving test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Division A saving test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_current_checkpoints():
    """Check current checkpoint status"""
    logger.info("="*60)  
    logger.info("CHECKING CURRENT DIVISION A CHECKPOINTS")
    logger.info("="*60)
    
    checkpoint_dir = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\checkpoints"
    
    try:
        files = sorted(os.listdir(checkpoint_dir))
        logger.info(f"Checkpoints in {checkpoint_dir}:")
        
        for file in files[-10:]:  # Last 10 files
            file_path = os.path.join(checkpoint_dir, file)
            stats = os.stat(file_path)
            size = stats.st_size
            mtime = datetime.fromtimestamp(stats.st_mtime)
            logger.info(f"  {file}: {size:,} bytes, modified: {mtime}")
        
        # Check if any files from today
        today_files = [f for f in files if datetime.fromtimestamp(os.path.getmtime(os.path.join(checkpoint_dir, f))).date() == datetime.now().date()]
        
        logger.info(f"\nFiles modified today: {len(today_files)}")
        for file in today_files:
            logger.info(f"  TODAY: {file}")
        
        if not today_files:
            logger.warning("⚠️  No checkpoints saved today - potential saving issue!")
        
        return len(today_files) > 0
        
    except Exception as e:
        logger.error(f"Error checking checkpoints: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Division A saving verification...")
    
    # Check current checkpoint status
    has_recent = check_current_checkpoints()
    
    # Test saving functionality
    test_success = test_division_a_saving()
    
    logger.info("="*60)
    logger.info("DIVISION A SAVING VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Recent checkpoints found: {'✓' if has_recent else '✗'}")
    logger.info(f"Saving test passed: {'✓' if test_success else '✗'}")
    
    if has_recent and test_success:
        logger.info("🎉 Division A saving appears to work correctly!")
    else:
        logger.warning("⚠️  Division A may have saving issues!")