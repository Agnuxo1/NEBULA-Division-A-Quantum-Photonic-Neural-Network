#!/usr/bin/env python3
"""
Windows Force Save Solution
===========================
Windows-compatible solution to force checkpoint save during training
Uses file-based signaling instead of Unix signals
"""

import os
import time
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_force_save_trigger():
    """Create trigger file that training script can detect"""
    
    logger.info("=== WINDOWS FORCE SAVE SOLUTION ===")
    
    # Create trigger file in models directory where training script can find it
    trigger_file = r"D:\NEBULA_DIVISION_A\models\FORCE_SAVE_NOW.trigger"
    
    try:
        with open(trigger_file, 'w') as f:
            f.write(f"FORCE_SAVE_REQUESTED\n")
            f.write(f"timestamp: {time.time()}\n")
            f.write(f"reason: User requested emergency checkpoint save\n")
            f.write(f"epoch: current\n")
        
        logger.info(f"✓ Force save trigger created: {trigger_file}")
        return trigger_file
        
    except Exception as e:
        logger.error(f"Failed to create trigger file: {e}")
        return None

def modify_training_for_trigger_check():
    """Modify the training script to check for trigger file"""
    
    training_script = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts\NEBULA_Official_Training.py"
    
    # Read current content
    try:
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if trigger check already exists
        if 'FORCE_SAVE_NOW.trigger' in content:
            logger.info("Trigger check already exists in script")
            return True
        
        # Find the training loop section and add trigger check
        # Look for the save checkpoint condition
        save_condition = "if epoch % self.config['save_every'] == 0 or is_best:"
        
        if save_condition in content:
            # Add trigger file check
            new_condition = '''# Check for force save trigger file
                trigger_file = os.path.join(self.model_save_dir, 'FORCE_SAVE_NOW.trigger')
                force_save = os.path.exists(trigger_file)
                
                if epoch % self.config['save_every'] == 0 or is_best or force_save:
                    if force_save:
                        logger.info("🚨 FORCE SAVE TRIGGERED by trigger file!")
                        try:
                            os.remove(trigger_file)  # Remove trigger after use
                            logger.info("✓ Trigger file removed")
                        except:
                            pass'''
            
            # Replace the condition
            modified_content = content.replace(save_condition, new_condition)
            
            # Write back
            with open(training_script, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            logger.info("✓ Training script modified to check for trigger file")
            return True
        else:
            logger.error("Could not find save condition to modify")
            return False
            
    except Exception as e:
        logger.error(f"Failed to modify training script: {e}")
        return False

def wait_for_checkpoint_creation():
    """Wait and monitor for new checkpoint creation"""
    
    models_dir = r"D:\NEBULA_DIVISION_A\models"
    
    # Get initial file list
    initial_files = set(os.listdir(models_dir))
    logger.info(f"Initial files in models directory: {len(initial_files)}")
    
    # Wait up to 2 minutes for new checkpoint
    max_wait = 120  # 2 minutes
    check_interval = 5  # Check every 5 seconds
    
    for i in range(0, max_wait, check_interval):
        time.sleep(check_interval)
        
        try:
            current_files = set(os.listdir(models_dir))
            new_files = current_files - initial_files
            
            if new_files:
                logger.info(f"✓ NEW CHECKPOINT DETECTED: {new_files}")
                for new_file in new_files:
                    file_path = os.path.join(models_dir, new_file)
                    size = os.path.getsize(file_path) / (1024*1024)  # MB
                    logger.info(f"  {new_file}: {size:.1f} MB")
                return True
                
            # Check if any existing files were modified recently
            for file in current_files:
                if file.endswith('.pth'):
                    file_path = os.path.join(models_dir, file)
                    mod_time = os.path.getmtime(file_path)
                    if time.time() - mod_time < 60:  # Modified in last minute
                        logger.info(f"✓ RECENTLY UPDATED: {file}")
            
            logger.info(f"Waiting... {i+check_interval}/{max_wait}s")
            
        except Exception as e:
            logger.error(f"Error checking files: {e}")
    
    logger.warning("No new checkpoint detected within timeout period")
    return False

def check_training_process():
    """Check if training process is still running"""
    
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'NEBULA_Official_Training.py' in ' '.join(cmdline):
                logger.info(f"Training process found (PID {proc.info['pid']})")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    logger.error("Training process not found")
    return False

if __name__ == "__main__":
    logger.info("Windows-compatible force save solution")
    
    # Step 1: Check if training is running
    if not check_training_process():
        logger.error("Training not running - cannot force save")
        exit(1)
    
    # Step 2: Modify training script for trigger checking
    logger.info("Step 1: Modifying training script for trigger detection...")
    modify_success = modify_training_for_trigger_check()
    
    if not modify_success:
        logger.error("Could not modify training script")
        exit(1)
    
    # Step 3: Create trigger file
    logger.info("Step 2: Creating force save trigger...")
    trigger_file = create_force_save_trigger()
    
    if not trigger_file:
        logger.error("Could not create trigger file")
        exit(1)
    
    # Step 4: Wait for checkpoint creation
    logger.info("Step 3: Waiting for checkpoint creation...")
    logger.info("The modified training script will check for the trigger file and save on next epoch iteration")
    
    success = wait_for_checkpoint_creation()
    
    if success:
        logger.info("✅ SUCCESS: Force save completed!")
    else:
        logger.warning("⚠️  Timeout: No new checkpoint detected, but trigger was sent")
        logger.info("The checkpoint may still be created when the training loop next checks")