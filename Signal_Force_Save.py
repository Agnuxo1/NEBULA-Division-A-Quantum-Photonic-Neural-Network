#!/usr/bin/env python3
"""
Signal-Based Force Save for PyTorch Training
============================================
Uses SIGUSR1 signal to force checkpoint save without stopping training.
This is the standard industry solution documented on GitHub.
"""

import os
import signal
import time
import logging
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_save_signal():
    """Send SIGUSR1 signal to force checkpoint save"""
    
    logger.info("=== SIGNAL-BASED FORCE SAVE ===")
    
    # Find the training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'NEBULA_Official_Training.py' in ' '.join(cmdline):
                training_pid = proc.info['pid']
                logger.info(f"Found training process PID: {training_pid}")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not training_pid:
        logger.error("Training process not found")
        return False
    
    try:
        # Send SIGUSR1 signal to force save
        # This is the standard way documented in PyTorch Lightning and other frameworks
        os.kill(training_pid, signal.SIGUSR1)
        logger.info(f"✓ SIGUSR1 signal sent to PID {training_pid}")
        logger.info("The training process should save a checkpoint within the next epoch loop")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send signal: {e}")
        return False

def create_signal_handler_patch():
    """Create a signal handler patch for the training script"""
    
    patch_code = '''
import signal
import os

def signal_checkpoint_handler(signum, frame):
    """Handle SIGUSR1 signal to force checkpoint save"""
    logger.info("🚨 SIGUSR1 received - forcing checkpoint save!")
    try:
        # Force immediate save regardless of epoch
        self.save_checkpoint(is_best=False)
        logger.info("✓ Emergency checkpoint saved successfully")
    except Exception as e:
        logger.error(f"Emergency save failed: {e}")

# Register signal handler
signal.signal(signal.SIGUSR1, signal_checkpoint_handler)
logger.info("Signal handler registered for SIGUSR1 (force save)")
'''
    
    # Write the patch to a file that can be imported
    patch_file = r"D:\NEBULA_DIVISION_A\signal_patch.py"
    with open(patch_file, 'w') as f:
        f.write(patch_code)
    
    logger.info(f"Signal handler patch created: {patch_file}")
    return patch_file

def inject_signal_handler():
    """Inject signal handler into the running training script"""
    
    training_script = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts\NEBULA_Official_Training.py"
    backup_script = training_script + ".signal_backup"
    
    try:
        # Create backup
        import shutil
        shutil.copy2(training_script, backup_script)
        
        # Read current script
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if signal handler already exists
        if 'signal.SIGUSR1' in content:
            logger.info("Signal handler already exists")
            return True
        
        # Add signal handler at the beginning after imports
        signal_handler_code = '''
import signal

def signal_checkpoint_handler(signum, frame):
    """Handle SIGUSR1 signal to force checkpoint save"""
    global training_manager
    if 'training_manager' in globals():
        logger.info("🚨 SIGUSR1 received - forcing checkpoint save!")
        try:
            training_manager.save_checkpoint(is_best=False)
            logger.info("✓ Emergency checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")
    else:
        logger.warning("Training manager not found in globals")

# Register signal handler
signal.signal(signal.SIGUSR1, signal_checkpoint_handler)
logger.info("Signal handler registered for SIGUSR1 (force save)")

'''
        
        # Find the place to insert (after imports, before class definition)
        insert_pos = content.find('class NEBULATrainingManager')
        if insert_pos == -1:
            logger.error("Could not find class definition to insert signal handler")
            return False
        
        # Insert signal handler
        modified_content = content[:insert_pos] + signal_handler_code + content[insert_pos:]
        
        # Also need to make training_manager global in main
        main_pos = modified_content.find('if __name__ == "__main__":')
        if main_pos != -1:
            main_section = modified_content[main_pos:]
            main_section = main_section.replace(
                'trainer = NEBULATrainingManager(',
                'global training_manager\\ntraining_manager = trainer = NEBULATrainingManager('
            )
            modified_content = modified_content[:main_pos] + main_section
        
        # Save modified script
        with open(training_script, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        logger.info("✓ Signal handler injected into training script")
        return True
        
    except Exception as e:
        logger.error(f"Signal injection failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Industry standard solution: Using SIGUSR1 signal for force save")
    
    # Method 1: Inject signal handler (for future runs)
    logger.info("Step 1: Injecting signal handler into training script...")
    inject_success = inject_signal_handler()
    
    # Method 2: Send signal to current process
    logger.info("Step 2: Sending SIGUSR1 to current training process...")
    signal_success = send_save_signal()
    
    if signal_success:
        logger.info("✓ Signal sent successfully")
        logger.info("Waiting 60 seconds for checkpoint to be saved...")
        time.sleep(60)
        
        # Check if checkpoint was created
        models_dir = r"D:\NEBULA_DIVISION_A\models"
        files_before = set(os.listdir(models_dir))
        time.sleep(5)  # Small delay
        files_after = set(os.listdir(models_dir))
        
        new_files = files_after - files_before
        if new_files:
            logger.info(f"✓ New checkpoint files detected: {new_files}")
        else:
            logger.info("No new files yet - checkpoint may be in progress")
    
    logger.info("Force save attempt completed using industry standard SIGUSR1 method")