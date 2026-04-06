#!/usr/bin/env python3
"""
Force Save Current Training State
=================================
Attempts to force save current training checkpoint without stopping training
"""

import os
import sys
import time
import logging

# Add to path
sys.path.append(r'D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_save_trigger():
    """Create a trigger file that the training script can check for"""
    
    # Create a trigger file
    trigger_file = r"D:\NEBULA_DIVISION_A\FORCE_SAVE_NOW.trigger"
    
    try:
        with open(trigger_file, 'w') as f:
            f.write(f"FORCE_SAVE_REQUESTED_{int(time.time())}\n")
            f.write("Please save current checkpoint immediately\n")
        
        logger.info(f"✓ Force save trigger created: {trigger_file}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create trigger: {e}")
        return False

def modify_config_for_immediate_save():
    """Temporarily modify config to force save on next opportunity"""
    
    config_file = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts\config.py"
    
    if not os.path.exists(config_file):
        logger.warning("Config file not found, trying alternative approach")
        return False
    
    try:
        # Read current config
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check if we can modify save_every
        if 'save_every' in content:
            # Backup original
            with open(config_file + '.backup', 'w') as f:
                f.write(content)
            
            # Modify to save every epoch
            modified_content = content.replace("'save_every': 5", "'save_every': 1")
            
            with open(config_file, 'w') as f:
                f.write(modified_content)
            
            logger.info("✓ Modified config to save every epoch")
            return True
            
    except Exception as e:
        logger.error(f"✗ Config modification failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== FORCE SAVE ATTEMPT ===")
    
    # Try trigger file approach
    if force_save_trigger():
        logger.info("Trigger file created - training may check for it")
    
    # Try config modification
    if modify_config_for_immediate_save():
        logger.info("Config modified to save more frequently")
    
    logger.info("Force save attempts completed")