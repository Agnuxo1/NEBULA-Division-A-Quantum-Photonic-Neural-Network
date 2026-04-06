#!/usr/bin/env python3
"""
Windows Process Injection for Emergency PyTorch Checkpoint Save
===============================================================
Simplified Windows-compatible process injection to force torch.save()
Based on pydbattach principles but adapted for Windows
"""

import os
import sys
import psutil
import pickle
import socket
import threading
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowsPyTorchInjector:
    """Windows-compatible PyTorch process injector for emergency saves"""
    
    def __init__(self):
        self.target_pid = None
        self.injection_port = 31337  # Random high port
        
    def find_training_process(self):
        """Find the NEBULA training process"""
        for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'NEBULA_Official_Training.py' in ' '.join(cmdline):
                    self.target_pid = proc.info['pid']
                    logger.info(f"Found training process PID: {self.target_pid}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.error("Training process not found")
        return False
    
    def create_injection_payload(self):
        """Create PyTorch checkpoint save payload"""
        save_code = '''
import torch
import os
import logging
from datetime import datetime

def emergency_checkpoint_save():
    """Emergency checkpoint save function"""
    try:
        # Find model and optimizer in globals
        model_obj = None
        optimizer_obj = None
        epoch_num = 0
        best_auc = 0.0
        
        # Search through globals for PyTorch objects
        for name, obj in globals().items():
            if hasattr(obj, 'state_dict') and hasattr(obj, 'parameters'):
                if 'model' in name.lower() or 'nebula' in name.lower():
                    model_obj = obj
                    print(f"Found model object: {name}")
            elif hasattr(obj, 'state_dict') and hasattr(obj, 'param_groups'):
                if 'optim' in name.lower():
                    optimizer_obj = obj  
                    print(f"Found optimizer object: {name}")
            elif 'epoch' in name.lower() and isinstance(obj, int):
                epoch_num = obj
            elif 'auc' in name.lower() and isinstance(obj, (int, float)):
                best_auc = float(obj)
        
        # Try to find training manager if available
        for name, obj in globals().items():
            if hasattr(obj, 'model') and hasattr(obj, 'optimizer'):
                if 'trainer' in name.lower() or 'manager' in name.lower():
                    model_obj = obj.model
                    optimizer_obj = obj.optimizer
                    if hasattr(obj, 'current_epoch'):
                        epoch_num = obj.current_epoch
                    if hasattr(obj, 'best_auc'):
                        best_auc = obj.best_auc
                    print(f"Found training manager: {name}")
                    break
        
        if model_obj is None:
            return "ERROR: No PyTorch model found in globals"
        
        # Create emergency checkpoint
        checkpoint = {
            'epoch': epoch_num,
            'emergency_save': True,
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': model_obj.state_dict(),
        }
        
        if optimizer_obj:
            checkpoint['optimizer_state_dict'] = optimizer_obj.state_dict()
        
        if best_auc > 0:
            checkpoint['best_auc'] = best_auc
        
        # Save emergency checkpoint
        save_path = r"D:\\NEBULA_DIVISION_A\\models\\emergency_injected_checkpoint.pth"
        torch.save(checkpoint, save_path)
        
        # Get file size
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        
        return f"SUCCESS: Emergency checkpoint saved to {save_path} ({file_size:.1f} MB)"
        
    except Exception as e:
        return f"ERROR: Emergency save failed: {str(e)}"

# Execute the save
result = emergency_checkpoint_save()
print(f"INJECTION RESULT: {result}")
'''
        return save_code
    
    def create_socket_injection_script(self):
        """Create script that will be injected via socket communication"""
        injection_script = f"""
import socket
import pickle
import sys
import threading

def socket_injection_handler():
    try:
        # Connect back to injector
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', {self.injection_port}))
        
        # Send confirmation
        sock.send(b"INJECTION_READY")
        
        # Receive and execute code
        data = sock.recv(65536)
        code_to_execute = data.decode('utf-8')
        
        # Execute the injected code in the main thread's globals
        import __main__
        exec(code_to_execute, __main__.__dict__)
        
        sock.close()
        
    except Exception as e:
        print(f"Socket injection error: {{e}}")

# Start socket handler in background thread  
threading.Thread(target=socket_injection_handler, daemon=True).start()
"""
        return injection_script
    
    def inject_via_modification(self):
        """Inject code by modifying the running script temporarily"""
        training_script = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts\NEBULA_Official_Training.py"
        
        try:
            # Read current script
            with open(training_script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if injection already exists
            if 'EMERGENCY_INJECTION_CODE' in content:
                logger.info("Injection code already present")
                return True
            
            # Create injection point
            injection_code = f'''
# EMERGENCY_INJECTION_CODE - AUTO GENERATED
import threading
import time

def emergency_injection_check():
    """Check for emergency injection trigger"""
    injection_file = r"D:\\NEBULA_DIVISION_A\\models\\EXECUTE_INJECTION.trigger"
    
    while True:
        try:
            if os.path.exists(injection_file):
                logger.info("🚨 EMERGENCY INJECTION TRIGGERED!")
                
                # Read and execute injection code
                with open(injection_file, 'r') as f:
                    code_to_execute = f.read()
                
                # Execute in current globals
                exec(code_to_execute, globals())
                
                # Remove trigger file
                os.remove(injection_file)
                logger.info("✓ Injection completed and trigger removed")
                
            time.sleep(1)  # Check every second
            
        except Exception as e:
            logger.error(f"Injection check error: {{e}}")
            time.sleep(5)

# Start injection checker in background thread
threading.Thread(target=emergency_injection_check, daemon=True).start()
logger.info("Emergency injection checker started")
# END EMERGENCY_INJECTION_CODE
'''
            
            # Find insertion point (after imports, before class)
            insert_pos = content.find('class NEBULATrainingManager')
            if insert_pos == -1:
                logger.error("Could not find insertion point")
                return False
            
            # Insert injection code
            modified_content = content[:insert_pos] + injection_code + content[insert_pos:]
            
            # Write modified script
            with open(training_script, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            logger.info("✓ Emergency injection code added to training script")
            return True
            
        except Exception as e:
            logger.error(f"Failed to inject code: {e}")
            return False
    
    def trigger_emergency_save(self):
        """Trigger the emergency save"""
        payload = self.create_injection_payload()
        trigger_file = r"D:\NEBULA_DIVISION_A\models\EXECUTE_INJECTION.trigger"
        
        try:
            with open(trigger_file, 'w', encoding='utf-8') as f:
                f.write(payload)
            
            logger.info(f"✓ Injection trigger created: {trigger_file}")
            
            # Wait for execution
            logger.info("Waiting for injection execution...")
            max_wait = 30
            for i in range(max_wait):
                if not os.path.exists(trigger_file):
                    logger.info("✓ Trigger file consumed - injection executed!")
                    return True
                time.sleep(1)
                logger.info(f"Waiting... {i+1}/{max_wait}s")
            
            logger.warning("Timeout waiting for injection execution")
            return False
            
        except Exception as e:
            logger.error(f"Failed to create trigger: {e}")
            return False
    
    def execute_emergency_injection(self):
        """Execute the complete emergency injection process"""
        logger.info("=== WINDOWS PYTORCH EMERGENCY INJECTION ===")
        
        # Step 1: Find training process
        if not self.find_training_process():
            return False
        
        # Step 2: Inject monitoring code into training script
        logger.info("Step 1: Injecting emergency monitoring code...")
        if not self.inject_via_modification():
            return False
        
        # Step 3: Wait a moment for injection to take effect
        logger.info("Waiting 5 seconds for injection to initialize...")
        time.sleep(5)
        
        # Step 4: Trigger emergency save
        logger.info("Step 2: Triggering emergency save...")
        if not self.trigger_emergency_save():
            return False
        
        # Step 5: Check results
        logger.info("Step 3: Checking results...")
        checkpoint_file = r"D:\NEBULA_DIVISION_A\models\emergency_injected_checkpoint.pth"
        
        if os.path.exists(checkpoint_file):
            size = os.path.getsize(checkpoint_file) / (1024 * 1024)
            logger.info(f"✅ SUCCESS! Emergency checkpoint created: {size:.1f} MB")
            logger.info(f"File: {checkpoint_file}")
            return True
        else:
            logger.error("❌ FAILED: Emergency checkpoint not found")
            return False

if __name__ == "__main__":
    injector = WindowsPyTorchInjector()
    success = injector.execute_emergency_injection()
    
    if success:
        logger.info("🎉 Emergency PyTorch injection completed successfully!")
        logger.info("This technique can now be used for future training emergencies!")
    else:
        logger.error("💥 Emergency injection failed")
        logger.info("Check logs above for specific error details")