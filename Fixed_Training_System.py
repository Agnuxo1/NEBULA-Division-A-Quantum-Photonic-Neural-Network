#!/usr/bin/env python3
"""
NEBULA Fixed Training System
===========================
Corrected training system with:
1. Forced checkpoint loading from epoch 28
2. 15-minute auto-save system
3. Robust checkpoint validation
"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta
import logging

# Add NEBULA path
sys.path.append(r'D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\NEBULA_PROJECT\scripts')
from NEBULA_Official_Training import NEBULATrainingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedNEBULATraining(NEBULATrainingManager):
    """Fixed NEBULA Training with guaranteed checkpoint loading and auto-save"""
    
    def __init__(self, dataset_path, model_save_dir='nebula_models'):
        super().__init__(dataset_path, model_save_dir)
        
        # Auto-save system
        self.auto_save_interval = 15 * 60  # 15 minutes in seconds
        self.last_auto_save = time.time()
        
        logger.info("🔧 FIXED NEBULA Training initialized with auto-save every 15 minutes")
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint available (autosave or official)"""
        logger.info("🔍 SEARCHING FOR LATEST CHECKPOINT...")
        
        # Look for auto-saves first (most recent)
        import glob
        autosave_files = glob.glob(os.path.join(self.model_save_dir, 'nebula_autosave_*.pth'))
        official_files = glob.glob(os.path.join(self.model_save_dir, 'nebula_official_epoch_*.pth'))
        
        all_checkpoints = autosave_files + official_files
        
        if not all_checkpoints:
            logger.error("❌ No checkpoints found!")
            return False
        
        # Sort by modification time (most recent first)
        all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        latest_checkpoint = all_checkpoints[0]
        checkpoint_type = "AUTO-SAVE" if "autosave" in latest_checkpoint else "OFFICIAL"
        
        logger.info(f"✅ FOUND LATEST {checkpoint_type} CHECKPOINT:")
        logger.info(f"   📁 File: {os.path.basename(latest_checkpoint)}")
        
        return self._load_checkpoint_file(latest_checkpoint)
    
    def load_fallback_checkpoint(self, target_epoch=28):
        """Load fallback checkpoint only when no recent checkpoints exist"""
        logger.info(f"🆘 FALLBACK: Loading emergency checkpoint from epoch {target_epoch}")
        logger.info("⚠️ This should only happen if no recent checkpoints were found")
        
        checkpoint_file = os.path.join(self.model_save_dir, f'nebula_official_epoch_{target_epoch:04d}.pth')
        
        if not os.path.exists(checkpoint_file):
            logger.error(f"❌ Fallback checkpoint not found: {checkpoint_file}")
            return False
        
        logger.warning("🚨 Using fallback - training will resume from older state")
        return self._load_checkpoint_file(checkpoint_file)
    
    def _load_checkpoint_file(self, checkpoint_file):
        """Internal method to load any checkpoint file"""
        logger.info(f"📂 LOADING CHECKPOINT: {os.path.basename(checkpoint_file)}")
        
        try:
            import torch
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_auc = checkpoint.get('best_auc', 0.0)
            self.training_history = checkpoint.get('training_history', [])
            
            # Store optimizer and scheduler states
            self._checkpoint_optimizer_state = checkpoint.get('optimizer_state_dict', None)
            self._checkpoint_scheduler_state = checkpoint.get('scheduler_state_dict', None)
            
            logger.info(f"✅ CHECKPOINT LOADED SUCCESSFULLY:")
            logger.info(f"   📊 Epoch: {self.current_epoch}")
            logger.info(f"   🏆 Best AUC: {self.best_auc:.4f}")
            logger.info(f"   📈 Training history: {len(self.training_history)} epochs")
            logger.info(f"   ⏰ Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
            
            # Set continuation epoch (will train from current_epoch + 1)
            logger.info(f"🚀 Will resume from epoch {self.current_epoch + 1}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CHECKPOINT LOAD FAILED: {e}")
            return False
    
    def auto_save_checkpoint(self, force=False):
        """Auto-save checkpoint every 15 minutes"""
        current_time = time.time()
        
        if force or (current_time - self.last_auto_save) >= self.auto_save_interval:
            logger.info("💾 AUTO-SAVE: Saving checkpoint...")
            
            # Create auto-save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_path = os.path.join(
                self.model_save_dir, 
                f'nebula_autosave_{timestamp}_epoch_{self.current_epoch:04d}.pth'
            )
            
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                'best_auc': self.best_auc,
                'config': self.config,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat(),
                'auto_save': True
            }
            
            try:
                import torch
                torch.save(checkpoint, auto_save_path)
                self.last_auto_save = current_time
                
                # Get file size
                size_mb = os.path.getsize(auto_save_path) / (1024 * 1024)
                logger.info(f"✅ AUTO-SAVE COMPLETE: {size_mb:.1f} MB")
                logger.info(f"   📁 File: {os.path.basename(auto_save_path)}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ AUTO-SAVE FAILED: {e}")
                return False
        
        return True  # No save needed yet
    
    def setup_auto_save_timer(self):
        """Setup background auto-save timer"""
        def auto_save_worker():
            while True:
                try:
                    time.sleep(60)  # Check every minute
                    self.auto_save_checkpoint()
                except Exception as e:
                    logger.error(f"Auto-save worker error: {e}")
        
        # Start auto-save thread
        auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        auto_save_thread.start()
        logger.info("⏱️ Auto-save timer started (15-minute intervals)")
    
    def setup_model(self):
        """Setup model with forced checkpoint loading"""
        logger.info("🔧 Setting up NEBULA model with forced checkpoint loading...")
        
        # Initialize NEBULA
        from NEBULA_Professional_System import NEBULAMedicalAI
        self.model = NEBULAMedicalAI()
        
        # Load calibration
        try:
            self.model.load_calibration('nebula_stepped_calibration_complete.json')
            logger.info("NEBULA calibrated vision loaded")
        except Exception as e:
            logger.warning(f"Calibration loading failed: {e}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # LOAD LATEST AVAILABLE CHECKPOINT (auto-save or official)
        success = self.load_latest_checkpoint()
        if not success:
            logger.warning("⚠️ No recent checkpoints found, trying fallback")
            success = self.load_fallback_checkpoint(28)
            if not success:
                logger.error("❌ Failed to load any checkpoint - ABORTING")
                raise RuntimeError("Critical: Could not load any checkpoint")
        
        # Setup optimizer and scheduler
        import torch.optim as optim
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup loss function
        import torch.nn as nn
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Load optimizer and scheduler states
        self.load_optimizer_states()
        
        # Mixed precision scaler
        if self.config['mixed_precision']:
            import torch.cuda.amp
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"🎯 FIXED MODEL SETUP COMPLETE")
        logger.info(f"   🏁 Starting from epoch {self.current_epoch + 1}")
        logger.info(f"   🏆 Target: Beat AUC {self.best_auc:.4f}")
        logger.info(f"   🎯 Goal: AUC > 0.936819 for first place")
    
    def train_epoch(self):
        """Enhanced train epoch with auto-save"""
        # Call parent training method
        train_loss = super().train_epoch()
        
        # Auto-save check during training
        self.auto_save_checkpoint()
        
        return train_loss
    
    def save_checkpoint(self, is_best=False):
        """Enhanced save checkpoint with better logging"""
        logger.info(f"💾 SAVING CHECKPOINT: Epoch {self.current_epoch + 1}")
        
        # Call parent save method
        super().save_checkpoint(is_best)
        
        if is_best:
            logger.info(f"🏆 NEW BEST MODEL SAVED! AUC: {self.best_auc:.4f}")
        
        # Also trigger auto-save
        self.auto_save_checkpoint(force=True)
    
    def train_fixed_continuous(self):
        """Fixed continuous training with auto-save and forced checkpoint loading"""
        logger.info("=" * 80)
        logger.info("🔧 NEBULA FIXED TRAINING SYSTEM")
        logger.info("Francisco Angulo de Lafuente and NEBULA Team")
        logger.info("=" * 80)
        logger.info("🎯 MISSION: Resume from epoch 28 (AUC 0.8696) and reach first place")
        logger.info("🏆 TARGET: AUC > 0.936819")
        logger.info("💾 AUTO-SAVE: Every 15 minutes")
        
        # Setup datasets and model (with forced checkpoint loading)
        self.setup_datasets()
        self.setup_model()  # This will force load epoch 28
        
        # Start auto-save system
        self.setup_auto_save_timer()
        
        # Immediate auto-save to confirm system works
        logger.info("🧪 Testing auto-save system...")
        self.auto_save_checkpoint(force=True)
        
        self.training_start_time = datetime.now()
        
        try:
            start_epoch = self.current_epoch
            target_auc = 0.936819
            
            logger.info(f"🚀 RESUMING TRAINING FROM EPOCH {start_epoch + 1}")
            logger.info(f"📊 Current best AUC: {self.best_auc:.4f}")
            logger.info(f"📈 Need improvement: {target_auc - self.best_auc:.4f} points")
            
            for epoch in range(start_epoch, self.config['max_epochs']):
                self.current_epoch = epoch
                
                # Training
                logger.info(f"\\n🏃 Epoch {epoch + 1}/{self.config['max_epochs']}")
                train_loss = self.train_epoch()
                
                # Validation
                if epoch % self.config['validate_every'] == 0:
                    val_loss, val_auc, pathology_aucs = self.validate_epoch()
                    
                    # Learning rate scheduling
                    self.scheduler.step(val_auc)
                    
                    # Check if best model
                    is_best = val_auc > self.best_auc
                    if is_best:
                        improvement = val_auc - self.best_auc
                        self.best_auc = val_auc
                        self.patience_counter = 0
                        logger.info(f"🎉 NEW BEST AUC! Improved by {improvement:.4f} points")
                        
                        # Check if we reached first place
                        if val_auc > target_auc:
                            logger.info("🏆🏆🏆 FIRST PLACE ACHIEVED! 🏆🏆🏆")
                            logger.info(f"🎯 Target: {target_auc:.6f}")
                            logger.info(f"✅ Actual: {val_auc:.6f}")
                            logger.info(f"🚀 Margin: +{val_auc - target_auc:.6f}")
                    else:
                        self.patience_counter += 1
                    
                    # Enhanced logging
                    elapsed_time = datetime.now() - self.training_start_time
                    progress_to_target = ((val_auc - 0.8696) / (target_auc - 0.8696)) * 100
                    
                    logger.info(f"📊 Epoch {epoch + 1} Results:")
                    logger.info(f"   🏃 Train Loss: {train_loss:.4f}")
                    logger.info(f"   ✅ Val Loss: {val_loss:.4f}")
                    logger.info(f"   🏆 Val AUC: {val_auc:.6f} {'🎉 (BEST!)' if is_best else ''}")
                    logger.info(f"   📈 Best AUC: {self.best_auc:.6f}")
                    logger.info(f"   🎯 Progress to 1st: {progress_to_target:.1f}%")
                    logger.info(f"   ⏳ Patience: {self.patience_counter}/{self.config['patience']}")
                    logger.info(f"   🕐 Elapsed: {elapsed_time}")
                    logger.info(f"   📚 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                    
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
                    
                    # Save checkpoint (with enhanced logic)
                    if epoch % self.config['save_every'] == 0 or is_best:
                        self.save_checkpoint(is_best)
                    
                    # Early stopping check
                    if (epoch >= self.config['min_epochs'] and 
                        self.patience_counter >= self.config['patience']):
                        logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                        logger.info(f"🏆 Final best AUC: {self.best_auc:.6f}")
                        if self.best_auc > target_auc:
                            logger.info("🎉 MISSION ACCOMPLISHED: First place achieved!")
                        else:
                            needed = target_auc - self.best_auc
                            logger.info(f"📈 Still need {needed:.4f} points for first place")
                        break
                
                # Memory cleanup
                import torch
                torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            self.auto_save_checkpoint(force=True)
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            self.auto_save_checkpoint(force=True)
            raise
        finally:
            # Final save
            logger.info("💾 Final checkpoint save...")
            self.save_checkpoint()
            
            total_time = datetime.now() - self.training_start_time
            logger.info("=" * 80)
            logger.info("🏁 NEBULA FIXED TRAINING COMPLETED")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total training time: {total_time}")
            logger.info(f"📊 Final epoch: {self.current_epoch + 1}")
            logger.info(f"🏆 Best validation AUC: {self.best_auc:.6f}")
            logger.info(f"📁 Model saved in: {self.model_save_dir}")
            
            if self.best_auc > 0.936819:
                logger.info("🎉🏆 CONGRATULATIONS: FIRST PLACE ACHIEVED! 🏆🎉")
            else:
                needed = 0.936819 - self.best_auc
                logger.info(f"📈 Progress: Need {needed:.4f} more points for first place")

def main():
    """Main function with fixed training system"""
    logger.info("🔧 Starting NEBULA Fixed Training System...")
    
    # Configuration
    dataset_path = r"D:\\NEBULA_DIVISION_A\\datasets\\grand-xray-slam-division-a"
    model_save_dir = r"D:\\NEBULA_DIVISION_A\\models"
    
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Create fixed training manager
    trainer = FixedNEBULATraining(dataset_path, model_save_dir)
    
    # Start fixed continuous training
    trainer.train_fixed_continuous()

if __name__ == "__main__":
    main()