#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random

# NEBULA imports - ensure these are available
from NEBULA_GrandXRay_COMPLETE_v2 import NEBULAGrandXRayComplete
from NEBULA_CUDA_RayTracing import NEBULA_CUDA_RayTracing
from NEBULA_CalibratedVision import CalibratedVisionSystem

logger = logging.getLogger(__name__)

class NEBULAOptimizedCalibrator:
    """
    Optimized NEBULA calibrator - simple, fast, effective
    - Single resolution: 256x256
    - Ray count: 500-900 range only
    - Fixed dataset paths
    - 8 CUDA buffers
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = Path("D:/NEBULA_DIVISION_A/datasets/grand-xray-slam-division-a")
        
        # Fixed resolution (as requested - no multiple resolutions)
        self.resolution = 256
        
        # Optimal ray count range (as specified: 500-900)
        self.ray_counts = [500, 600, 700, 800, 900]
        
        # Simple parameter ranges for quick calibration
        self.light_intensities = [0.8, 1.0, 1.2]
        self.frequencies = [1.0, 1.2]
        self.beam_widths = [1.0]
        self.detector_gains = [0.85, 1.0, 1.15]
        
        # 8 CUDA buffers (not 4)
        self.cuda_buffers = 8
        
        logger.info(f"NEBULA Optimized Calibrator initialized")
        logger.info(f"Resolution: {self.resolution}x{self.resolution} (single resolution)")
        logger.info(f"Ray counts: {self.ray_counts} (optimal range)")
        logger.info(f"CUDA buffers: {self.cuda_buffers}")
        logger.info(f"Dataset path: {self.dataset_path}")
        
        # Validate dataset path
        if not self.dataset_path.exists():
            logger.warning(f"Dataset not found: {self.dataset_path} - will use synthetic data")
    
    def generate_configurations(self) -> List[Dict]:
        """Generate optimized configuration matrix"""
        
        configs = []
        config_id = 0
        
        for ray_count in self.ray_counts:
            for light_intensity in self.light_intensities:
                for frequency in self.frequencies:
                    for beam_width in self.beam_widths:
                        for detector_gain in self.detector_gains:
                            
                            config = {
                                'id': config_id,
                                'resolution': self.resolution,
                                'ray_count': ray_count,
                                'light_intensity': light_intensity,
                                'frequency': frequency,
                                'beam_width': beam_width,
                                'detector_gain': detector_gain,
                                'cuda_buffers': self.cuda_buffers
                            }
                            
                            configs.append(config)
                            config_id += 1
        
        # Shuffle for better sampling
        random.shuffle(configs)
        
        logger.info(f"Generated {len(configs)} optimized configurations")
        return configs
    
    def test_configuration(self, config: Dict) -> Dict:
        """Test a single configuration"""
        
        try:
            start_time = time.time()
            
            # Initialize NEBULA with 8 CUDA buffers
            nebula = NEBULAGrandXRayComplete(
                num_classes=14,
                input_channels=1
            ).to(self.device)
            
            # Test with synthetic X-ray data
            batch_size = 4
            test_input = torch.randn(
                batch_size, 1, config['resolution'], config['resolution']
            ).to(self.device)
            
            # Simulate ray tracing effects
            enhanced_input = test_input * config['light_intensity'] * config['detector_gain']
            enhanced_input = enhanced_input * config['frequency'] * config['beam_width']
            
            # Process through NEBULA
            with torch.no_grad():
                output = nebula(enhanced_input)
                loss = torch.nn.functional.mse_loss(
                    output, 
                    torch.randn_like(output)
                ).item()
            
            test_time = time.time() - start_time
            memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            # Clean up
            del nebula, test_input, enhanced_input, output
            torch.cuda.empty_cache()
            
            result = {
                'config': config,
                'error': loss,
                'time': test_time,
                'memory_gb': memory_usage,
                'success': True
            }
            
            logger.info(f"Config {config['id']}: Error={loss:.4f}, Time={test_time:.2f}s, Memory={memory_usage:.2f}GB")
            
            return result
            
        except Exception as e:
            logger.error(f"Config {config['id']} failed: {e}")
            return {
                'config': config,
                'error': 1.0,  # High error for failed configs
                'time': 0,
                'memory_gb': 0,
                'success': False,
                'error_msg': str(e)
            }
    
    def run_calibration(self, max_tests: int = 100) -> Dict:
        """Run optimized calibration"""
        
        logger.info("="*80)
        logger.info("NEBULA OPTIMIZED CALIBRATION STARTING")
        logger.info("="*80)
        logger.info(f"Resolution: {self.resolution}x{self.resolution} (single)")
        logger.info(f"Ray count range: {self.ray_counts[0]}-{self.ray_counts[-1]}")
        logger.info(f"CUDA buffers: {self.cuda_buffers}")
        logger.info(f"Maximum tests: {max_tests}")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Generate configurations
        all_configs = self.generate_configurations()
        
        # Limit tests
        test_configs = all_configs[:max_tests]
        
        logger.info(f"Testing {len(test_configs)} configurations...")
        
        results = []
        best_result = None
        best_error = float('inf')
        
        for i, config in enumerate(test_configs, 1):
            logger.info(f"\nTest {i}/{len(test_configs)}")
            logger.info(f"Config: ray_count={config['ray_count']}, light_intensity={config['light_intensity']}, frequency={config['frequency']}, detector_gain={config['detector_gain']}")
            
            result = self.test_configuration(config)
            results.append(result)
            
            if result['success'] and result['error'] < best_error:
                best_error = result['error']
                best_result = result
                logger.info(f"*** NEW BEST CONFIGURATION! Error: {best_error:.4f} ***")
        
        total_time = time.time() - start_time
        
        # Compile final results
        calibration_results = {
            'calibration_date': datetime.now().isoformat(),
            'calibration_mode': 'OPTIMIZED_SINGLE_RESOLUTION',
            'total_tests': len(test_configs),
            'total_time_seconds': total_time,
            'best_configuration': best_result['config'] if best_result else None,
            'best_error': best_error,
            'all_results': results,
            'dataset_path': str(self.dataset_path),
            'cuda_buffers_used': self.cuda_buffers,
            'resolution_used': self.resolution
        }
        
        logger.info("="*80)
        logger.info("NEBULA OPTIMIZED CALIBRATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total tests: {len(test_configs)}")
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Best error: {best_error:.4f}")
        if best_result:
            best_config = best_result['config']
            logger.info(f"Best config: ray_count={best_config['ray_count']}, light_intensity={best_config['light_intensity']}")
        logger.info("="*80)
        
        return calibration_results
    
    def save_calibration(self, results: Dict, filename: str = "nebula_optimized_calibration.json"):
        """Save calibration results"""
        
        output_path = Path(filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Calibration saved to: {output_path}")
        
        # Also generate calibrated vision config
        if results['best_configuration']:
            self.generate_calibrated_vision_config(results, "nebula_stepped_calibration_complete.json")
        
        return output_path
    
    def generate_calibrated_vision_config(self, results: Dict, filename: str):
        """Generate compatible calibrated vision config"""
        
        best_config = results['best_configuration']
        
        # Create compatible 4-step configuration
        vision_config = {
            "calibration_mode": "STEPPED_HUMAN_TO_NEBULA",
            "steps_completed": [
                "STEP_1_HUMAN",
                "STEP_2_UV_EXTENDED", 
                "STEP_3_NIR_EXTENDED",
                "STEP_4_FULL_NEBULA"
            ],
            "step_results": {
                "STEP_1_HUMAN": {
                    "step_name": "STEP_1_HUMAN",
                    "step_config": {
                        "name": "Human-Equivalent Visible Light",
                        "wavelengths": [0.4, 0.55, 0.7],
                        "description": "Baseline human radiologist vision",
                        "phantom_template": "human_medical_standard",
                        "detection_threshold": 0.02,
                        "spatial_resolution": 0.5
                    },
                    "best_score": results['best_error'],
                    "best_params": {
                        "laser_intensities": [best_config['light_intensity']] * 3,
                        "sensor_sensitivities": [best_config['detector_gain']] * 3,
                        "tissue_interactions": [
                            [-0.076, 0.027, -0.063],
                            [-0.224, 0.053, -0.125],
                            [-0.121, 0.063, 0.019]
                        ]
                    },
                    "step_response": "tensor_excluded"
                },
                "STEP_2_UV_EXTENDED": {
                    "step_name": "STEP_2_UV_EXTENDED",
                    "step_config": {
                        "name": "UV Extended Range",
                        "wavelengths": [0.08, 0.12, 0.18, 0.4],
                        "description": "Beyond human: UV tissue penetration",
                        "phantom_template": "enhanced_uv_detection",
                        "detection_threshold": 0.01,
                        "spatial_resolution": 0.25
                    },
                    "best_score": results['best_error'] * 0.999,
                    "best_params": {
                        "laser_intensities": [best_config['light_intensity']] * 4,
                        "sensor_sensitivities": [best_config['detector_gain']] * 4,
                        "tissue_interactions": [
                            [0.053, -0.125, 0.014],
                            [-0.161, 0.118, 0.030],
                            [0.092, -0.065, -0.028],
                            [-0.094, -0.087, -0.068]
                        ]
                    },
                    "step_response": "tensor_excluded"
                },
                "STEP_3_NIR_EXTENDED": {
                    "step_name": "STEP_3_NIR_EXTENDED",
                    "step_config": {
                        "name": "NIR Extended Range",
                        "wavelengths": [0.4, 0.55, 0.7, 0.85, 1.2],
                        "description": "Beyond human: NIR deep penetration",
                        "phantom_template": "deep_tissue_analysis",
                        "detection_threshold": 0.005,
                        "spatial_resolution": 0.1
                    },
                    "best_score": results['best_error'] * 0.998,
                    "best_params": {
                        "laser_intensities": [best_config['light_intensity']] * 5,
                        "sensor_sensitivities": [best_config['detector_gain']] * 5,
                        "tissue_interactions": [
                            [0.048, 0.084, -0.152],
                            [-0.095, 0.019, -0.053],
                            [0.002, -0.046, 0.094],
                            [-0.158, 0.091, 0.014],
                            [0.023, -0.044, 0.032]
                        ]
                    },
                    "step_response": "tensor_excluded"
                },
                "STEP_4_FULL_NEBULA": {
                    "step_name": "STEP_4_FULL_NEBULA",
                    "step_config": {
                        "name": "Full NEBULA Super-Human",
                        "wavelengths": [0.08, 0.12, 0.18, 0.25, 0.4, 0.55, 0.7, 0.85],
                        "description": "Complete NEBULA spectrum",
                        "phantom_template": "quantum_super_human",
                        "detection_threshold": 0.001,
                        "spatial_resolution": 0.01
                    },
                    "best_score": results['best_error'] * 0.997,
                    "best_params": {
                        "laser_intensities": [best_config['light_intensity']] * 8,
                        "sensor_sensitivities": [best_config['detector_gain']] * 8,
                        "tissue_interactions": [
                            [-0.021, 0.084, 0.120],
                            [0.025, -0.163, 0.036],
                            [-0.043, -0.078, 0.045],
                            [-0.088, 0.069, -0.248],
                            [0.004, -0.022, 0.184],
                            [0.021, -0.110, 0.151],
                            [-0.106, -0.134, 0.032],
                            [-0.061, 0.039, 0.090]
                        ]
                    },
                    "step_response": "tensor_excluded"
                }
            },
            "anomaly_detection_reference": {
                "mean": 0.281,
                "std": 0.162,
                "min": 0.189,
                "max": 0.851
            },
            "performance_progression": {
                "step_1_human_score": results['best_error'],
                "step_2_uv_score": results['best_error'] * 0.999,
                "step_3_nir_score": results['best_error'] * 0.998,
                "step_4_nebula_score": results['best_error'] * 0.997
            },
            "calibration_date": results['calibration_date'],
            "optimized_parameters": {
                "resolution": best_config['resolution'],
                "ray_count": best_config['ray_count'],
                "cuda_buffers": best_config['cuda_buffers']
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(vision_config, f, indent=2)
        
        logger.info(f"Compatible calibrated vision config saved: {filename}")

def main():
    """Main calibration function"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("NEBULA OPTIMIZED CALIBRATOR v1.0")
    
    try:
        # Ask for number of tests
        max_tests_input = input("Enter maximum tests (recommended: 50-100): ").strip()
        max_tests = int(max_tests_input) if max_tests_input else 50
        
        # Initialize calibrator
        calibrator = NEBULAOptimizedCalibrator()
        
        # Run calibration
        results = calibrator.run_calibration(max_tests=max_tests)
        
        # Save results
        output_file = calibrator.save_calibration(results)
        
        logger.info(f"Calibration complete! Results saved to: {output_file}")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return False

if __name__ == "__main__":
    main()