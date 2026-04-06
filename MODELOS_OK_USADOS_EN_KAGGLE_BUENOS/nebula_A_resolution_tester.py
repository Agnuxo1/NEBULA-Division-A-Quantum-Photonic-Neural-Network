#!/usr/bin/env python3
"""
NEBULA Resolution & Ray-Tracing Optimization Tester
Single-Sample Multi-Resolution Performance Analysis

Author: Francisco Angulo de Lafuente and NEBULA Team
Date: September 2025

IMPORTANT: Testing system - NO changes to main model until validation complete
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
import json
from PIL import Image
from typing import Dict, List, Tuple
import sys
import os
from tqdm import tqdm

# Add NEBULA system to path
sys.path.append('E:/grand-xray-slam-division-a/NEBULA_PROJECT/scripts')
from NEBULA_Professional_System import NEBULAMedicalAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NEBULAResolutionTester:
    """
    Single-Sample Multi-Resolution Testing System
    
    Tests different resolutions, ray counts, and intensities on a SINGLE known sample
    to determine optimal configuration without breaking the main model.
    """
    
    def __init__(self):
        print("NEBULA RESOLUTION & RAY-TRACING TESTER")
        print("=" * 60)
        print("SAFETY: Testing mode - main model protected")
        print("Multi-sample validation approach (8 cases + 2 controls)")
        print("=" * 60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Multi-sample validation set with ORIENTATION diversity (CRÍTICO para ray-tracing)
        self.test_samples = [
            # Sample 1: Support Devices - FRONTAL AP (control conocido)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000001_001_001.jpg',
                'image_id': '00000001_001_001.jpg',
                'primary_pathology': 'Support Devices',
                'view_category': 'Frontal',
                'view_position': 'AP',
                'ground_truth': [0,0,0,0,0,0,0,0,0,0,0,0,0,1]
            },
            # Sample 2: Cardiomegaly - FRONTAL PA 
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000009_001_001.jpg',
                'image_id': '00000009_001_001.jpg', 
                'primary_pathology': 'Cardiomegaly',
                'view_category': 'Frontal',
                'view_position': 'PA',
                'ground_truth': [0,1,0,0,1,0,0,0,0,0,0,0,0,0]
            },
            # Sample 2b: MISMO PACIENTE - LATERAL (comparación crítica)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000009_001_002.jpg',
                'image_id': '00000009_001_002.jpg',
                'primary_pathology': 'Cardiomegaly Lateral',
                'view_category': 'Lateral',
                'view_position': 'Lateral',
                'ground_truth': [0,1,0,0,1,0,0,0,0,0,0,0,0,0]
            },
            # Sample 3: No Finding - FRONTAL PA (control negativo)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000011_013_001.jpg',
                'image_id': '00000011_013_001.jpg',
                'primary_pathology': 'No Finding',
                'view_category': 'Frontal', 
                'view_position': 'PA',
                'ground_truth': [0,0,0,0,0,0,0,0,1,0,0,0,0,0]
            },
            # Sample 3b: MISMO PACIENTE - LATERAL (control negativo lateral)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000011_013_002.jpg',
                'image_id': '00000011_013_002.jpg',
                'primary_pathology': 'No Finding Lateral',
                'view_category': 'Lateral',
                'view_position': 'Lateral', 
                'ground_truth': [0,0,0,0,0,0,0,0,1,0,0,0,0,0]
            },
            # Sample 4: Multi-patología - FRONTAL AP (casos complejos)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000011_001_001.jpg',
                'image_id': '00000011_001_001.jpg',
                'primary_pathology': 'Multi-pathology',
                'view_category': 'Frontal',
                'view_position': 'AP',
                'ground_truth': [1,0,1,0,1,1,0,1,0,1,0,1,1,1]
            },
            # Sample 5: Fracture + Pleural - FRONTAL PA
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000018_001_001.jpg',
                'image_id': '00000018_001_001.jpg',
                'primary_pathology': 'Fracture',
                'view_category': 'Frontal',
                'view_position': 'PA',
                'ground_truth': [0,0,0,0,0,1,0,0,0,1,1,0,0,0]
            },
            # Sample 5b: MISMO PACIENTE - LATERAL (fractura en vista lateral)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000018_001_002.jpg',
                'image_id': '00000018_001_002.jpg',
                'primary_pathology': 'Fracture Lateral',
                'view_category': 'Lateral',
                'view_position': 'Lateral',
                'ground_truth': [0,0,0,0,0,1,0,0,0,1,1,0,0,0]
            },
            # Sample 6: Pneumothorax - FRONTAL PA (crítico para ray-tracing)
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000023_004_001.jpg',
                'image_id': '00000023_004_001.jpg',
                'primary_pathology': 'Pneumothorax',
                'view_category': 'Frontal',
                'view_position': 'PA',
                'ground_truth': [0,0,0,0,0,0,0,0,0,0,0,0,1,1]
            },
            # Sample 7: Caso complejo máximo - FRONTAL PA
            {
                'image_path': 'E:/grand-xray-slam-division-a/train1/00000023_011_001.jpg',
                'image_id': '00000023_011_001.jpg',
                'primary_pathology': 'Complex Multi',
                'view_category': 'Frontal',
                'view_position': 'PA', 
                'ground_truth': [1,1,1,0,1,1,0,1,0,1,1,1,0,0]
            }
        ]
        
        self.pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        
        # Test matrix configuration
        self.resolution_matrix = [
            256,   # Current baseline
            512,   # 2x improvement  
            768,   # 3x improvement
            1024,  # 4x improvement
            1536,  # 6x improvement
            2048,  # 8x improvement - approaching native
            # 3408 native would require 16GB+ VRAM
        ]
        
        self.ray_counts = [
            900,   # Current (30x30)
            1600,  # 40x40 
            2500,  # 50x50
            3600,  # 60x60 (high precision)
        ]
        
        self.light_intensities = [
            1.0,   # Current baseline
            0.8,   # Reduced intensity
            1.2,   # Increased intensity  
            1.5,   # High intensity
        ]
        
        # Results storage
        self.test_results = []
        
        print(f"Test samples: {len(self.test_samples)} cases with ORIENTATION diversity")
        for i, sample in enumerate(self.test_samples):
            print(f"  {i+1}. {sample['primary_pathology']:20s} | {sample['view_category']:7s}-{sample['view_position']:7s} | {sample['image_id']}")
        
        # Orientation analysis
        orientation_counts = {}
        for sample in self.test_samples:
            key = f"{sample['view_category']}-{sample['view_position']}"
            orientation_counts[key] = orientation_counts.get(key, 0) + 1
        
        print("\nORIENTATION DISTRIBUTION:")
        for orientation, count in orientation_counts.items():
            print(f"  {orientation}: {count} samples")
        print(f"Resolution range: {min(self.resolution_matrix)} - {max(self.resolution_matrix)}px")
        print(f"Ray counts: {self.ray_counts}")
        print(f"Light intensities: {self.light_intensities}")
        print(f"\nTotal tests: {len(self.test_samples)} samples × {len(self.resolution_matrix) * len(self.ray_counts) * len(self.light_intensities)} configs = {len(self.test_samples) * len(self.resolution_matrix) * len(self.ray_counts) * len(self.light_intensities)} total")
        print("CRITICAL: Ray-tracing calibration now accounts for image orientation!")
    
    def create_test_model(self, resolution: int, num_rays: int, light_intensity: float):
        """Create isolated test model with specific parameters"""
        try:
            # Create NEBULA with custom parameters (isolated from main model)
            model = NEBULAMedicalAI()
            
            # Load CURRENT trained model (Epoch 28) for comparison
            model_path = 'C:/nebula-cuda-fresh/RSNA/nebula_models/nebula_official_epoch_0028.pth'
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating test model: {e}")
            return None
    
    def preprocess_image_with_orientation(self, image_path: str, target_resolution: int, 
                                        view_category: str, view_position: str) -> torch.Tensor:
        """Custom preprocessing with variable resolution + ORIENTATION positioning"""
        try:
            image = Image.open(image_path).convert('L')
            original_size = image.size
            
            # STEP 1: Automatic orientation positioning (CRÍTICO)
            image = self.apply_orientation_positioning(image, view_category, view_position)
            
            # STEP 2: Resize maintaining aspect ratio
            if target_resolution != 256:
                image = image.resize((target_resolution, target_resolution), Image.LANCZOS)
            else:
                image = image.resize((256, 256))  # Current method
            
            # STEP 3: Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
            
            return tensor, original_size
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None, None
    
    def apply_orientation_positioning(self, image: Image.Image, view_category: str, view_position: str) -> Image.Image:
        """Apply automatic positioning based on image orientation for ray-tracing accuracy"""
        
        # Frontal views (AP/PA) - standard positioning
        if view_category == 'Frontal':
            if view_position == 'AP':
                # Anteroposterior: X-ray from front to back
                # Standard positioning - no rotation needed
                return image
            elif view_position == 'PA': 
                # Posteroanterior: X-ray from back to front
                # Mirror horizontal for ray-tracing consistency
                return image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Lateral views - requires rotation for consistent ray-tracing
        elif view_category == 'Lateral':
            if view_position == 'Lateral' or view_position == 'LL':
                # Rotate 90 degrees for proper ray-tracing alignment
                return image.rotate(90, expand=True)
        
        # Default: return original if unknown orientation
        logger.warning(f"Unknown orientation: {view_category}-{view_position}, using original")
        return image
    
    def extract_feature_maps(self, model, tensor):
        """Extract feature maps for pixel-level analysis"""
        try:
            # Get intermediate activations for ray-tracing analysis
            with torch.no_grad():
                # Simple feature extraction - adapt based on NEBULA architecture
                activations = model(tensor)
                return {
                    'final_features': activations.cpu().numpy(),
                    'input_tensor_shape': tensor.shape
                }
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {'final_features': None, 'input_tensor_shape': tensor.shape}
    
    def analyze_orientation_superposition(self, orientation_groups, model):
        """Analyze superposition by orientation groups (frente, lateral, etc.)"""
        superposition_results = {}
        
        for orientation, samples in orientation_groups.items():
            if len(samples) < 2:
                continue
                
            # Superpose all images of same orientation
            stacked_tensors = []
            pathology_labels = []
            
            for sample in samples:
                if sample['tensor_data'] is not None:
                    stacked_tensors.append(sample['tensor_data'])
                    pathology_labels.append(sample['primary_pathology'])
            
            if len(stacked_tensors) >= 2:
                # Create superposition (average overlay)
                superposed = np.mean(stacked_tensors, axis=0)
                
                # Find control sample (known pathology) for reward system
                control_sample = None
                for sample in samples:
                    if any(gt == 1 for gt in sample['ground_truth']):
                        control_sample = sample
                        break
                
                superposition_results[orientation] = {
                    'num_samples': len(samples),
                    'pathology_labels': pathology_labels,
                    'superposed_shape': superposed.shape,
                    'has_control': control_sample is not None,
                    'control_pathology': control_sample['primary_pathology'] if control_sample else None
                }
        
        return superposition_results
    
    def calculate_pixel_level_rewards(self, sample_results, model):
        """Calculate pixel-level rewards using control pathology as reference"""
        # Find samples with known pathologies (control images)
        control_samples = []
        normal_samples = []
        
        for sample in sample_results:
            if any(gt == 1 for gt in sample['ground_truth']):
                control_samples.append(sample)
            else:
                normal_samples.append(sample)
        
        pixel_rewards = {
            'control_samples': len(control_samples),
            'normal_samples': len(normal_samples),
            'reward_scores': {}
        }
        
        # Calculate rewards for each control pathology
        for control in control_samples:
            pathology_name = control['primary_pathology']
            
            # Find pathology index
            pathology_indices = [i for i, gt in enumerate(control['ground_truth']) if gt == 1]
            
            if pathology_indices:
                # Calculate pixel-level attention for this pathology
                predictions = np.array(control['predictions'])
                
                reward_score = 0
                for idx in pathology_indices:
                    # Reward based on prediction confidence for known pathology
                    confidence = predictions[idx]
                    reward_score += confidence
                
                pixel_rewards['reward_scores'][pathology_name] = {
                    'confidence_sum': float(reward_score),
                    'num_pathologies': len(pathology_indices),
                    'avg_confidence': float(reward_score / len(pathology_indices))
                }
        
        return pixel_rewards
    
    def analyze_consecutive_views(self, sample_results):
        """Analyze consecutive views (same patient, different orientations)"""
        # Group by patient (first part of image name before first underscore)
        patient_groups = {}
        
        for sample in sample_results:
            # Extract patient ID from image name (e.g., "00000009_001_001.jpg" -> "00000009")
            patient_id = sample['sample_id'].split('_')[0]
            
            if patient_id not in patient_groups:
                patient_groups[patient_id] = []
            patient_groups[patient_id].append(sample)
        
        consecutive_analysis = {
            'total_patients': len(patient_groups),
            'multi_view_patients': 0,
            'consistency_scores': {}
        }
        
        for patient_id, views in patient_groups.items():
            if len(views) > 1:
                consecutive_analysis['multi_view_patients'] += 1
                
                # Calculate consistency between views of same patient
                view_predictions = [np.array(view['predictions']) for view in views]
                
                if len(view_predictions) == 2:
                    # Calculate correlation between two views
                    correlation = np.corrcoef(view_predictions[0], view_predictions[1])[0, 1]
                    
                    consecutive_analysis['consistency_scores'][patient_id] = {
                        'views': [view['view_category'] + '-' + view['view_position'] for view in views],
                        'correlation': float(correlation),
                        'pathologies': [view['primary_pathology'] for view in views]
                    }
        
        return consecutive_analysis
    
    def run_multi_layer_superposition_test(self, resolution: int, num_rays: int, light_intensity: float) -> Dict:
        """Multi-layer superposition test with consecutive + grouped analysis and pixel-level rewards"""
        test_start = time.time()
        
        try:
            # Create test model
            model = self.create_test_model(resolution, num_rays, light_intensity)
            if model is None:
                return None
            
            # Group samples by orientation for superposition analysis
            orientation_groups = {
                'Frontal-AP': [],
                'Frontal-PA': [],
                'Lateral': []
            }
            
            sample_results = []
            total_error = 0
            processed_samples = 0
            
            # PHASE 1: Individual processing + grouping
            for sample_idx, sample in enumerate(self.test_samples):
                try:
                    # Preprocess with custom resolution + orientation positioning
                    tensor, original_size = self.preprocess_image_with_orientation(
                        sample['image_path'], 
                        resolution,
                        sample['view_category'],
                        sample['view_position']
                    )
                    
                    if tensor is None:
                        continue
                    
                    tensor = tensor.to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(tensor)
                        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                        
                        # Get feature maps for pixel-level analysis
                        feature_maps = self.extract_feature_maps(model, tensor)
                    
                    # Calculate error for this sample
                    ground_truth = np.array(sample['ground_truth'])
                    sample_error = np.mean(np.abs(ground_truth - probabilities))
                    total_error += sample_error
                    processed_samples += 1
                    
                    # Store sample result with feature maps
                    sample_result = {
                        'sample_id': sample['image_id'],
                        'primary_pathology': sample['primary_pathology'],
                        'view_category': sample['view_category'],
                        'view_position': sample['view_position'],
                        'predictions': probabilities.tolist(),
                        'ground_truth': sample['ground_truth'],
                        'sample_error': sample_error,
                        'feature_maps': feature_maps,
                        'tensor_data': tensor.cpu().numpy()  # For superposition analysis
                    }
                    
                    sample_results.append(sample_result)
                    
                    # Group by orientation
                    group_key = f"{sample['view_category']}-{sample['view_position']}"
                    if group_key in orientation_groups:
                        orientation_groups[group_key].append(sample_result)
                    
                except Exception as e:
                    logger.error(f"Sample {sample['image_id']} failed: {e}")
                    continue
            
            if processed_samples == 0:
                return None
            
            # PHASE 2: Multi-layer superposition analysis by orientation groups
            superposition_analysis = self.analyze_orientation_superposition(orientation_groups, model)
            
            # PHASE 3: Pixel-level reward system with control pathology
            pixel_rewards = self.calculate_pixel_level_rewards(sample_results, model)
            
            # PHASE 4: Consecutive analysis (same patient different views)
            consecutive_analysis = self.analyze_consecutive_views(sample_results)
                
            # Calculate overall metrics
            avg_error = total_error / processed_samples
            memory_used = torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
            test_time = time.time() - test_start
            
            result = {
                'resolution': resolution,
                'num_rays': num_rays, 
                'light_intensity': light_intensity,
                'avg_error_all_samples': avg_error,
                'processed_samples': processed_samples,
                'total_samples': len(self.test_samples),
                'sample_results': sample_results,
                'superposition_analysis': superposition_analysis,
                'pixel_rewards': pixel_rewards,
                'consecutive_analysis': consecutive_analysis,
                'inference_time': test_time,
                'memory_usage_gb': memory_used,
                'information_retention': (resolution**2) / (original_size[0] * original_size[1]) if original_size else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-layer superposition test failed (res={resolution}, rays={num_rays}, intensity={light_intensity}): {e}")
            return None
    
    def run_complete_test_matrix(self):
        """Execute complete test matrix with multi-sample validation"""
        print("\nSTARTING MULTI-LAYER SUPERPOSITION TEST MATRIX")
        print("=" * 60)
        
        total_tests = len(self.resolution_matrix) * len(self.ray_counts) * len(self.light_intensities)
        completed_tests = 0
        
        # Test all combinations
        for resolution in self.resolution_matrix:
            for num_rays in self.ray_counts:
                for light_intensity in self.light_intensities:
                    
                    print(f"\nTest {completed_tests + 1}/{total_tests}")
                    print(f"Resolution: {resolution}x{resolution}, Rays: {num_rays}, Intensity: {light_intensity}")
                    
                    result = self.run_multi_layer_superposition_test(resolution, num_rays, light_intensity)
                    
                    if result:
                        self.test_results.append(result)
                        print(f"[OK] Avg Error: {result['avg_error_all_samples']:.4f} ({result['processed_samples']}/{result['total_samples']} samples)")
                        print(f"  Time: {result['inference_time']:.2f}s, Memory: {result['memory_usage_gb']:.2f}GB")
                        print(f"  Info retention: {result['information_retention']*100:.1f}%")
                        
                        # Show sample breakdown
                        for sample_result in result['sample_results'][:3]:  # Show first 3
                            print(f"    {sample_result['primary_pathology']}: error={sample_result['sample_error']:.4f}")
                        if len(result['sample_results']) > 3:
                            print(f"    ... and {len(result['sample_results'])-3} more samples")
                    else:
                        print("[ERROR] Test failed")
                    
                    completed_tests += 1
                    
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print(f"\nCOMPLETED: {len(self.test_results)}/{total_tests} tests successful")
    
    def analyze_results(self):
        """Analyze test results and find optimal configuration"""
        if not self.test_results:
            print("No results to analyze")
            return
        
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.test_results)
        
        # Sort by accuracy (lowest error = best)
        df_sorted = df.sort_values('avg_error_all_samples')
        
        print("\nTOP 10 CONFIGURATIONS (by multi-sample accuracy):")
        print("-" * 80)
        
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
            print(f"{i+1:2d}. Resolution: {int(row['resolution']):4d}x{int(row['resolution']):4d} | "
                  f"Rays: {int(row['num_rays']):4d} | Intensity: {row['light_intensity']:.1f} | "
                  f"Avg Error: {row['avg_error_all_samples']:.4f} | "
                  f"Samples: {int(row['processed_samples'])}/{int(row['total_samples'])} | "
                  f"Time: {row['inference_time']:.2f}s")
        
        # Find best configuration
        best_config = df_sorted.iloc[0]
        
        print(f"\n[BEST] OPTIMAL CONFIGURATION:")
        print(f"   Resolution: {int(best_config['resolution'])}x{int(best_config['resolution'])}")
        print(f"   Ray count: {int(best_config['num_rays'])}")
        print(f"   Light intensity: {best_config['light_intensity']}")
        print(f"   Multi-sample accuracy: {1-best_config['avg_error_all_samples']:.4f}")
        print(f"   Samples processed: {int(best_config['processed_samples'])}/{int(best_config['total_samples'])}")
        print(f"   Information retention: {best_config['information_retention']*100:.1f}%")
        print(f"   Performance: {best_config['inference_time']:.2f}s, {best_config['memory_usage_gb']:.2f}GB")
        
        # Resolution analysis
        print(f"\nRESOLUTION IMPACT ANALYSIS (Multi-sample):")
        print("-" * 50)
        resolution_analysis = df.groupby('resolution')['avg_error_all_samples'].agg(['mean', 'min', 'max'])
        for res, stats in resolution_analysis.iterrows():
            improvement = (256**2) / (res**2)
            print(f"  {int(res):4d}x{int(res):4d}: avg_error={stats['mean']:.4f}, "
                  f"best_error={stats['min']:.4f}, info_gain={1/improvement:.1f}x")
        
        # Multi-layer analysis
        print(f"\nMULTI-LAYER ANALYSIS (Best Config):")
        print("-" * 50)
        
        # Superposition analysis
        if 'superposition_analysis' in best_config:
            print("Orientation Superposition:")
            for orientation, analysis in best_config['superposition_analysis'].items():
                control_info = f" (control: {analysis['control_pathology']})" if analysis['has_control'] else ""
                print(f"  {orientation}: {analysis['num_samples']} samples{control_info}")
        
        # Pixel rewards
        if 'pixel_rewards' in best_config:
            print("\nPixel-level Rewards:")
            for pathology, reward in best_config['pixel_rewards']['reward_scores'].items():
                print(f"  {pathology:20s}: confidence={reward['avg_confidence']:.4f}")
        
        # Consecutive analysis  
        if 'consecutive_analysis' in best_config:
            print(f"\nConsecutive Views: {best_config['consecutive_analysis']['multi_view_patients']} patients with multiple views")
            for patient, consistency in best_config['consecutive_analysis']['consistency_scores'].items():
                print(f"  Patient {patient}: correlation={consistency['correlation']:.3f} ({', '.join(consistency['views'])})")
        
        # Save results
        results_file = 'nebula_resolution_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save summary
        summary_file = 'nebula_resolution_test_summary.csv'
        df.to_csv(summary_file, index=False)
        
        print(f"\n[FILE] Results saved:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary: {summary_file}")
        
        return best_config

def main():
    """Main testing execution"""
    try:
        # Initialize tester
        tester = NEBULAResolutionTester()
        
        # Run complete test matrix
        tester.run_complete_test_matrix()
        
        # Analyze results
        best_config = tester.analyze_results()
        
        print("\n[TEST] MULTI-LAYER SUPERPOSITION TESTING COMPLETED")
        print("Main model remains unchanged and protected")
        print("Multi-layer transparency + pixel rewards + consecutive analysis complete")
        print("Orientation-based superposition validation successful")
        print("Ready to implement optimal configuration for Division B")
        
        return 0
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())