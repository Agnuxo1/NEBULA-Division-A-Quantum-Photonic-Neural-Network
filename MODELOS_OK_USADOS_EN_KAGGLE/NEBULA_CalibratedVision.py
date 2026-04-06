#!/usr/bin/env python3
"""
NEBULA CALIBRATED VISION SYSTEM
===============================
Standardized calibrated vision system for ALL NEBULA models
Ensures consistent multi-spectral analysis across entire framework

!!! CRITICAL CALIBRATION REQUIREMENT !!!
===========================================
THIS MODULE REQUIRES MANDATORY CALIBRATION BEFORE USE:
1. Run NEBULA_Stepped_Calibration.py FIRST
2. Generates nebula_stepped_calibration_complete.json
3. This module loads and applies calibrated vision parameters
4. NEVER use NEBULA models without this calibrated vision system

Usage:
    from NEBULA_CalibratedVision import CalibratedVisionSystem
    vision = CalibratedVisionSystem()
    vision.load_calibration('nebula_stepped_calibration_complete.json')
    calibrated_response = vision.process(xray_image)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CalibratedVisionSystem(nn.Module):
    """
    Standardized calibrated vision system for all NEBULA models
    Loads and applies 4-step calibrated parameters consistently
    """
    
    def __init__(self):
        super().__init__()
        self.calibration_loaded = False
        self.calibration_config = None
        
        # Will be populated after loading calibration
        self.step_analyzers = nn.ModuleDict()
        self.anomaly_reference = None
        
        logger.info("NEBULA Calibrated Vision System initialized")
        logger.info("WARNING: Calibration must be loaded before use!")
    
    def load_calibration(self, calibration_file: str = 'nebula_stepped_calibration_complete.json'):
        """
        Load calibration parameters from stepped calibration file
        """
        calibration_path = Path(calibration_file)
        
        if not calibration_path.exists():
            raise FileNotFoundError(
                f"CALIBRATION FILE NOT FOUND: {calibration_file}\n"
                f"You MUST run NEBULA_Stepped_Calibration.py first!\n"
                f"NEVER use NEBULA models with uncalibrated vision!"
            )
        
        logger.info(f"Loading NEBULA calibration from: {calibration_file}")
        
        with open(calibration_path, 'r') as f:
            self.calibration_config = json.load(f)
        
        # Validate calibration file
        if self.calibration_config.get('calibration_mode') != 'STEPPED_HUMAN_TO_NEBULA':
            raise ValueError(
                f"Invalid calibration file! Expected STEPPED_HUMAN_TO_NEBULA mode\n"
                f"Found: {self.calibration_config.get('calibration_mode')}\n"
                f"Please run NEBULA_Stepped_Calibration.py to generate proper calibration!"
            )
        
        # Load anomaly detection reference
        self.anomaly_reference = self.calibration_config['anomaly_detection_reference']
        
        # Initialize step analyzers with calibrated parameters
        self._initialize_step_analyzers()
        
        self.calibration_loaded = True
        
        logger.info("NEBULA Calibration loaded successfully!")
        logger.info(f"Calibration date: {self.calibration_config.get('calibration_date')}")
        logger.info(f"Steps completed: {len(self.calibration_config['steps_completed'])}")
        logger.info(f"Anomaly reference range: [{self.anomaly_reference['min']:.4f}, {self.anomaly_reference['max']:.4f}]")
    
    def _initialize_step_analyzers(self):
        """Initialize calibrated analyzers for each step"""
        
        step_results = self.calibration_config['step_results']
        
        for step_name, step_data in step_results.items():
            step_config = step_data['step_config']
            best_params = step_data['best_params']
            
            wavelengths = step_config['wavelengths']
            num_wavelengths = len(wavelengths)
            
            # Create calibrated analyzer for this step
            analyzer = CalibratedStepAnalyzer(
                wavelengths=wavelengths,
                laser_intensities=best_params['laser_intensities'],
                sensor_sensitivities=best_params['sensor_sensitivities'],
                tissue_interactions=best_params['tissue_interactions']
            )
            
            self.step_analyzers[step_name] = analyzer
            
            logger.info(f"Initialized {step_name}: {num_wavelengths} wavelengths")
    
    def _check_calibration_loaded(self):
        """Ensure calibration is loaded before processing"""
        if not self.calibration_loaded:
            raise RuntimeError(
                "CALIBRATION NOT LOADED!\n"
                "You must call load_calibration() first!\n"
                "NEVER use NEBULA models with uncalibrated vision!\n"
                "Run: vision.load_calibration('nebula_stepped_calibration_complete.json')"
            )
    
    def process_all_steps(self, xray_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process X-ray through all 4 calibrated steps
        Returns responses from each calibration step
        """
        self._check_calibration_loaded()
        
        step_responses = {}
        
        # Process through each calibrated step
        for step_name, analyzer in self.step_analyzers.items():
            step_response = analyzer(xray_image)
            step_responses[step_name] = step_response
            
        return step_responses
    
    def get_averaged_response(self, xray_image: torch.Tensor) -> torch.Tensor:
        """
        Get averaged response from all 4 steps (matches calibration methodology)
        This is the primary method for anomaly detection
        """
        self._check_calibration_loaded()
        
        step_responses = self.process_all_steps(xray_image)
        
        # Average all step responses (spatially)
        normalized_responses = []
        for step_name, response in step_responses.items():
            # Average across wavelength dimension to get spatial response
            spatial_response = torch.mean(response, dim=1)
            normalized_responses.append(spatial_response)
        
        # Average all 4 steps
        averaged_response = torch.mean(torch.stack(normalized_responses), dim=0)
        
        return averaged_response
    
    def detect_anomalies(self, xray_image: torch.Tensor, 
                        threshold_multiplier: float = 2.0) -> torch.Tensor:
        """
        Detect anomalies using calibrated reference
        
        Args:
            xray_image: Input X-ray image
            threshold_multiplier: How many std devs from reference mean
            
        Returns:
            Anomaly mask (1 = anomaly, 0 = normal)
        """
        self._check_calibration_loaded()
        
        # Get calibrated response
        response = self.get_averaged_response(xray_image)
        
        # Compare to calibrated reference
        ref_mean = self.anomaly_reference['mean']
        ref_std = self.anomaly_reference['std']
        
        # Calculate deviation from reference
        deviation = torch.abs(response - ref_mean)
        threshold = ref_std * threshold_multiplier
        
        # Anomaly detection
        anomaly_mask = (deviation > threshold).float()
        
        return anomaly_mask
    
    def get_calibration_info(self) -> Dict:
        """Get information about current calibration"""
        self._check_calibration_loaded()
        
        return {
            'calibration_loaded': self.calibration_loaded,
            'calibration_date': self.calibration_config.get('calibration_date'),
            'steps_completed': self.calibration_config['steps_completed'],
            'performance_progression': self.calibration_config['performance_progression'],
            'anomaly_reference': self.anomaly_reference
        }

class CalibratedStepAnalyzer(nn.Module):
    """
    Individual step analyzer with calibrated parameters
    """
    
    def __init__(self, wavelengths: List[float], 
                 laser_intensities: List[float],
                 sensor_sensitivities: List[float],
                 tissue_interactions: List[List[float]]):
        super().__init__()
        
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)
        
        # Load calibrated parameters (frozen - no learning)
        self.laser_intensities = torch.FloatTensor(laser_intensities)
        self.sensor_sensitivities = torch.FloatTensor(sensor_sensitivities)
        self.tissue_interactions = torch.FloatTensor(tissue_interactions)
        
        logger.info(f"Calibrated analyzer: {self.num_wavelengths} wavelengths")
    
    def forward(self, xray_image: torch.Tensor) -> torch.Tensor:
        """
        Process X-ray with calibrated parameters
        
        Args:
            xray_image: [batch, height, width] or [batch, 1, height, width]
            
        Returns:
            Calibrated multi-wavelength response [batch, wavelengths, height, width]
        """
        # Handle input dimensions - ensure 4D tensor [batch, channels, height, width]
        if len(xray_image.shape) == 3:
            xray_image = xray_image.unsqueeze(1)  # Add channel dimension
        elif len(xray_image.shape) == 2:
            xray_image = xray_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Ensure we have the right format: [batch, 1, height, width]
        if len(xray_image.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape: {xray_image.shape}")
        
        if xray_image.shape[1] != 1:
            # Take first channel if multi-channel
            xray_image = xray_image[:, 0:1, :, :]
        
        # Remove channel dimension for processing: [batch, height, width]
        phantom = xray_image.squeeze(1)
        
        # Extract dimensions
        batch_size, height, width = phantom.shape
        wavelength_responses = []
        
        # Process each wavelength with calibrated parameters
        for i, wavelength in enumerate(self.wavelengths):
            # Get calibrated parameters for this wavelength
            absorption = self.tissue_interactions[i, 0]
            scattering = self.tissue_interactions[i, 1]
            penetration = self.tissue_interactions[i, 2]
            
            # Wavelength-specific tissue interaction (calibrated)
            tissue_response = phantom * torch.exp(-phantom * absorption)  # Absorption
            tissue_response *= torch.exp(-phantom * scattering)  # Scattering  
            tissue_response *= (1 + penetration)  # Penetration
            
            # Apply calibrated laser intensity and sensor sensitivity
            intensity = self.laser_intensities[i]
            sensitivity = self.sensor_sensitivities[i]
            
            final_response = tissue_response * intensity * sensitivity
            wavelength_responses.append(final_response)
        
        # Stack wavelength responses
        stacked_responses = torch.stack(wavelength_responses, dim=1)  # [batch, wavelengths, height, width]
        
        return stacked_responses

def load_nebula_calibrated_vision(calibration_file: str = 'nebula_stepped_calibration_complete.json') -> CalibratedVisionSystem:
    """
    Convenience function to load calibrated vision system
    
    Returns:
        Ready-to-use CalibratedVisionSystem
    """
    vision = CalibratedVisionSystem()
    vision.load_calibration(calibration_file)
    return vision

# Test function
def test_calibrated_vision():
    """Test calibrated vision system"""
    logger.info("Testing NEBULA Calibrated Vision System")
    
    try:
        # Load calibrated vision
        vision = load_nebula_calibrated_vision()
        
        # Test with dummy X-ray
        test_xray = torch.randn(1, 1, 256, 256)  # [batch, channel, height, width]
        
        # Test all steps
        step_responses = vision.process_all_steps(test_xray)
        logger.info(f"Processed through {len(step_responses)} calibrated steps")
        
        # Test averaged response
        averaged_response = vision.get_averaged_response(test_xray)
        logger.info(f"Averaged response shape: {averaged_response.shape}")
        
        # Test anomaly detection
        anomaly_mask = vision.detect_anomalies(test_xray)
        logger.info(f"Anomaly detection shape: {anomaly_mask.shape}")
        logger.info(f"Anomalies detected: {torch.sum(anomaly_mask).item()} pixels")
        
        # Show calibration info
        cal_info = vision.get_calibration_info()
        logger.info(f"Calibration info: {cal_info}")
        
        logger.info("NEBULA Calibrated Vision System: TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"NEBULA Calibrated Vision System: TEST FAILED")
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("NEBULA CALIBRATED VISION SYSTEM v1.0")
    test_calibrated_vision()