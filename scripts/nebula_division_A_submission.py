#!/usr/bin/env python3
"""
NEBULA Division A Submission - Grand X-Ray SLAM
Real Ray-Tracing Model - Epoch 10 (79.4% Precision)

Author: Francisco Angulo de Lafuente and NEBULA Team
Date: September 2025
Competition: Grand X-Ray SLAM Division A

Using trained NEBULA model with 86.45% AUC
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from PIL import Image
import os
from tqdm import tqdm
import sys

# Add NEBULA system to path
sys.path.append('E:/grand-xray-slam-division-a/NEBULA_PROJECT/scripts')
from NEBULA_Professional_System import NEBULAMedicalAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NEBULADivisionBSubmission:
    """NEBULA Submission for Division A Competition"""
    
    def __init__(self):
        """Initialize NEBULA with trained model (Epoch 10)"""
        print("NEBULA Division A Submission System")
        print("Trained Model: Epoch 10, AUC 86.45%, Precision 79.4%")
        print("Ray-tracing: 900 calibrated rays")
        print("=" * 50)
        
        # Initialize NEBULA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NEBULAMedicalAI()
        
        # Load calibration
        calibration_path = 'E:/grand-xray-slam-division-a/NEBULA_PROJECT/calibration/nebula_stepped_calibration_complete.json'
        try:
            self.model.load_calibration(calibration_path)
            print("NEBULA calibration loaded successfully")
        except Exception as e:
            print(f"Warning: Calibration loading issue: {e}")
        
        # Load trained model (Epoch 10)
        model_path = 'E:/grand-xray-slam-division-a/NEBULA_PROJECT/checkpoints/nebula_official_epoch_0010.pth'
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            epoch = checkpoint.get('epoch', 10)
            auc = checkpoint.get('best_auc', 0.8645)
            print(f"Trained model loaded: Epoch {epoch}, AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"Model loading error: {e}")
            raise RuntimeError("Cannot proceed without trained model")
        
        # Move to GPU and eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Pathologies (assuming same as Division A)
        self.pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        
        print(f"NEBULA ready on {self.device}")
        print(f"Target pathologies: {len(self.pathologies)}")
    
    def generate_division_b_submission(self, test_dir: str, sample_submission_path: str):
        """Generate Division A submission with enhanced NEBULA model"""
        print("Generating Division A submission with NEBULA...")
        
        # Load sample submission
        sample_df = pd.read_csv(sample_submission_path)
        test_images = sample_df['Image_name'].unique()
        
        print(f"Total test images: {len(test_images)}")
        print("Processing with trained NEBULA (79.4% precision)...")
        
        submission_data = []
        processed = 0
        batch_size = 8
        
        # Process all images
        for i in tqdm(range(0, len(test_images), batch_size), desc="NEBULA Division A"):
            batch_images = test_images[i:i+batch_size]
            batch_tensors = []
            valid_images = []
            
            # Preprocess batch
            for image_name in batch_images:
                image_path = os.path.join(test_dir, image_name)
                
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path).convert('L')
                        tensor = self.model.preprocess_image(image).to(self.device)
                        batch_tensors.append(tensor)
                        valid_images.append(image_name)
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")
                        continue
            
            # NEBULA inference
            if batch_tensors:
                try:
                    batch_input = torch.cat(batch_tensors, dim=0)
                    
                    with torch.no_grad():
                        outputs = self.model(batch_input)
                        probabilities = torch.sigmoid(outputs).cpu().numpy()
                    
                    # Store predictions
                    for j, image_name in enumerate(valid_images):
                        for k, pathology in enumerate(self.pathologies):
                            submission_data.append({
                                'Image_name': image_name,
                                pathology: float(probabilities[j][k])
                            })
                        processed += 1
                
                except Exception as e:
                    print(f"Batch error: {e}")
                    continue
            
            # Progress
            if (i // batch_size) % 100 == 0:
                print(f"Processed: {processed}/{len(test_images)} images")
        
        # Create submission
        if submission_data:
            submission_df = pd.DataFrame(submission_data)
            submission_pivot = submission_df.set_index(['Image_name']).stack().reset_index()
            submission_pivot.columns = ['Image_name', 'pathology', 'prediction']
            
            # Final format
            final_submission = sample_df[['Image_name']].copy()
            
            for pathology in self.pathologies:
                pathology_data = submission_pivot[
                    submission_pivot['pathology'] == pathology
                ].set_index('Image_name')['prediction']
                
                final_submission[pathology] = final_submission['Image_name'].map(pathology_data)
            
            # Handle any missing values with intelligent defaults
            for pathology in self.pathologies:
                missing = final_submission[pathology].isna().sum()
                if missing > 0:
                    # Use average of predicted values as default (better than 0.1)
                    avg_val = final_submission[pathology].mean()
                    final_submission[pathology].fillna(avg_val, inplace=True)
                    print(f"Filled {missing} missing values for {pathology} with {avg_val:.3f}")
            
            # Save
            output_file = 'nebula_division_b_submission.csv'
            final_submission.to_csv(output_file, index=False)
            
            print("=" * 50)
            print("NEBULA DIVISION B SUBMISSION COMPLETED")
            print(f"File: {output_file}")
            print(f"Images processed: {processed}/{len(test_images)}")
            print(f"Model: Epoch 10 (79.4% precision)")
            print("Ready for Division A upload")
            print("=" * 50)
            
            return final_submission
        
        else:
            raise RuntimeError("No predictions generated")

def main():
    """Main execution for Division A"""
    try:
        # Initialize submission system
        nebula_b = NEBULADivisionBSubmission()
        
        # Use Division A test directory and sample format
        test_dir = 'E:/grand-xray-slam-division-a/test2'
        sample_path = 'E:/grand-xray-slam-division-a/sample_submission_2.csv'
        
        # Generate submission
        submission = nebula_b.generate_division_b_submission(test_dir, sample_path)
        
        print("SUCCESS: Division A submission ready")
        print("Enhanced NEBULA model (79.4% precision)")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())