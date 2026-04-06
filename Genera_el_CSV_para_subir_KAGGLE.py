#!/usr/bin/env python3
"""
Script para generar submission usando el modelo NEBULA REAL
=============================================================
Este script usa el modelo original completo, no una versión simplificada
"""

import sys
import os

# CRÍTICO: Cambiar al directorio del modelo ANTES de cualquier import
os.chdir(r"C:\NEBULA_DIVISION_A\MODELOS_OK_USADOS_EN_KAGGLE")
sys.path.insert(0, r"C:\NEBULA_DIVISION_A\MODELOS_OK_USADOS_EN_KAGGLE")

# Ahora importar todo lo necesario del script original
from nebula_unified_CNN_OK_SIMPLE_PRECISION_TOTAL import (
    NEBULAConfig,
    NEBULAGrandXRay,
    OptimizedRayTracer,
    EfficientSelfAttention,
    ResidualBlock,
    TestDataset,
    memory_manager
)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Solución para PyTorch 2.6
torch.serialization.add_safe_globals([
    'NEBULAConfig', 
    '__main__.NEBULAConfig',
    'AdaptiveMemoryManager',
    'ThresholdOptimizer'
])

print("=" * 70)
print("NEBULA GRAND X-RAY - GENERADOR DE SUBMISSION PARA KAGGLE")
print("Usando el modelo ORIGINAL completo con arquitectura de 900 rayos")
print("=" * 70)

# CONFIGURACIÓN - PATHS CORRECTOS
CHECKPOINT_PATH = r"C:\NEBULA_DIVISION_A\MODELOS_OK_USADOS_EN_KAGGLE\checkpoint_epoch_14.pth"
TEST_DIR = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\test1"
SAMPLE_SUBMISSION = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\sample_submission_1.csv"
OUTPUT_FILE = "nebula_division_a_submission_prob.csv"

# Verificar archivos
print("\nVerificando archivos...")
assert Path(CHECKPOINT_PATH).exists(), f"No existe: {CHECKPOINT_PATH}"
assert Path(TEST_DIR).exists(), f"No existe: {TEST_DIR}"
assert Path(SAMPLE_SUBMISSION).exists(), f"No existe: {SAMPLE_SUBMISSION}"
print("✓ Todos los archivos encontrados")

# Configuración del dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsando dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# PASO 1: Crear configuración idéntica a la del entrenamiento
print("\n1. Inicializando configuración NEBULA...")
config = NEBULAConfig()
config.batch_size = 8  # Reducido para evitar problemas de memoria en predicción
config.num_workers = 0  # 0 para Windows para evitar problemas

print(f"   - Resolución: {config.resolution}")
print(f"   - Número de rayos: {config.num_rays}")
print(f"   - Bandas espectrales: {config.num_spectral_bands}")
print(f"   - Clases: {config.num_classes}")

# PASO 2: Cargar el checkpoint
print("\n2. Cargando checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

# Mostrar información del checkpoint
if isinstance(checkpoint, dict):
    print("   Contenido del checkpoint:")
    for key in checkpoint.keys():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
            if key == 'metrics' and isinstance(checkpoint[key], dict):
                print(f"   - {key}:")
                for metric_key, metric_val in checkpoint[key].items():
                    if isinstance(metric_val, (int, float)):
                        print(f"     · {metric_key}: {metric_val:.4f}")
            elif key in ['best_auc', 'best_precision']:
                print(f"   - {key}: {checkpoint[key]:.4f}")
            else:
                print(f"   - {key}: {checkpoint[key]}")

# PASO 3: Crear el modelo NEBULA completo
print("\n3. Creando modelo NEBULA Grand X-Ray...")
model = NEBULAGrandXRay(config).to(DEVICE)

# Calcular número de parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total de parámetros: {total_params:,}")

# PASO 4: Cargar los pesos del modelo
print("\n4. Cargando pesos del modelo...")
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   ✓ Pesos cargados desde model_state_dict")
else:
    print("   ERROR: El checkpoint no contiene model_state_dict")
    sys.exit(1)

model.eval()
print("   ✓ Modelo en modo evaluación")

# PASO 5: Cargar datos de test
print("5. Preparando datos de test...")
sample_df = pd.read_csv(SAMPLE_SUBMISSION)
test_image_names = sample_df['Image_name'].tolist()
print(f"   Total de imágenes de test: {len(test_image_names)}")

# Crear dataset de test usando la clase del modelo original
test_dataset = TestDataset(
    image_dir=TEST_DIR,
    image_names=test_image_names,
    config=config
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False  # False para evitar problemas en Windows
)

print(f"   Batches totales: {len(test_loader)}")

# PASO 6: Generar predicciones
print("\n6. Generando predicciones con el modelo NEBULA...")
print("   (Esto puede tomar varios minutos)")

all_predictions = []
all_image_ids = []

# Condiciones médicas
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Prediciendo")):
        images = batch['image'].to(DEVICE)
        image_ids = batch['image_id']
        
        try:
            # Usar mixed precision si está configurado
            if config.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Aplicar sigmoid para obtener probabilidades
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.append(predictions)
            all_image_ids.extend(image_ids)
            
            # Limpiar memoria periódicamente
            if batch_idx % 50 == 0:
                memory_manager.adaptive_cleanup()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n   ⚠ Memoria insuficiente en batch {batch_idx}")
                # Limpiar memoria agresivamente
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Reducir batch a la mitad y reintentar
                half_size = len(images) // 2
                for sub_batch in [0, half_size]:
                    sub_images = images[sub_batch:sub_batch+half_size]
                    sub_outputs = model(sub_images)
                    sub_predictions = torch.sigmoid(sub_outputs).cpu().numpy()
                    all_predictions.append(sub_predictions)
            else:
                raise e

# PASO 7: Procesar predicciones
print("\n7. Procesando predicciones...")
all_predictions = np.vstack(all_predictions)
print(f"   Shape de predicciones: {all_predictions.shape}")
print(f"   Rango de valores: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")

# PASO 8: Crear DataFrame de submission con probabilidades
print("\n8. Creando archivo de submission con probabilidades...")
submission = pd.DataFrame()
submission['Image_name'] = all_image_ids

for i, condition in enumerate(CONDITIONS):
    submission[condition] = all_predictions[:, i]

# Verificar que tenemos todas las imágenes
if len(submission) != len(test_image_names):
    print(f"   ⚠ ADVERTENCIA: {len(submission)} predicciones vs {len(test_image_names)} esperadas")

# PASO 9: Guardar submission
submission.to_csv(OUTPUT_FILE, index=False)
print(f"\n   ✓ Archivo guardado: {OUTPUT_FILE}")

# PASO 10: Mostrar una muestra del archivo de submission
print("\n9. Muestra del archivo de submission:")
print("=" * 70)
print(submission.head())
print("=" * 70)


# Verificación final
print("\n" + "=" * 70)
print("✓ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivo de probabilidades listo para Kaggle: {OUTPUT_FILE}")
print("Puedes subirlo directamente a:")
print("https://www.kaggle.com/competitions/grand-xray-slam-division-a/submit")
print("\n¡Buena suerte en la competición!")