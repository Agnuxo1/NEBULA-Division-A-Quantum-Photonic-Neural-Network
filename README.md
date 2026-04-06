# NEBULA Division A — Quantum Photonic Neural Network
## Kaggle Grand X-Ray SLAM Competition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/Agnuxo1/NEBULA-Division-A-Quantum-Photonic-Neural-Network)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

NEBULA Division A is the primary training and inference system developed for the **Kaggle Grand X-Ray SLAM** medical imaging competition. It features a unified CNN with quantum-photonic principles, P2P expert coordination, and RTX 3090-optimized GPU pipeline targeting AUC > 0.93 on chest X-ray multi-pathology classification.

---

## Features

- **Unified CNN Architecture**: `nebula_unified_CNN_OK_SIMPLE_PRECISION_TOTAL.py` — production-ready multi-label classifier
- **P2P Expert System (14 Experts)**: Each expert specializes in a specific pathology; turn-based coordination prevents GPU conflicts
- **RTX 3090 Optimized**: Tensor Cores (TF32), batch size 256, 80%+ GPU utilization (up from 2-4%)
- **Auto-Save System**: Checkpoints every 15 minutes to survive session interruptions
- **Automatic Kaggle Submission**: CSV generation and direct submission via `Genera_el_CSV_para_subir_KAGGLE.py`
- **Background Training**: Scripts designed for persistent background execution
- **Checkpoint Recovery**: Resume training from any saved epoch (e.g., epoch 28 → AUC 0.8696)

---

## Architecture

```
NEBULA Division A
├── nebula_unified_CNN_OK_SIMPLE_PRECISION_TOTAL.py  # Main training script
├── Fixed_Training_System.py                          # Fault-tolerant training loop
├── Genera_el_CSV_para_subir_KAGGLE.py                # Kaggle submission generator
├── Generate_Current_Submission.py                    # Current model submission
├── Analyze_Checkpoint.py                             # Checkpoint analysis tool
├── Force_Save.py / Signal_Force_Save.py              # Manual checkpoint triggers
├── Background_Kaggle_Monitor.py                      # Training monitor
├── Windows_Force_Save.py                             # Windows-specific save utility
├── Windows_Process_Injection.py                      # Process monitoring utility
├── Test_Division_A_Saving.py                         # Save system tests
├── models/                                           # Saved checkpoints (.pth)
├── MODELOS_OK_USADOS_EN_KAGGLE/                      # Validated competition models
├── MODELOS_OK_USADOS_EN_KAGGLE_BUENOS/               # Top-performing models
├── SIMPLE_NEBULA_CNN_OK_CONCURSO/                    # Competition-ready CNN
├── auto_submissions/                                 # Auto-generated CSVs
├── submissions/                                      # Final submission files
├── results/                                          # Training metrics
├── scripts/                                          # Utility scripts
└── test_models/                                      # Model evaluation outputs
```

---

## Requirements

```
torch>=2.0
torchvision
numpy
pandas
scikit-learn
Pillow
tqdm
kaggle
```

### Hardware
- **GPU**: NVIDIA RTX 3090 (24 GB) recommended; RTX 3070+ minimum
- **RAM**: 32 GB+
- **Storage**: 200 GB (for dataset and checkpoints)

---

## Installation

```bash
git clone https://github.com/Agnuxo1/NEBULA-Division-A-Quantum-Photonic-Neural-Network.git
cd NEBULA-Division-A-Quantum-Photonic-Neural-Network
pip install torch torchvision numpy pandas scikit-learn Pillow tqdm kaggle

# Configure Kaggle credentials (environment variables — do NOT hardcode)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Usage

```bash
# Start main training (RTX 3090 optimized)
python Fixed_Training_System.py

# Run unified CNN training
python nebula_unified_CNN_OK_SIMPLE_PRECISION_TOTAL.py

# Generate Kaggle submission CSV
python Genera_el_CSV_para_subir_KAGGLE.py

# Analyze a checkpoint
python Analyze_Checkpoint.py --checkpoint models/nebula_official_epoch_0030.pth

# Monitor background training
python Background_Kaggle_Monitor.py
```

---

## Competition Results

| Metric | Value |
|--------|-------|
| Best AUC (Kaggle) | 0.8696+ (epoch 28) |
| Target AUC | > 0.936819 (1st place) |
| GPU Utilization | 80%+ (RTX 3090) |
| Throughput | 416+ samples/min |
| Batch Size | 256 (TF32 Tensor Cores) |

---

## Security Note

Kaggle API credentials must be provided via environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`. Never commit credential files to the repository.

---

## Author

**Francisco Angulo de Lafuente**
- GitHub: [@Agnuxo1](https://github.com/Agnuxo1)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
