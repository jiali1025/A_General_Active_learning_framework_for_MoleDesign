# Unified Active Learning Framework for Molecular Design

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)

A comprehensive framework for active learning in molecular property prediction and design, implementing multiple acquisition strategies for efficient data exploration and model improvement.

## 🚀 Overview

This repository contains implementations of various active learning strategies for molecular design:

- **Pure Uncertainty Sampling**: Traditional uncertainty-based acquisition
- **Target Property**: Property-guided molecular selection with 4 variants
- **Mixed Strategies**: Hybrid approaches combining uncertainty and diversity (8 variants)
- **Expected Improvement**: Bayesian optimization inspired acquisition (3 variants)
- **Threshold Testing**: Performance-based selection strategies (4 variants)
- **Retrosynthesis Integration**: Synthesis-aware molecular discovery

## 📁 Project Structure

```
├── src/                              # Source code organized by strategy
│   ├── acquisition_function_tot.py   # Core acquisition functions (786 lines)
│   ├── pure_uncertainty/             # Pure uncertainty sampling
│   ├── target_property/              # 4 target property variants
│   ├── mix_version/                  # 8 mixed strategy variants
│   │   ├── synthesizability_v2/      # Synthesis-aware strategies (4 variants)
│   │   │   ├── uncertainty_1_target_7_batch/  ├── uncertainty_2_target_6_batch/
│   │   │   ├── uncertainty_4_target_4_batch/  └── uncertainty_target_batch/
│   │   ├── uncertainty_1_target_7_batch/  ├── uncertainty_2_target_6_batch/
│   │   ├── uncertainty_4_target_4_batch/  └── uncertainty_target_nobatch_new/
│   ├── expected_improvement/         # 3 expected improvement variants
│   ├── threshold_test/              # 4 threshold testing variants
│   └── retro/                       # Retrosynthesis integration
├── models/                          # Pre-trained models (download via scripts)
├── data/
│   ├── raw/                         # Raw datasets
│   ├── processed/                   # Processed datasets
│   └── external/                    # External data (test sets)
├── scripts/                         # Download and utility scripts
├── notebooks/                       # Jupyter notebooks for analysis
├── configs/                         # Configuration files
├── experiments/                     # Experimental results & analysis
├── docs/                            # Documentation & papers
├── Figures/                         # Figures for publications
└── [Research Folders]               # Original research structure preserved
    ├── xTB_calculations/  ├── xTB_ML_models/  ├── TDDFT_calculations/
    ├── active_learning/   ├── chemprop_for_see_change/  └── ...
```

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/jiali1025/A_General_Active_learning_framework_for_MoleDesign.git
cd A_General_Active_learning_framework_for_MoleDesign
```

2. **Create conda environment**:
```bash
conda create -n unified_al python=3.8
conda activate unified_al
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Data and Models

**Large datasets and trained models are stored externally to keep the repository lightweight.**

### Available Datasets
- **Complete Dataset**: `dt_tot_round1.csv` (655K molecules, 76MB)
- **Training Data**: `xTB_ML_combined_train_data.csv` (314K molecules, 36MB)  
- **Test Sets**: Various test splits for evaluation
- **External Data**: Additional molecular databases

### Pre-trained Models
- **Pure Uncertainty**: Final round models (Round 9)
- **Target Property**: 4 variants × final models
- **Mixed Strategies**: 8 variants × final models (including synthesizability v2)  
- **Expected Improvement**: 3 variants × final models
- **Threshold Testing**: 4 variants × final models
- **Retrosynthesis**: Synthesis-aware models

### 📥 Download Data and Models

```bash
# Download datasets (Google Drive public folder)
gdown --folder https://drive.google.com/drive/folders/1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat

# Download models (replace FOLDER_ID with your Google Drive models folder ID)
gdown --folder https://drive.google.com/drive/folders/FOLDER_ID

# Or use the provided download scripts
bash scripts/download_models.sh   # Linux / macOS
./scripts/download_models.ps1      # Windows PowerShell
```

## 🎯 Quick Start

1. **Download required data** (see above)
2. **Prepare your molecular dataset** in SMILES format
3. **Run active learning**:

```python
from src.acquisition_function_tot import pure_uncertainty
from src.pure_uncertainty.main import main

# Example: Pure uncertainty sampling
main()  # Runs 9 rounds of active learning
```

## 📖 Acquisition Strategies

### 1. Pure Uncertainty Sampling
- **Code**: `src/pure_uncertainty/`
- **Usage**: General-purpose active learning
- **Rounds**: 9 training rounds implemented

### 2. Target Property Sampling
- **Variants**: v1_sep, v2_sep, v1_sep_batch, v2_sep_batch
- **Code**: `src/target_property/`  
- **Usage**: Property optimization tasks
- **Target Properties**: xTB_S1, xTB_T1 energies

### 3. Mixed Strategies
- **Code**: `src/mix_version/`
- **8 Variants**: Different uncertainty/diversity/synthesis combinations
- **Synthesizability v2**: 4 synthesis-aware sub-variants
  - `uncertainty_1_target_7_batch`, `uncertainty_2_target_6_batch`
  - `uncertainty_4_target_4_batch`, `uncertainty_target_batch`
- **General Variants**: Core uncertainty/diversity combinations
- **Usage**: Complex multi-objective scenarios with synthesis feasibility

### 4. Expected Improvement  
- **Code**: `src/expected_improvement/`
- **3 Variants**: v_001, v_003, v_005 (different thresholds)
- **Usage**: Continuous property optimization

### 5. Threshold Testing
- **Code**: `src/threshold_test/`
- **4 Variants**: v_5_02, v_5_04, v_5_06, v_5_08
- **Usage**: Classification-like screening

### 6. Retrosynthesis Integration
- **Code**: `src/retro/`
- **Usage**: Synthesizable drug discovery
- **Integration**: AiZynthFinder compatibility

## 🧪 Experimental Results

Results include:
- Learning curves for each strategy
- Molecular diversity analysis
- Property prediction performance  
- Synthesis accessibility scores

See `notebooks/` for detailed analysis and visualization.

## 📚 Dependencies

Key dependencies (fixed versions for stability):
- `rdkit==2023.9.2` - Molecular manipulation
- `chemprop==1.5.2` - Graph neural networks for molecules  
- `torch==1.13.1` - Deep learning framework
- `numpy==1.24.3`, `pandas==2.0.3` - Data processing
- `scikit-learn==1.2.1` - Machine learning utilities

See `requirements.txt` for complete list.

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ChemProp for molecular property prediction
- RDKit for molecular manipulation  
- Active learning community for methodological insights