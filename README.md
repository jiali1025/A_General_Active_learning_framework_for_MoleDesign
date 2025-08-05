# Unified Active Learning Framework for Molecular Design

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](#citation)

A comprehensive framework for active learning in molecular property prediction and design, implementing multiple acquisition strategies for efficient data exploration and model improvement.

## Overview

This repository contains implementations of various active learning strategies for molecular design:

- **Pure Uncertainty Sampling**: Traditional uncertainty-based acquisition
- **Target Property**: Property-guided molecular selection  
- **Mixed Strategies**: Hybrid approaches combining uncertainty and diversity
- **Expected Improvement**: Bayesian optimization inspired acquisition
- **Threshold Testing**: Performance-based selection strategies
- **Retrosynthesis Integration**: Synthesis-aware molecular discovery

## Project Structure

```
├── src/                        # Source code
│   ├── acquisition_function_tot.py  # Core acquisition functions
│   ├── pure_uncertainty/      # Pure uncertainty sampling
│   ├── target_property/       # Target property methods
│   ├── mix_version/           # Mixed acquisition strategies
│   ├── expected_improvement/  # Expected improvement methods
│   ├── threshold_test/        # Threshold-based methods
│   └── retro/                 # Retrosynthesis integration
├── models/                    # Pre-trained models (downloaded separately)
├── data/                      # Data management
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── external/              # External data sources  
├── notebooks/                 # Jupyter notebooks for analysis
├── experiments/               # Experimental results and analysis
├── docs/                      # Documentation and papers
├── scripts/                   # Utility scripts
└── configs/                   # Configuration files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jiali1025/A_General_Active_learning_framework_for_MoleDesign.git
cd A_General_Active_learning_framework_for_MoleDesign
```

2. Create and activate a conda environment:
```bash
conda create -n unified_al python=3.8
conda activate unified_al
```

3. Install dependencies:
```bash
pip install -r requirements.txt  # You'll need to create this
```

## Download Pre-trained Models

Due to GitHub's file size limitations, pre-trained models are hosted separately:

### Option 1: Automatic Download (Recommended)

**Linux/Mac:**
```bash
bash scripts/download_models.sh
```

**Windows:**
```powershell
.\scripts\download_models.ps1
```

### Option 2: Manual Download

Models are available at:
- **Hugging Face Hub**: [Coming Soon] 
- **Zenodo**: [DOI to be assigned]

Download and extract to the `models/` directory following the structure above.

## Quick Start

1. Download the models (see above)
2. Prepare your molecular dataset in SMILES format
3. Run active learning with your preferred strategy:

```python
from src.acquisition_function_tot import AcquisitionFunction
from src.pure_uncertainty.main import run_active_learning

# Example: Pure uncertainty sampling
results = run_active_learning(
    data_path="data/your_dataset.csv",
    model_type="chemprop",
    acquisition_strategy="pure_uncertainty",
    n_rounds=10,
    batch_size=100
)
```

## Acquisition Strategies

### 1. Pure Uncertainty Sampling
- **Location**: `src/pure_uncertainty/`
- **Description**: Selects molecules with highest prediction uncertainty
- **Best for**: General-purpose active learning

### 2. Target Property Sampling  
- **Location**: `src/target_property/`
- **Variants**: v1_sep, v2_sep, v1_sep_batch, v2_sep_batch
- **Description**: Focuses on specific molecular properties
- **Best for**: Property optimization tasks

### 3. Mixed Strategies
- **Location**: `src/mix_version/`
- **Variants**: Multiple uncertainty/diversity combinations
- **Description**: Balances exploration and exploitation
- **Best for**: Complex multi-objective scenarios

### 4. Expected Improvement
- **Location**: `src/expected_improvement/`  
- **Variants**: v_001, v_003, v_005 (different threshold parameters)
- **Description**: Bayesian optimization approach
- **Best for**: Continuous property optimization

### 5. Threshold Testing
- **Location**: `src/threshold_test/`
- **Variants**: v_5_02, v_5_04, v_5_06, v_5_08
- **Description**: Performance-threshold based selection
- **Best for**: Classification-like molecular screening

### 6. Retrosynthesis Integration
- **Location**: `src/retro/`
- **Description**: Synthesis-aware molecular selection
- **Best for**: Synthesizable drug discovery

## Data Management

### Raw Data
The complete raw datasets are available at: [Google Drive Link](https://drive.google.com/drive/folders/157SqOv5A0NVQMEXT0wPlbo7ndq91Fs8V?usp=sharing)

### Dataset Structure
- **Training Data**: Initial labeled molecular data
- **Candidate Pool**: Unlabeled molecules for active selection
- **Test Set**: Hold-out evaluation data
- **External Sources**: Additional molecular databases

## Results and Analysis

Experimental results including:
- Learning curves for each strategy
- Molecular diversity analysis  
- Property prediction performance
- Synthesis accessibility scores

See `notebooks/` for detailed analysis and visualization.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{unified_active_learning_2024,
  title={A Unified Active Learning Framework for Molecular Design},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024},
  note={[DOI to be assigned]}
}
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ChemProp for molecular property prediction
- RDKit for molecular manipulation
- Active learning community for methodological insights

## Contact

- **Author**: [Your Name]
- **Email**: [Your Email] 
- **GitHub**: [@jiali1025](https://github.com/jiali1025)
