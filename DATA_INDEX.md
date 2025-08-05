# Data and Model Index

## 📊 Complete Dataset Inventory

### 🗂️ Raw Training Data
| File | Size | Molecules | Description | Location |
|------|------|-----------|-------------|----------|
| `xTB_ML_combined_train_data.csv` | 36MB | 314,088 | Combined xTB training data | External download |
| `xTB_ML_calcs_compiled.csv` | 17MB | - | Compiled xTB calculations | External download |
| `xTB_ML_dye_UVVis_comb.csv` | 5.6MB | - | Dye UV-Vis combined data | External download |

### 🎯 Active Learning Datasets  
| File | Size | Molecules | Description | Location |
|------|------|-----------|-------------|----------|
| `dt_tot_round1.csv` | 76MB | 655,199 | Complete AL candidate pool | External download |
| `test_emit.csv` | - | - | Emission test set | External download |
| `test_sens.csv` | - | - | Sensitivity test set | External download |

### 🧪 Test Sets by Strategy
| Strategy | Test Files | Location | Description |
|----------|-------------|----------|-------------|
| Pure Uncertainty | `test_set/` | External | Round-by-round test data |
| Target Property | `test_performance/` | External | Property-specific tests |
| Mixed Versions | Multiple test dirs | External | Strategy-specific evaluations |

## 🤖 Trained Models Inventory

### 📍 Model Structure
```
models/
├── pure_uncertainty/           # 1 strategy
│   └── round_9/               # Final training round
├── target_property/           # 4 variants
│   ├── v1_sep_batch/         ├── v2_sep_batch/
│   ├── v1_sep/               └── v2_sep/
├── mix_version/              # 8 variants  
│   ├── synthesizability_v2/   # 4 synthesis-aware sub-variants
│   │   ├── uncertainty_1_target_7_batch/  ├── uncertainty_2_target_6_batch/
│   │   ├── uncertainty_4_target_4_batch/  └── uncertainty_target_batch/
│   ├── uncertainty_1_target_7_batch/  ├── uncertainty_2_target_6_batch/
│   ├── uncertainty_4_target_4_batch/  └── uncertainty_target_nobatch_new/
├── expected_improvement/     # 3 variants
│   ├── v_001/  ├── v_003/   └── v_005/
├── threshold_test/          # 4 variants
│   ├── v_5_02/  ├── v_5_04/  ├── v_5_06/  └── v_5_08/
└── retro/                   # 1 strategy
    └── synthesis_aware/
```

### 🎯 Model Details

#### Pure Uncertainty Sampling
- **Rounds**: 1-9 (final: round_9) 
- **Architecture**: ChemProp ensemble (5 models)
- **Files**: `fold_0/`, `args.json`, `test_scores.csv`, logs
- **Performance**: Uncertainty-based molecular selection

#### Target Property Strategies (4 variants)
- **v1_sep**: Separate property targeting
- **v2_sep**: Enhanced separation method  
- **v1_sep_batch**: Batch version of v1
- **v2_sep_batch**: Batch version of v2
- **Properties**: xTB_S1, xTB_T1 energies

#### Mixed Strategies (8 variants)
- **synthesizability_v2**: Advanced synthesis feasibility (4 sub-variants)
  - `uncertainty_1_target_7_batch`: 1:7 uncertainty/property ratio with synthesis
  - `uncertainty_2_target_6_batch`: 2:6 ratio with synthesis
  - `uncertainty_4_target_4_batch`: 4:4 ratio with synthesis
  - `uncertainty_target_batch`: Standard batch with synthesis
- **General Variants**: Core uncertainty/diversity combinations
  - `uncertainty_1_target_7_batch`: 1:7 ratio (general)
  - `uncertainty_2_target_6_batch`: 2:6 ratio (general)
  - `uncertainty_4_target_4_batch`: 4:4 ratio (general)
  - `uncertainty_target_nobatch_new`: Non-batch uncertainty+target

#### Expected Improvement (3 variants)
- **v_001**: Threshold = 0.01
- **v_003**: Threshold = 0.03  
- **v_005**: Threshold = 0.05
- **Method**: Bayesian optimization inspired

#### Threshold Testing (4 variants)
- **v_5_02**: Threshold = 5.02
- **v_5_04**: Threshold = 5.04
- **v_5_06**: Threshold = 5.06
- **v_5_08**: Threshold = 5.08

#### Retrosynthesis Integration
- **Integration**: AiZynthFinder synthesis prediction
- **Files**: Model files + synthesis feasibility scores

## 📥 Download Instructions

### Option 1: Google Drive (Current)
```bash
# Raw data and preprocessed files
https://drive.google.com/drive/folders/1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat?usp=sharing
```

### Option 2: Script Download (Planned)
```bash
# Download all data
bash scripts/download_data.sh

# Download specific models  
bash scripts/download_models.sh --strategy pure_uncertainty
bash scripts/download_models.sh --strategy target_property --variant v2_sep_batch
```

### Option 3: External Repositories (Planned)
- **Hugging Face Hub**: Model weights and configurations
- **Zenodo**: Complete dataset archive with DOI
- **Local Backup**: `backup_2025-08-05_12-53-01/` folder

## 🔒 File Integrity & Verification

All downloaded files include:
- **SHA256 checksums** for integrity verification
- **File size validation**
- **Format consistency checks**

## 📊 Storage Summary

| Category | File Count | Total Size | Location |
|----------|------------|------------|----------|
| Raw Data | 10+ files | ~100MB | External |
| Processed Data | 5+ files | ~150MB | External |  
| Models | 22+ variants | ~600MB | External |
| Code | 100+ files | ~5MB | GitHub repo |
| **Total** | **140+ files** | **~855MB** | **Mixed** |

**Repository Size**: ~10MB (code only)
**Full Project Size**: ~855MB (with all data/models)

## 🌐 External Storage Links

### 📁 Google Drive (Primary)
- **Data & Models Folder**: `https://drive.google.com/drive/folders/1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat`
- 公开子目录示例：
  - `data/`：所有 CSV、原始 & 处理后数据
  - `models/`：各策略压缩包（例如 `pure_uncertainty_round9.tar.gz`）

### 📚 Zenodo Archive (DOI Pending)
- **Complete Project Archive**: `https://doi.org/10.5281/zenodo.XXXXXXX`
- **Model Weights Collection**: `https://zenodo.org/record/XXXXXXX`
- **Training Datasets**: `https://zenodo.org/record/XXXXXXX`

### 📁 Google Drive (Current Active)
- **Data Folder**: `https://drive.google.com/drive/folders/1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat`
- **Model Checkpoints**: Direct download via scripts (recommended)

### 🔗 Quick Download Commands
```bash
# Download entire Google Drive folder via gdown (requires gdown >=4.6)
# Install: pip install gdown

gdown --folder https://drive.google.com/drive/folders/1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat -O ./downloads/

# Example: download only pure_uncertainty models
gdown --id FILE_ID_PURE_UNCERTAINTY -O pure_uncertainty_models.tar.gz

# Extract
mkdir -p models/pure_uncertainty
 tar -xzf pure_uncertainty_models.tar.gz -C models/pure_uncertainty/
```

### 📊 Model Performance Summary
| Strategy | Variants | Best Performance | Model Size | Download Priority |
|----------|----------|------------------|------------|-------------------|
| Pure Uncertainty | 1 | Round 9 | ~50MB | High |
| Target Property | 4 | v2_sep_batch | ~200MB | High |
| **Synthesizability v2** | **4** | **uncertainty_4_target_4_batch** | **~150MB** | **Very High** |
| Expected Improvement | 3 | v_003 | ~120MB | Medium |
| Threshold Testing | 4 | v_5_06 | ~160MB | Medium |
| Retrosynthesis | 1 | synthesis_aware | ~80MB | High |

### 🔒 Data Integrity & Checksums
All downloadable files include SHA256 verification:
```bash
# Verify downloaded files
sha256sum -c checksums.txt
```

Checksum file (`checksums.txt`) available at each storage location.