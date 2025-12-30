# Installation Guide

## System Requirements

- Python 3.8 or higher
- pip (Python package manager)
- 8GB RAM minimum
- GPU recommended (CUDA 11.8+) but not required

## Quick Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/InsiderThreatGNN.git
cd InsiderThreatGNN
```

### Step 2: Create Virtual Environment (Recommended)

#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Visit: https://www.cert.org/insider-threat-center/datasets/
2. Download "CERT Insider Threat Dataset v4.2" (or latest version)
3. Extract to: `data/raw/cert_r4.2/`

Your folder structure should look like:
```
data/raw/cert_r4.2/
├── users.csv
├── devices.csv
├── logins.csv
├── files.csv
├── emails.csv
├── web.csv
└── insiders.csv
```

### Step 5: Verify Installation

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test PyTorch Geometric
python -c "import torch_geometric; print('PyTorch Geometric: OK')"

# Test Pandas
python -c "import pandas; print('Pandas: OK')"

# Run full test
python src/gnn_models.py
```

---

## Detailed Installation

### For GPU Users (NVIDIA CUDA)

#### Check CUDA Installation

```bash
nvidia-smi
```

If CUDA is not installed:
1. Visit: https://developer.nvidia.com/cuda-downloads
2. Download and install CUDA 11.8+
3. Install cuDNN (optional but recommended)

#### Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Install PyTorch Geometric with CUDA

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

### For Apple Silicon (M1/M2/M3)

```bash
# PyTorch automatically uses Metal Performance Shaders
pip install torch torchvision torchaudio
pip install torch-geometric
```

### For CPU Only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch==2.0.0
```

Or for CUDA support:
```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

---

### Issue: "No module named 'torch_geometric'"

**Solution:**
```bash
pip install torch-geometric

# If above doesn't work:
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

---

### Issue: "Dataset not found at ./data/cert_r4.2"

**Solution:**
1. Check folder path: `data/raw/cert_r4.2/` exists
2. Verify CSV files are extracted:
   ```bash
   ls data/raw/cert_r4.2/
   ```
3. Expected files:
   - users.csv
   - devices.csv
   - logins.csv
   - insiders.csv

---

### Issue: "CUDA out of memory"

**Solution 1: Use CPU instead**
```python
# Edit src/gnn_models.py
device = torch.device("cpu")
```

**Solution 2: Reduce batch size or epochs**
```python
# In main():
trainer.fit(train_graphs, augmenter, epochs=50)  # Fewer epochs
```

---

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn==1.3.0
```

---

### Issue: "pip: command not found"

**Solution:**
```bash
# macOS
python3 -m pip install -r requirements.txt

# Windows
python -m pip install -r requirements.txt
```

---

### Issue: Virtual environment not activating

**Check activation:**
- **Linux/Mac:** You should see `(venv)` at the start of terminal line
- **Windows:** `(venv)` should appear in command prompt

**If not working:**
```bash
# Delete and recreate
rm -rf venv  # Linux/Mac
rmdir venv   # Windows

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

---

### Issue: "Permission denied" on Linux/Mac

**Solution:**
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

---

## Verify Everything Works

### Test 1: Import All Modules

```python
from src.gnn_models import (
    DataLoader, GraphBuilder, GNNModel, SelfSupervisedLearner,
    GraphAugmenter, Trainer, AnomalyDetector, Evaluator
)
print("✓ All imports successful!")
```

### Test 2: Check Data Folder

```bash
python -c "from pathlib import Path; print(Path('data/raw/cert_r4.2').exists())"
```

Expected output: `True`

### Test 3: Run Full Pipeline

```bash
python src/gnn_models.py
```

Expected output:
- "Using device: cuda" or "Using device: cpu"
- Training loss decreasing over epochs
- Final metrics around 0.96+ accuracy

---

## Requirements.txt Breakdown

```
torch==2.0.0                    # Deep learning framework
torch-geometric==2.3.0          # Graph neural networks
scikit-learn==1.3.0            # Machine learning utilities
pandas==2.0.0                  # Data manipulation
numpy==1.24.0                  # Numerical computing
matplotlib==3.7.0              # Plotting
seaborn==0.12.0               # Statistical visualization
```

---

## Performance Optimization

### GPU Setup
- Ensures faster training (10-50x speedup)
- Required for large graphs
- Check GPU availability: `nvidia-smi`

### CPU-Only Setup
- Works fine for learning
- Training takes longer (5-15 minutes)
- Use fewer epochs for quick testing

### Memory Optimization
- Reduce epochs: `trainer.fit(graphs, augmenter, epochs=50)`
- Use CPU mode: `device = torch.device("cpu")`
- Clear cache: `torch.cuda.empty_cache()`

---

## Next Steps

1. ✅ Complete installation
2. ✅ Download CERT dataset
3. ✅ Run: `python src/gnn_models.py`
4. ✅ Check [docs/USAGE.md](../docs/USAGE.md) for usage examples
5. ✅ Review [README.md](../README.md) for project overview

---

## Getting Help

**For installation issues:**
1. Check this document's "Troubleshooting" section
2. Visit: https://pytorch.org/get-started/locally/
3. Visit: https://pytorch-geometric.readthedocs.io/

**For code issues:**
1. Check [docs/USAGE.md](../docs/USAGE.md)
2. Open an issue on GitHub
3. Review [README.md](../README.md)

---

**Last Updated:** December 30, 2025  
**Status:** ✅ Verified & Working
