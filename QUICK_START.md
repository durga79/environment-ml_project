# 🚀 QUICK START GUIDE

## Environmental ML Project - Get Running in 5 Minutes

---

## Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab

---

## Installation (2 minutes)

```bash
cd environmental_ml_project
pip install -r requirements.txt
```

---

## Execution (5 steps, ~2 hours total)

### Step 1: Verify ⚡ (30 seconds)
```bash
./verify_project.sh
```

### Step 2: Data Generation 📊 (10 minutes)
```bash
jupyter notebook notebooks/01_data_collection_preprocessing.ipynb
# Run All Cells
```

### Step 3: Student 1 Analysis 🌫️ (20 minutes)
```bash
jupyter notebook notebooks/02_student1_air_quality_analysis.ipynb
# Run All Cells
```

### Step 4: Student 2 Analysis 📝 (18 minutes)
```bash
jupyter notebook notebooks/03_student2_text_analysis.ipynb
# Run All Cells
```

### Step 5: Final Analysis 🎯 (13 minutes)
```bash
# Integrated analysis
jupyter notebook notebooks/04_integrated_analysis.ipynb
# Run All Cells

# Visualizations
jupyter notebook notebooks/05_visualizations_presentation.ipynb
# Run All Cells
```

---

## Google Colab (Alternative)

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate and Install
%cd /content/drive/MyDrive/environmental_ml_project
!pip install -r requirements.txt

# Cell 3+: Run notebooks in order
```

---

## Expected Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 85-90% | 0.82-0.88 |
| XGBoost | 86-91% | 0.83-0.89 |
| Logistic Regression | 80-85% | 0.78-0.83 |
| SVM | 81-86% | 0.79-0.84 |

---

## Output Files

After execution, you'll have:
- 3 CSV datasets in `datasets/`
- 40+ visualizations in `results/figures/`
- JSON metrics in `results/metrics/`
- 2 trained model files in `results/`

---

## Troubleshooting

**Import error?**  
→ Check first cell has `sys.path.append('../src')`

**No data?**  
→ Run notebook 01 first

**Too slow?**  
→ Reduce sample sizes in generators

**Plotting issues in Colab?**  
→ Add `%matplotlib inline` at top

---

## Documentation

- **README.md** - Project overview
- **EXECUTION_GUIDE.md** - Detailed instructions
- **PROJECT_SUMMARY.md** - Complete summary
- **PROJECT_REPORT_STRUCTURE.md** - Report writing guide
- **COMPLETION_REPORT.md** - Technical specifications

---

## Key Commands

```bash
# Verify project
./verify_project.sh

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Create submission ZIP
cd /tmp
zip -r environmental_ml_project.zip environmental_ml_project/ \
  -x "*.pyc" "*__pycache__*"
```

---

## Submission Checklist

- [ ] All 5 notebooks executed
- [ ] Results reviewed
- [ ] IEEE report written (8-10 pages)
- [ ] Presentation prepared (10 min)
- [ ] ZIP file created
- [ ] Deadline: Dec 12, 2025

---

**Need Help?**  
See EXECUTION_GUIDE.md for detailed instructions.

**Good luck!** 🎓
