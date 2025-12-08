# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### NLTK Download Errors

**Error:** `LookupError: Resource stopwords not found`

**Solutions:**

1. **Option 1 - Run the fix script:**
   ```bash
   python fix_nltk.py
   ```

2. **Option 2 - Manual download:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Option 3 - No action needed!**
   The code automatically uses fallback stopword lists if NLTK data is unavailable.
   Your project will work fine without NLTK data.

---

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
- Ensure the first code cell in every notebook has:
  ```python
  import sys
  sys.path.append('../src')
  ```

---

### Missing Dataset Errors

**Error:** `FileNotFoundError: datasets/air_quality_data.csv`

**Solution:**
- Run notebook `01_data_collection_preprocessing.ipynb` first
- This generates all required datasets

---

### Slow Execution

**Issue:** Notebooks taking too long to run

**Solutions:**

1. **Reduce dataset sizes** in data generators:
   ```python
   # In notebook 01, change:
   air_quality_generator = AirQualityDataGenerator(n_samples=10000)  # instead of 15000
   text_generator = ClimateTextDataGenerator(n_samples=7000)  # instead of 9500
   ```

2. **Reduce SHAP sample size** in notebook 02:
   ```python
   # Change:
   shap_values_rf = shap_rf.calculate_shap_values(X_test[:200])  # instead of 500
   ```

---

### Google Colab Issues

**Issue:** Plotting not working

**Solution:**
Add at the top of each notebook:
```python
%matplotlib inline
```

**Issue:** Drive mount errors

**Solution:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

### Memory Errors

**Error:** `MemoryError` or kernel crashes

**Solutions:**

1. **Restart kernel** and run cells individually
2. **Reduce dataset sizes** (see "Slow Execution" above)
3. **Use smaller cross-validation folds:**
   ```python
   cv = StratifiedKFold(n_splits=3)  # instead of 5
   ```

---

### Package Version Conflicts

**Issue:** Incompatible package versions

**Solution:**
```bash
# Create a fresh virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Wordcloud Issues

**Error:** Wordcloud rendering problems

**Solution:**
```bash
pip install --upgrade wordcloud matplotlib
```

---

### SHAP Plotting Errors

**Error:** SHAP plots not displaying

**Solution:**
```python
# Add before SHAP plotting:
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
```

---

### Still Having Issues?

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Verify project structure:**
   ```bash
   ./verify_project.sh
   ```

3. **Reinstall dependencies:**
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

4. **Review error messages:**
   - Check the last cell that ran successfully
   - Look for specific error messages
   - Consult EXECUTION_GUIDE.md for detailed steps

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| NLTK errors | `python fix_nltk.py` or ignore (fallback works) |
| Import errors | Add `sys.path.append('../src')` |
| No data | Run notebook 01 first |
| Too slow | Reduce sample sizes in generators |
| Memory error | Reduce dataset/CV sizes |
| Colab plotting | Add `%matplotlib inline` |

---

**Most issues can be resolved by:**
1. Running notebooks in order (01-05)
2. Ensuring all dependencies are installed
3. Using the verification script

**The NLTK issue you encountered is already handled** - the code will use built-in stopwords if NLTK fails to download!
