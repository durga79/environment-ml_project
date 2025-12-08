# 🚀 Google Colab - Easy Setup Guide

## Method 1: Direct Colab Links (Easiest!)

You can open notebooks directly in Colab by replacing the GitHub URL pattern. Here's how:

### Step 1: Upload to GitHub (Recommended)

1. Create a GitHub repository
2. Upload the `environmental_ml_project` folder
3. Your notebooks will be at: `https://github.com/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/01_*.ipynb`

### Step 2: Convert to Colab Link

Replace:
```
https://github.com/USERNAME/REPO/blob/main/PATH/notebook.ipynb
```

With:
```
https://colab.research.google.com/github/USERNAME/REPO/blob/main/PATH/notebook.ipynb
```

### Example:
```
https://colab.research.google.com/github/yourusername/environmental_ml_project/blob/main/notebooks/01_data_collection_preprocessing.ipynb
```

---

## Method 2: Colab Badge (For README)

Add these badges to your README to open directly in Colab:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/01_data_collection_preprocessing.ipynb)
```

---

## Method 3: Google Drive (No GitHub Needed)

### One-Time Setup:

1. **Upload project to Google Drive:**
   - Upload entire `environmental_ml_project` folder to your Drive
   - Note the path (e.g., `MyDrive/environmental_ml_project/`)

2. **Open Colab:** https://colab.research.google.com

3. **Create a setup notebook** with this code:

```python
# ==================================================
# COLAB SETUP CELL - Run this first in every notebook
# ==================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/environmental_ml_project')

# Install dependencies (run once)
!pip install -q -r requirements.txt

# Add src to path
import sys
if '../src' not in sys.path:
    sys.path.append('src')

print("✅ Colab setup complete!")
print(f"📂 Working directory: {os.getcwd()}")
```

4. **Copy notebook content:**
   - Open the `.ipynb` file from your Drive in Colab
   - Add the setup cell at the beginning
   - Run all cells

---

## Method 4: Colab Chrome Extension

### Install Extension:
1. Install **"Open in Colab"** Chrome extension
2. Visit any `.ipynb` file on GitHub
3. Click the Colab icon in your browser
4. Notebook opens automatically in Colab!

Extension: https://chrome.google.com/webstore (search "Open in Colab")

---

## Method 5: Direct File Upload (Quick & Dirty)

In Colab, use the file upload widget:

```python
from google.colab import files
import io
import os

# Upload notebooks manually
uploaded = files.upload()

# Upload datasets
uploaded = files.upload()

# Install requirements
!pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap lime nltk
```

---

## Recommended Workflow for This Project

### Option A: GitHub (Best for Sharing)

1. Push project to GitHub
2. Share Colab links with team
3. Everyone can run directly from links

### Option B: Google Drive (Best for Solo Work)

1. Upload to Drive once
2. Create one setup notebook
3. Copy setup cell to each notebook
4. Run in Colab

---

## Pre-Made Colab Links (If You Upload to GitHub)

Replace `YOUR_USERNAME` with your GitHub username:

**Notebook 01 - Data Prep:**
```
https://colab.research.google.com/github/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/01_data_collection_preprocessing.ipynb
```

**Notebook 02 - Student 1:**
```
https://colab.research.google.com/github/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/02_student1_air_quality_analysis.ipynb
```

**Notebook 03 - Student 2:**
```
https://colab.research.google.com/github/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/03_student2_text_analysis.ipynb
```

**Notebook 04 - Integrated:**
```
https://colab.research.google.com/github/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/04_integrated_analysis.ipynb
```

**Notebook 05 - Visualization:**
```
https://colab.research.google.com/github/YOUR_USERNAME/environmental_ml_project/blob/main/notebooks/05_visualizations_presentation.ipynb
```

---

## Setup Cell Template for Each Notebook

Add this as the **first cell** in every notebook when using Colab:

```python
# ============================================
# 🚀 GOOGLE COLAB SETUP
# ============================================
# Run this cell first!

import os
import sys

# Check if in Colab
try:
    import google.colab
    IN_COLAB = True
    print("🔵 Running in Google Colab")
except:
    IN_COLAB = False
    print("🟢 Running locally")

if IN_COLAB:
    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    
    # Navigate to project (UPDATE THIS PATH!)
    project_path = '/content/drive/MyDrive/environmental_ml_project'
    os.chdir(project_path)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    !pip install -q -r requirements.txt
    
    # Enable plots
    %matplotlib inline
    
    print(f"✅ Setup complete! Working in: {os.getcwd()}")

# Add src to path (works for both local and Colab)
if 'src' not in sys.path:
    sys.path.append('src')
    
print("✅ All imports ready!")
```

---

## Tips for Colab

### Speed Up Execution:
```python
# Use GPU (if needed for deep learning)
# Runtime → Change runtime type → GPU
```

### Save Results Back to Drive:
```python
# Results automatically save to Drive if you're working from there
# No extra steps needed!
```

### Download Results Locally:
```python
from google.colab import files

# Download a specific file
files.download('results/final_report_data.json')

# Download all results
!zip -r results.zip results/
files.download('results.zip')
```

---

## Quick Start for Colab

**Fastest way to get started:**

1. Upload `environmental_ml_project` folder to Google Drive
2. Go to https://colab.research.google.com
3. File → Open notebook → Google Drive → Navigate to your notebook
4. Add setup cell (see template above)
5. Run all cells!

**That's it!** No complicated setup needed.

---

## Troubleshooting Colab

**Issue:** "No module named 'src'"  
**Fix:** Make sure setup cell has `sys.path.append('src')`

**Issue:** "File not found"  
**Fix:** Check `os.getcwd()` - make sure you're in the right directory

**Issue:** Drive disconnects  
**Fix:** Re-run the mount cell: `drive.mount('/content/drive', force_remount=True)`

**Issue:** Runtime crashes  
**Fix:** Reduce dataset sizes or use Runtime → Factory reset runtime

---

## For Presentation

**Create a master notebook** that imports and runs all analyses:

```python
# Master Presentation Notebook

# Setup (as above)
# ...

# Run all analyses
%run notebooks/01_data_collection_preprocessing.ipynb
%run notebooks/02_student1_air_quality_analysis.ipynb
%run notebooks/03_student2_text_analysis.ipynb
%run notebooks/04_integrated_analysis.ipynb
%run notebooks/05_visualizations_presentation.ipynb

print("🎉 All analyses complete!")
```

---

**Need help?** See TROUBLESHOOTING.md for more solutions!
