# 🚀 Quick Start Guide

## Setup (5 minutes)

### Step 1: Install Dependencies

```bash
cd /home/durga/environmental_ml_project

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 2: Launch Jupyter

```bash
jupyter notebook
```

### Step 3: Run the Analysis

1. Open `notebooks/COMPLETE_ANALYSIS_ALL_STUDENTS.ipynb`
2. Click **"Kernel"** → **"Restart & Run All"**
3. Wait 15-20 minutes for completion

That's it! ✅

---

## Project Structure

```
environmental_ml_project/
├── notebooks/
│   └── COMPLETE_ANALYSIS_ALL_STUDENTS.ipynb  ← Run this!
├── src/                    ← Python modules (auto-loaded)
├── datasets/               ← Generated automatically
├── results/                ← Figures & metrics
├── requirements.txt        ← Dependencies
└── conference_101719.tex   ← LaTeX report template
```

---

## What Gets Generated

When you run the notebook, it automatically:
1. ✅ Generates 3 datasets (12,000 rows each)
2. ✅ Trains 6 ML models (2 per student)
3. ✅ Evaluates with multiple metrics
4. ✅ Creates SHAP & LIME interpretability
5. ✅ Generates visualizations
6. ✅ Saves results to `results/` folder

---

## Notebook Structure

The single notebook contains:

### 🎯 Student 1: Air Quality (Random Forest, XGBoost)
- Features: PM2.5, temperature, humidity, pollutants
- Target: AQI category (Good/Moderate/Unhealthy/Hazardous)
- Interpretability: SHAP

### 🎯 Student 2: Climate Text (Logistic Regression, SVM)
- Features: Policy/news text (TF-IDF)
- Target: Sentiment (Positive/Neutral/Negative)
- Interpretability: LIME

### 🎯 Student 3: Water Quality (Decision Tree, Gradient Boosting)
- Features: pH, dissolved oxygen, BOD, COD, coliforms
- Target: Safety category (Safe/Moderate/Unsafe/Highly Unsafe)
- Interpretability: SHAP

### 🎯 Integrated Analysis
- Cross-dataset insights
- Model comparisons
- Visualizations

---

## For Presentation (10 minutes)

The notebook is designed for a 10-minute live demo:

**Timeline**:
- **00:00-01:00**: Introduction & Research Questions
- **01:00-03:30**: Student 1 - Air Quality Analysis
- **03:30-06:00**: Student 2 - Climate Text Analysis
- **06:00-08:30**: Student 3 - Water Quality Analysis
- **08:30-09:30**: Integrated Analysis & Comparisons
- **09:30-10:00**: Conclusions & Q&A

**Presentation Tips**:
1. Run the entire notebook before presenting
2. Each student presents their own section
3. Show key visualizations (confusion matrices, SHAP plots)
4. Highlight model performance comparisons

---

## Troubleshooting

**Q: ModuleNotFoundError**  
A: Run `pip install -r requirements.txt`

**Q: NLTK data missing**  
A: Run `python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

**Q: Cells fail to run**  
A: Restart kernel and run all cells from the beginning

**Q: Out of memory**  
A: Reduce dataset size in data generation cells (change `n_samples=12000` to `n_samples=8000`)

---

## Requirements Met ✅

This project meets all assignment requirements:

- ✅ 3 students, 3 datasets
- ✅ 10,000+ rows per dataset
- ✅ Text + structured numeric data
- ✅ 2 ML methods per student (6 total)
- ✅ Interpretability (SHAP + LIME)
- ✅ CRISP-DM methodology
- ✅ Multiple performance metrics
- ✅ Collaborative analysis

---

## Next Steps

1. ✅ Run the notebook
2. ✅ Review results in `results/` folder
3. ✅ Customize LaTeX report (`conference_101719.tex`)
4. ✅ Practice presentation
5. ✅ Prepare for Q&A

**Good luck with your presentation! 🎉**
