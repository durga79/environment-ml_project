# Environmental Impact Analysis: Multi-Modal ML Study

## Project Overview

This project applies machine learning methods to analyze environmental sustainability through three interconnected datasets for a 3-student team project.

### Team Structure
- **Student 1**: Air Quality Analysis (Random Forest, XGBoost)
- **Student 2**: Climate Text Sentiment (Logistic Regression, SVM)
- **Student 3**: Water Quality Safety (Decision Tree, Gradient Boosting)

### Research Questions

1. Can we accurately predict air quality index categories using environmental sensor data?
2. How effectively can NLP methods classify climate policy sentiment from text?
3. Can machine learning classify water safety categories from water quality measurements?
4. What relationships exist between air quality, water quality, and environmental policy discourse?

## Project Structure

```
environmental_ml_project/
├── notebooks/
│   └── COMPLETE_ANALYSIS_ALL_STUDENTS.ipynb  # Single comprehensive notebook
├── src/                                       # Python modules
│   ├── data_generator.py                      # Generate datasets
│   ├── data_preprocessing.py                  # Data cleaning
│   ├── models.py                              # ML model training
│   ├── evaluation.py                          # Performance metrics
│   └── interpretability.py                    # SHAP & LIME
├── datasets/                                  # Generated during runtime
│   ├── air_quality_data.csv
│   ├── climate_text_data.csv
│   └── water_quality_data.csv
├── results/                                   # Generated during runtime
│   ├── figures/                               # Visualizations
│   └── metrics/                               # Performance metrics
├── requirements.txt                           # Python dependencies
├── conference_101719.tex                      # LaTeX report template
├── README.md                                  # This file
└── QUICK_START.md                             # Quick setup guide
```

## Quick Setup & Run

### Option 1: Google Colab (Recommended for Presentation)

**Perfect for the live demo!** No installation needed.

1. Upload the project folder to Google Drive
2. Open `COMPLETE_ANALYSIS_ALL_STUDENTS.ipynb` in Colab
3. Run Cell 1 (installs all packages automatically)
4. Run all cells (Runtime → Run all)

**📖 See `GOOGLE_COLAB_GUIDE.md` for detailed instructions**

**Estimated Runtime**: 20-25 minutes

---

### Option 2: Local Installation

**Prerequisites**: Python 3.8+, Jupyter Notebook

```bash
# 1. Navigate to project directory
cd environmental_ml_project

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Start Jupyter
jupyter notebook

# 6. Open: notebooks/COMPLETE_ANALYSIS_ALL_STUDENTS.ipynb
# 7. Click: "Kernel" → "Restart & Run All"
```

**Estimated Runtime**: 15-20 minutes

## What's Included

### Datasets (Auto-generated)
1. **Air Quality Data**: 12,000 rows × 16 columns
   - Features: Temperature, humidity, PM2.5, PM10, NO2, CO, O3, SO2, etc.
   - Target: AQI category (Good, Moderate, Unhealthy, Hazardous)

2. **Climate Text Data**: 12,000 rows × 3 columns
   - Features: Policy/news text (avg. 50 words)
   - Target: Sentiment (Positive, Neutral, Negative)

3. **Water Quality Data**: 12,000 rows × 17 columns
   - Features: pH, dissolved oxygen, turbidity, BOD, COD, coliforms, etc.
   - Target: Safety category (Safe, Moderate, Unsafe, Highly Unsafe)

### Machine Learning Methods

| Student | Dataset | Models | Interpretability |
|---------|---------|--------|-----------------|
| 1 | Air Quality | Random Forest, XGBoost | SHAP |
| 2 | Climate Text | Logistic Regression, SVM | LIME |
| 3 | Water Quality | Decision Tree, Gradient Boosting | SHAP |

### Methodology

This project follows the **CRISP-DM** methodology:
1. **Business Understanding**: Environmental impact assessment
2. **Data Understanding**: Exploratory data analysis
3. **Data Preparation**: Cleaning, preprocessing, feature engineering
4. **Modeling**: Train 6 ML models across 3 datasets
5. **Evaluation**: Multiple metrics (Accuracy, F1, Cohen's Kappa, etc.)
6. **Deployment**: Interpretability analysis with SHAP & LIME

## Key Features

✅ **Complete Analysis**: All 3 students' work in one streamlined notebook  
✅ **6 ML Models**: 2 models per student (Random Forest, XGBoost, Logistic Regression, SVM, Decision Tree, Gradient Boosting)  
✅ **Interpretability**: SHAP for tree models, LIME for text models  
✅ **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa  
✅ **Cross-Validation**: Stratified K-Fold for robust evaluation  
✅ **Visualizations**: Confusion matrices, feature importance, performance comparisons  
✅ **Integrated Analysis**: Multi-modal insights across all datasets  

## Performance Metrics

Each model is evaluated using:
- **Classification**: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa
- **Cross-Validation**: 5-fold stratified CV with mean ± std
- **Visualization**: Confusion matrices, ROC curves (where applicable)
- **Feature Importance**: SHAP values, LIME explanations

## Deliverables

1. ✅ **Jupyter Notebook**: `COMPLETE_ANALYSIS_ALL_STUDENTS.ipynb`
2. ✅ **LaTeX Report Template**: `conference_101719.tex` (IEEE format)
3. ✅ **Python Code**: Modular and reusable (`src/` directory)
4. ✅ **Generated Datasets**: All CSV files in `datasets/`
5. ✅ **Results**: Figures and metrics in `results/`

## Presentation Guide (10 minutes)

**Suggested Time Allocation**:
- **Introduction** (1 min): Problem statement, research questions
- **Student 1** (2.5 min): Air Quality - data, models (RF, XGBoost), SHAP
- **Student 2** (2.5 min): Climate Text - data, models (LR, SVM), LIME
- **Student 3** (2.5 min): Water Quality - data, models (DT, GB), SHAP
- **Integrated Analysis** (1 min): Cross-dataset insights
- **Conclusion** (0.5 min): Key findings, future work

## Requirements Met

✅ 3 Students, 3 Datasets (one per student)  
✅ 10,000+ rows per dataset  
✅ Text data (Climate Text) + Numeric data (Air & Water Quality)  
✅ 2 methods per student (6 total)  
✅ Interpretability (SHAP + LIME)  
✅ CRISP-DM methodology  
✅ Multiple performance metrics  
✅ Literature review capability (in LaTeX template)  
✅ Collaborative analysis  

## Dependencies

All required packages are in `requirements.txt`:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- nltk, wordcloud
- shap, lime

## Troubleshooting

**Issue**: ModuleNotFoundError  
**Solution**: Make sure you've installed all dependencies: `pip install -r requirements.txt`

**Issue**: NLTK data not found  
**Solution**: Run: `python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

**Issue**: Notebook cells fail  
**Solution**: Restart kernel and run all cells from the beginning

## Authors

- Student 1: John Smith (D12345678) - Air Quality Analysis
- Student 2: Jane Doe (D87654321) - Climate Text Sentiment
- Student 3: Bob Johnson (D11223344) - Water Quality Safety

## License

This is an academic project for educational purposes.

---

**For detailed setup instructions, see `QUICK_START.md`**
