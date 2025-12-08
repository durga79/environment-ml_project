# Environmental Impact Analysis: Multi-Modal ML Study

## Project Overview

This project applies machine learning methods to analyze environmental sustainability through two interconnected datasets:
1. **Air Quality Data** (Student 1): Numeric/categorical data for pollution prediction
2. **Climate Policy Text Data** (Student 2): Text analysis for policy impact classification
3. **Integrated Dataset**: Multi-modal analysis combining both sources

## Research Questions

1. Can we accurately predict air quality index categories using environmental sensor data?
2. How effectively can NLP methods classify climate policy documents by impact sentiment?
3. What relationships exist between air quality trends and environmental policy discourse?

## Project Structure

```
environmental_ml_project/
├── datasets/                          # All datasets used
│   ├── air_quality_data.csv
│   ├── climate_news_text.csv
│   └── integrated_multimodal.csv
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 01_data_collection_preprocessing.ipynb
│   ├── 02_student1_air_quality_analysis.ipynb
│   ├── 03_student2_text_analysis.ipynb
│   ├── 04_integrated_analysis.ipynb
│   └── 05_visualizations_presentation.ipynb
├── src/                              # Python modules
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── interpretability.py
├── results/                          # Output files
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

## Setup Instructions

### Option 1: Google Colab (Recommended for Presentation)

1. Upload the entire project folder to Google Drive
2. Open Google Colab
3. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. Navigate to the project directory:
```python
%cd /content/drive/MyDrive/environmental_ml_project
```
5. Install requirements:
```python
!pip install -r requirements.txt
```
6. Run notebooks in order (01 through 05)

### Option 2: Local Setup

```bash
# Clone or download the project
cd environmental_ml_project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding**: Define environmental research objectives
2. **Data Understanding**: Exploratory analysis, statistics, visualizations
3. **Data Preparation**: Cleaning, transformation, feature engineering
4. **Modeling**: Apply multiple ML methods with cross-validation
5. **Evaluation**: Multi-metric assessment and interpretability analysis
6. **Deployment**: Reproducible notebooks and documentation

## Methods Applied

### Student 1: Air Quality Prediction
- **Random Forest Classifier**: Ensemble method for AQI prediction
- **XGBoost**: Gradient boosting for improved accuracy
- **Interpretability**: SHAP values for feature importance analysis

### Student 2: Climate Text Classification
- **TF-IDF + Logistic Regression**: Traditional NLP approach
- **BERT-based Classification**: Transformer model for text understanding
- **Interpretability**: LIME for text explanation

## Performance Metrics

- **Classification**: Accuracy, F1-Score, Cohen's Kappa, Precision, Recall, AUC-ROC
- **Regression** (if applicable): RMSE, MAE, MAPE, R²

## Execution Order

Run notebooks in the following sequence:

1. `01_data_collection_preprocessing.ipynb` - Download and prepare all datasets
2. `02_student1_air_quality_analysis.ipynb` - Student 1's analysis
3. `03_student2_text_analysis.ipynb` - Student 2's analysis
4. `04_integrated_analysis.ipynb` - Combined multi-modal analysis
5. `05_visualizations_presentation.ipynb` - Generate presentation materials

## Data Sources

- **Air Quality Data**: European Environment Agency (EEA) Open Data Portal
- **Climate Text Data**: EU Climate Policy Documents and Environmental Reports
- All sources are publicly available and properly cited in the report

## Key Findings

Results and findings are detailed in:
- Individual student notebooks (02 and 03)
- Integrated analysis (04)
- Final IEEE-format report (PDF)

## Authors

- Student 1: Air Quality Analysis
- Student 2: Climate Text Analysis

## Academic Integrity

All code, data sources, and external libraries are properly referenced. No AI coding assistants were used in the development of this project.

## License

This project is for academic purposes only (Data Mining & Machine Learning Module, 2025).

