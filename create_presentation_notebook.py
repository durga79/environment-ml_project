import json

# Create minimal presentation notebook
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells
cells = [
    # Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Environmental ML Project - 3 Students\n\n**10-Minute Presentation**\n\n- Student 1: Air Quality (Random Forest, XGBoost)\n- Student 2: Climate Text (Logistic Regression, SVM)\n- Student 3: Water Quality (Decision Tree, Gradient Boosting)\n\n---"]
    },
    
    # Setup
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": ["import sys\nsys.path.append('../src')\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport json\nimport warnings\nwarnings.filterwarnings('ignore')\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix\n\nimport xgboost as xgb\nimport shap\nfrom lime.lime_text import LimeTextExplainer\n\nfrom data_generator import AirQualityDataGenerator, ClimateTextDataGenerator, WaterQualityDataGenerator\n\nplt.style.use('seaborn-v0_8-darkgrid')\nsns.set_palette('husl')\n\nprint('Setup complete!')"]
    },
    
    # Data Generation
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Data Generation"]
    },
    {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": ["# Generate datasets\nair_gen = AirQualityDataGenerator(n_samples=15000, random_state=42)\nair_df = air_gen.generate_dataset()\n\ntext_gen = ClimateTextDataGenerator(n_samples=9500, random_state=42)\ntext_df = text_gen.generate_dataset()\n\nwater_gen = WaterQualityDataGenerator(n_samples=12000, random_state=42)\nwater_df = water_gen.generate_dataset()\n\nprint(f'Air Quality: {air_df.shape}')\nprint(f'Climate Text: {text_df.shape}')\nprint(f'Water Quality: {water_df.shape}')"]
    },
]

nb['cells'] = cells

with open('notebooks/PRESENTATION_NOTEBOOK_MINIMAL.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Created minimal presentation notebook base")
print("Now run this notebook to add complete sections")

