import json
import os

def add_colab_setup_to_notebook(notebook_path):
    """Add auto-setup cell to beginning of notebook"""
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Create setup cell
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================\n",
            "# 🚀 AUTO-SETUP: Works in Colab & Local\n",
            "# ============================================\n",
            "import os\n",
            "import sys\n",
            "\n",
            "# Detect if running in Colab\n",
            "try:\n",
            "    import google.colab\n",
            "    IN_COLAB = True\n",
            "    print(\"🔵 Running in Google Colab\")\n",
            "except:\n",
            "    IN_COLAB = False\n",
            "    print(\"🟢 Running locally\")\n",
            "\n",
            "if IN_COLAB:\n",
            "    # Clone repo if not exists\n",
            "    if not os.path.exists('environment-ml_project'):\n",
            "        print(\"📥 Cloning repository...\")\n",
            "        !git clone https://github.com/durga79/environment-ml_project.git\n",
            "    \n",
            "    # Change to project directory\n",
            "    if os.path.exists('environment-ml_project'):\n",
            "        os.chdir('environment-ml_project')\n",
            "    \n",
            "    # Create directories\n",
            "    os.makedirs('results/figures', exist_ok=True)\n",
            "    os.makedirs('results/metrics', exist_ok=True)\n",
            "    os.makedirs('datasets', exist_ok=True)\n",
            "    \n",
            "    # Install dependencies\n",
            "    print(\"📦 Installing dependencies...\")\n",
            "    !pip install -q -r requirements.txt\n",
            "    \n",
            "    # Add src to path\n",
            "    sys.path.insert(0, 'src')\n",
            "    \n",
            "    # Enable plots\n",
            "    %matplotlib inline\n",
            "else:\n",
            "    # Local setup - just add src to path\n",
            "    if '../src' not in sys.path:\n",
            "        sys.path.append('../src')\n",
            "\n",
            "print(f\"✅ Setup complete! Working in: {os.getcwd()}\")\n",
            "print(f\"✅ Python path includes: {[p for p in sys.path if 'src' in p]}\")"
        ]
    }
    
    # Insert setup cell at the beginning
    nb['cells'].insert(0, setup_cell)
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    print(f"✅ Updated: {notebook_path}")

# Update all notebooks
notebooks = [
    'notebooks/01_data_collection_preprocessing.ipynb',
    'notebooks/02_student1_air_quality_analysis.ipynb',
    'notebooks/03_student2_text_analysis.ipynb',
    'notebooks/04_integrated_analysis.ipynb',
    'notebooks/05_visualizations_presentation.ipynb'
]

for nb_path in notebooks:
    if os.path.exists(nb_path):
        add_colab_setup_to_notebook(nb_path)
    else:
        print(f"⚠️  Not found: {nb_path}")

print("\n✅ All notebooks updated with auto-setup!")
