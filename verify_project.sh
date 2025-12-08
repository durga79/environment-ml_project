#!/bin/bash

echo "=========================================="
echo "Environmental ML Project - Verification"
echo "=========================================="
echo ""

echo "1. Checking directory structure..."
dirs=("datasets" "notebooks" "src" "results" "results/figures" "results/metrics")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir exists"
    else
        echo "  ✗ $dir missing - creating..."
        mkdir -p "$dir"
    fi
done

echo ""
echo "2. Checking Python modules..."
modules=("src/data_generator.py" "src/data_preprocessing.py" "src/models.py" "src/evaluation.py" "src/interpretability.py")
for module in "${modules[@]}"; do
    if [ -f "$module" ]; then
        echo "  ✓ $module exists"
    else
        echo "  ✗ $module missing!"
    fi
done

echo ""
echo "3. Checking notebooks..."
notebooks=("notebooks/01_data_collection_preprocessing.ipynb" "notebooks/02_student1_air_quality_analysis.ipynb" "notebooks/03_student2_text_analysis.ipynb" "notebooks/04_integrated_analysis.ipynb" "notebooks/05_visualizations_presentation.ipynb")
for notebook in "${notebooks[@]}"; do
    if [ -f "$notebook" ]; then
        echo "  ✓ $notebook exists"
    else
        echo "  ✗ $notebook missing!"
    fi
done

echo ""
echo "4. Checking documentation..."
docs=("README.md" "EXECUTION_GUIDE.md" "PROJECT_REPORT_STRUCTURE.md" "requirements.txt")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✓ $doc exists"
    else
        echo "  ✗ $doc missing!"
    fi
done

echo ""
echo "5. Project Statistics:"
echo "  - Python modules: $(ls -1 src/*.py 2>/dev/null | wc -l)"
echo "  - Notebooks: $(ls -1 notebooks/*.ipynb 2>/dev/null | wc -l)"
echo "  - Documentation files: $(ls -1 *.md 2>/dev/null | wc -l)"

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: chmod +x verify_project.sh"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Execute notebooks in order (01 through 05)"
echo "4. Review EXECUTION_GUIDE.md for detailed instructions"
echo ""
