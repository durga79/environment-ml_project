"""
Google Colab Setup Cell - Copy this into your notebook!
"""

setup_cell_code = '''
# ============================================
# 🚀 GOOGLE COLAB SETUP - Run this first!
# ============================================
import os
import sys

# Clone repository
if not os.path.exists('environment-ml_project'):
    !git clone https://github.com/durga79/environment-ml_project.git
    os.chdir('environment-ml_project')
else:
    os.chdir('environment-ml_project')

# Create necessary directories (IMPORTANT!)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

# Install dependencies
print("📦 Installing dependencies...")
!pip install -q -r requirements.txt

# Add src to path
sys.path.append('src')
%matplotlib inline

# Fix path helper function
def get_fig_path(filename):
    """Get correct path for saving figures in Colab"""
    return os.path.join('results', 'figures', filename)

def get_data_path(filename):
    """Get correct path for datasets in Colab"""
    return os.path.join('datasets', filename)

print("✅ Ready! Working in:", os.getcwd())
print("✅ Directories created!")
print("✅ Use get_fig_path('filename.png') for saving figures")
'''

print(setup_cell_code)
