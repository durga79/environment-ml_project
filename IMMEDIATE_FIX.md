# 🚨 IMMEDIATE FIX - Run This Now!

## Quick Fix Cell (Copy & Paste This)

If you're getting `FileNotFoundError` when saving figures, run this cell **BEFORE** your plotting code:

```python
# 🔧 QUICK FIX - Run this cell first!
import os

# Create all necessary directories
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

print("✅ Directories created! Now you can save figures.")
```

## Or Replace plt.savefig with This:

Instead of:
```python
plt.savefig('results/figures/pollutants_dist.png', dpi=300)
```

Use:
```python
# Create directory if needed, then save
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/pollutants_dist.png', dpi=300)
```

## Or Use This Helper Function:

Add this at the top of your notebook (after imports):

```python
def safe_savefig(filename, **kwargs):
    """Save figure, creating directory if needed"""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, **kwargs)
    print(f"✅ Saved: {filename}")
```

Then use:
```python
safe_savefig('results/figures/pollutants_dist.png', dpi=300)
```

## ✅ Permanent Fix

The notebooks have been updated to automatically create directories before saving. After you push the updated notebooks to GitHub, this won't happen again!

**To update GitHub:**
```bash
cd /tmp/environmental_ml_project
git add notebooks/*.ipynb
git commit -m "Fix: Auto-create directories before saving figures"
git push
```

Then refresh your Colab notebook from GitHub!
