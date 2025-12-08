#!/usr/bin/env python3
"""
Quick fix for NLTK data download issues
Run this if you encounter NLTK-related errors
"""

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK data...")

try:
    nltk.download('punkt', quiet=False)
    print("✓ punkt downloaded")
except Exception as e:
    print(f"✗ punkt failed: {e}")

try:
    nltk.download('stopwords', quiet=False)
    print("✓ stopwords downloaded")
except Exception as e:
    print(f"✗ stopwords failed: {e}")

print("\nNOTE: If downloads fail, the code will use fallback methods.")
print("The project will still work correctly!")
