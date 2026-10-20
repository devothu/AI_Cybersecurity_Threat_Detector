
# AI Cybersecurity Threat Detector
# First program: verify our Python environment

import sys
import pandas as pd
import numpy as np
import sklearn
import streamlit as st

print("=" * 45)
print(" AI CYBERSECURITY THREAT DETECTOR")
print("=" * 45)

print("Python version:", sys.version.split()[0])
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)

print("\nProject environment is working!")
print("Next: load and analyse cybersecurity data.")
