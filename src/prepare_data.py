
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. DEFINE FILE LOCATIONS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Tuesday-WorkingHours.pcap_ISCX.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "traffic_cleaned.csv"

print("=" * 60)
print("AI CYBERSECURITY THREAT DETECTOR")
print("STEP 4: DATA CLEANING")
print("=" * 60)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE}\n"
        "Check the filename inside data/raw."
    )

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD THE DATASET
# --------------------------------------------------

print("\nLoading the original dataset...")
df = pd.read_csv(RAW_FILE, low_memory=False, encoding="utf-8-sig")

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# Clean column names
df.columns = df.columns.str.strip()

# Remove accidental duplicate column names, keeping the first
df = df.loc[:, ~df.columns.duplicated()].copy()

# Find the label column regardless of capitalization
label_column = next(
    (column for column in df.columns
     if column.lower() == "label"),
    None
)

if label_column is None:
    raise ValueError(
        "No Label column found. Check the dataset format."
    )

# Standardize the label values
df[label_column] = (
    df[label_column]
    .astype("string")
    .str.strip()
    .str.upper()
)

# --------------------------------------------------
# 3. REMOVE INVALID LABELS AND DUPLICATES
# --------------------------------------------------

before = len(df)

df = df.dropna(subset=[label_column])
df = df[df[label_column] != ""]

print("\nRows removed due to missing labels:", before - len(df))

before = len(df)
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", before - len(df))

# --------------------------------------------------
# 4. CREATE TWO TARGET CLASSES
# --------------------------------------------------

# BENIGN remains BENIGN.
# Every other labelled traffic category becomes ATTACK.
df["Label"] = np.where(
    df[label_column] == "BENIGN",
    "BENIGN",
    "ATTACK"
)

if label_column != "Label":
    df = df.drop(columns=[label_column])

print("\nBinary traffic labels:")
print(df["Label"].value_counts())

# --------------------------------------------------
# 5. REMOVE IDENTIFIERS THAT CAN CAUSE LEAKAGE
# --------------------------------------------------

# These fields identify individual flows or endpoints rather
# than providing generalizable numerical traffic measurements.
identifier_columns = [
    column for column in df.columns
    if column.strip().lower() in {
        "flow id",
        "source ip",
        "destination ip",
        "timestamp"
    }
]

if identifier_columns:
    df = df.drop(columns=identifier_columns)

print("\nIdentifier columns removed:", identifier_columns)

# --------------------------------------------------
# 6. CLEAN NUMERIC FEATURES
# --------------------------------------------------

feature_columns = [
    column for column in df.columns
    if column != "Label"
]

for column in feature_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Replace infinite values with missing values.
df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

# Remove features containing no usable numeric values.
empty_columns = [
    column for column in feature_columns
    if df[column].isna().all()
]

if empty_columns:
    df = df.drop(columns=empty_columns)
    print("Empty features removed:", empty_columns)

# Remove features that have the same value in every row.
feature_columns = [
    column for column in df.columns
    if column != "Label"
]

constant_columns = [
    column for column in feature_columns
    if df[column].nunique(dropna=True) <= 1
]

if constant_columns:
    df = df.drop(columns=constant_columns)
    print("Constant features removed:", constant_columns)

# Keep missing feature values for now.
# We will handle them properly during model training.

# --------------------------------------------------
# 7. SAVE THE PROCESSED DATA
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)

print("Cleaned rows:", len(df))
print("Remaining features:", len(df.columns) - 1)
print("Missing feature values:", int(
    df.drop(columns=["Label"]).isna().sum().sum()
))

print("\nFinal class distribution:")
print(df["Label"].value_counts())

print("\nProcessed dataset saved to:")
print(OUTPUT_FILE)
