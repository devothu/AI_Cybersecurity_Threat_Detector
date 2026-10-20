
from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Tuesday-WorkingHours.pcap_ISCX.csv"
)

print("=" * 55)
print("CYBERSECURITY DATA INSPECTION")
print("=" * 55)

if not DATA_FILE.exists():
    print("Dataset not found!")
    print("Expected location:", DATA_FILE)
    print("Download the CSV and place it in data/raw/.")
else:
    # Read only the first 5,000 rows initially
    df = pd.read_csv(DATA_FILE, nrows=5000, low_memory=False)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    print("\nDataset loaded successfully!")
    print("Rows inspected:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nFirst five records:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    if "Label" in df.columns:
        print("\nTraffic labels in the inspected sample:")
        print(df["Label"].value_counts())
    else:
        print("\nWARNING: No 'Label' column found.")
        print("Check the CSV column names.")
