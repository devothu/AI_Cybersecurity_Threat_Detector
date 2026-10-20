
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. FILE LOCATIONS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "processed" / "traffic_cleaned.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_FILE = MODEL_DIR / "threat_detector.joblib"
REPORT_FILE = REPORT_DIR / "classification_report.txt"
MATRIX_FILE = REPORT_DIR / "confusion_matrix.png"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD THE CLEANED DATA
# --------------------------------------------------

print("=" * 60)
print("AI CYBERSECURITY THREAT DETECTOR")
print("STEP 5: MODEL TRAINING")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE, low_memory=False)

if "Label" not in df.columns:
    raise ValueError("The dataset must contain a Label column.")

df = df[df["Label"].isin(["BENIGN", "ATTACK"])].copy()

# Separate inputs from the target label
X = df.drop(columns=["Label"])
y = df["Label"]

# Keep only numeric features
X = X.select_dtypes(include="number")

# Convert infinite values to missing values
X = X.replace([float("inf"), float("-inf")], float("nan"))

# Remove completely empty features
X = X.dropna(axis=1, how="all")

# Remove constant features
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.shape[1] == 0:
    raise ValueError("No usable numeric features remain.")

if y.nunique() < 2:
    raise ValueError("Training requires both BENIGN and ATTACK records.")

print("\nDataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts())

print("\nFeatures selected:", X.shape[1])


# --------------------------------------------------
# 3. SPLIT INTO TRAINING AND TESTING DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 4. BUILD THE MACHINE-LEARNING PIPELINE
# --------------------------------------------------

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])


# --------------------------------------------------
# 5. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining Random Forest...")
print("This may take a few minutes.")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 6. EVALUATE ON UNSEEN TEST RECORDS
# --------------------------------------------------

print("\nEvaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)

attack_precision = precision_score(
    y_test,
    predictions,
    pos_label="ATTACK",
    zero_division=0,
)

attack_recall = recall_score(
    y_test,
    predictions,
    pos_label="ATTACK",
    zero_division=0,
)

attack_f1 = f1_score(
    y_test,
    predictions,
    pos_label="ATTACK",
    zero_division=0,
)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Attack precision:  {attack_precision:.4f}")
print(f"Attack recall:     {attack_recall:.4f}")
print(f"Attack F1-score:   {attack_f1:.4f}")

report = classification_report(
    y_test,
    predictions,
    labels=["BENIGN", "ATTACK"],
    zero_division=0,
)

print("\nClassification report:")
print(report)


# --------------------------------------------------
# 7. SAVE THE CLASSIFICATION REPORT
# --------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("AI CYBERSECURITY THREAT DETECTOR\n")
    file.write("Random Forest Evaluation\n\n")

    file.write(f"Dataset records: {len(df)}\n")
    file.write(f"Features: {X.shape[1]}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
    file.write(f"Attack precision: {attack_precision:.4f}\n")
    file.write(f"Attack recall: {attack_recall:.4f}\n")
    file.write(f"Attack F1-score: {attack_f1:.4f}\n\n")

    file.write(report)


# --------------------------------------------------
# 8. SAVE THE CONFUSION MATRIX
# --------------------------------------------------

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["BENIGN", "ATTACK"],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["BENIGN", "ATTACK"],
)

display.plot(values_format="d")
plt.title("Cybersecurity Threat Detector - Test Results")
plt.tight_layout()
plt.savefig(MATRIX_FILE, dpi=150)
plt.close()


# --------------------------------------------------
# 9. SAVE THE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "labels": ["BENIGN", "ATTACK"],
    },
    MODEL_FILE,
)

print("\nSaved trained model to:")
print(MODEL_FILE)

print("\nSaved evaluation report to:")
print(REPORT_FILE)

print("\nSaved confusion matrix to:")
print(MATRIX_FILE)

print("\nSTEP 5 COMPLETE!")
