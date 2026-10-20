# AI Cybersecurity Threat Detector

An AI-powered cybersecurity project that uses machine learning to analyse network traffic and identify potentially suspicious activity. The project includes a data preparation pipeline, a trained classification model, and an interactive Streamlit dashboard for exploring traffic and investigating predictions.

## Project Overview

The goal of this project is to demonstrate how machine learning can support network traffic analysis and cybersecurity threat detection.

The system processes network traffic records, prepares data for modelling, trains a classification model, and provides a dashboard for exploring predictions and reviewing potential threats.

## Features

- **Data preparation:** Removes duplicate records, handles missing values, and removes constant features.
- **Machine learning:** Uses a Random Forest classifier to classify network traffic.
- **Interactive dashboard:** Built with Streamlit.
- **Traffic exploration:** Inspect network traffic and model predictions.
- **CSV threat scanning:** Upload a CSV file and process records in batches.
- **Threat severity indicators:** Assign preliminary severity categories based on model scores.
- **Investigation summaries:** Summarise predicted benign and suspicious traffic.
- **Visual reports:** Generate an Excel summary report with charts.
- **Exportable results:** Download scan results for further analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib and Seaborn
- OpenPyXL

## Dataset

The project was developed using the `Tuesday-WorkingHours.pcap_ISCX.csv` network traffic dataset.

The original dataset contained 445,909 rows and 79 columns. During data preparation, 24,065 duplicate rows were removed, leaving 421,844 records. Ten constant features were removed, leaving 68 features.

The cleaned dataset contained the following labels:

| Class | Records |
|---|---:|
| BENIGN | 412,692 |
| ATTACK | 9,152 |
| **Total** | **421,844** |

These figures describe the prepared dataset, not the number of threats detected in every future scan.

## Project Structure

```text
AI_Cybersecurity_Threat_Detector/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets, trained model files, and reports may be stored locally and are excluded from the public repository by `.gitignore`.

## Getting Started

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username after publishing the repository.

```bash
git clone https://github.com/YOUR-USERNAME/AI_Cybersecurity_Threat_Detector.git
cd AI_Cybersecurity_Threat_Detector
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset and train the model

Place the dataset in the expected `data/raw/` directory, then run:

```bash
python src/inspect_data.py
python src/prepare_data.py
python src/train_model.py
```

These scripts may depend on the expected dataset filename and project paths.

### 5. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will provide a local URL, normally `http://localhost:8501`.

**Note:** The trained model and processed dataset are not included in the public repository by default. Run the preparation and training scripts first, and confirm that the generated files are saved where the dashboard expects them.

## Model Evaluation

The Random Forest model was evaluated using a held-out test split. Earlier results showed very high classification performance, including approximately 100% accuracy.

These results should be interpreted cautiously. Similar records in a random train-test split can make performance appear stronger than it will be on genuinely new network traffic. Independent validation and checks for data leakage are needed before making claims about real-world detection performance.

## Responsible Use and Limitations

- Predictions are indicators for investigation, not proof that malicious activity occurred.
- Severity categories are preliminary triage indicators based on model scores; they do not establish the actual impact of an incident.
- Model performance depends on the dataset, feature quality, and similarity between training and incoming traffic.
- The project is for educational and research purposes and is not a replacement for professional security monitoring or incident response.
- Do not upload confidential traffic captures, personal information, credentials, or sensitive datasets to a public repository.

## Future Improvements

- Validate the model against independent datasets.
- Investigate possible data leakage and class imbalance.
- Improve memory efficiency for large CSV scans.
- Add more detailed evaluation metrics and visualisations.
- Improve explainability so analysts can better understand predictions.

## Author

Developed as a machine learning and cybersecurity portfolio project to explore the application of data science to network traffic analysis.