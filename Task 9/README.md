# Task 9 — Industrial Predictive Maintenance

## Objective
Predict whether a machine will fail using sensor measurements, with a focus on reducing false alarms.

## Dataset
Use the AI4I 2020 Predictive Maintenance Dataset from UCI or Kaggle. The commonly distributed CSV includes:
- Sensor inputs: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- Main target: `Machine failure`
- Failure-mode indicators: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

## Run
```bash
pip install -r requirements.txt
python predictive_maintenance.py --data ai4i2020.csv
```

## Method
The primary model is a Random Forest binary classifier for `Machine failure`. The script uses stratified train/validation/test partitions. It selects a probability threshold on the validation set to minimise false discovery rate (FDR), subject to a recall target of at least 0.50 where possible. It then refits on the training partition and evaluates once on the untouched test partition.

FDR is calculated as:
`False positives / (True positives + False positives)`, which equals `1 - precision`.

## Outputs
- `outputs/test_metrics.csv`
- `outputs/validation_threshold_analysis.csv`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.csv` and `.png`
- `outputs/predictive_maintenance_model.joblib`

## Important
The AI4I dataset is highly imbalanced, so accuracy alone can be misleading. Review precision, recall, F1-score, confusion matrix and FDR. The five failure-mode columns are excluded from model inputs because they are labels/outcomes, not sensor predictors. This script predicts overall failure, not the specific failure mode. A separate multi-label failure-mode model can be added if your supervisor requires predictions for TWF/HDF/PWF/OSF/RNF individually.


## Run the Streamlit web app

From this project folder, create and activate a virtual environment, then run:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Upload the dataset CSV through the sidebar/page uploader.
