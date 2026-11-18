# Task 4 — Loan Approval Prediction

## Objective
Predict whether a loan application will be approved, and compare Logistic Regression with a Decision Tree.

## Dataset
Download the Kaggle Loan Approval Prediction dataset. This script expects a target column named `Loan_Status` (commonly `Y`/`N`). If your target has another name, pass it using `--target`.

## Run
```bash
pip install -r requirements.txt
python loan_approval.py --data train.csv
```
For a differently named target:
```bash
python loan_approval.py --data loan_data.csv --target loan_approved
```

## What the script does
1. Removes rows without a target label.
2. Excludes common identifier columns.
3. Imputes missing numeric values with the median and categorical values with the most frequent value.
4. Scales numeric features and one-hot encodes categorical features.
5. Trains Logistic Regression and Decision Tree models.
6. Compares accuracy, precision, recall and F1-score and saves confusion matrices.

## Outputs
`outputs/model_comparison.csv`, two confusion-matrix images, and `best_loan_approval_model.joblib`.

## Important
Loan decisions are high-impact decisions. This is an educational internship model, not a production lending tool. Check for data leakage, bias, class imbalance and fairness across relevant groups. Do not use protected attributes to make real lending decisions. A model score alone should never be treated as a real approval decision.


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
