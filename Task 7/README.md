# Task 7 — Walmart Sales Forecasting

## Objective
Use historical weekly sales to predict sales in a later time period and plot actual versus predicted values.

## Dataset
Download the Kaggle Walmart Recruiting — Store Sales Forecasting `train.csv`. Expected columns include:
- `Date`
- `Weekly_Sales`
- `Store` and `Dept` (recommended to retain separate series)
- `IsHoliday` (optional)

## Run
```bash
pip install -r requirements.txt
python sales_forecasting.py --data train.csv
```
Optional single store/department:
```bash
python sales_forecasting.py --data train.csv --store 1 --department 1
```

## Method
The script creates calendar features and historical lag/rolling-average features. It uses a chronological 80/20 split, rather than randomly shuffling dates, to reduce time leakage. Random Forest and HistGradientBoosting regressors are compared using MAE and RMSE.

## Outputs
- `outputs/forecast_model_comparison.csv`
- `outputs/actual_vs_predicted_sales.png`
- `outputs/best_sales_model.joblib`

## Important limitations
Lag features are built from observed historical sales. The script evaluates a later holdout period; it does not recursively generate future weeks beyond the dataset. For a genuine future forecast, define the forecast horizon and generate future rows using known calendar variables and recursively updated predictions. Avoid using future sales values when creating predictors.


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
