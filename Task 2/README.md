# Task 2 — Customer Segmentation

## Objective
Group mall customers using annual income and spending score with K-Means clustering.

## Dataset
Download the Kaggle **Mall Customer Segmentation Data** CSV and place it in this folder. The script expects:
- `Annual Income (k$)`
- `Spending Score (1-100)`

It can also retain other columns such as `CustomerID`, `Gender`, and `Age` in the output.

## Run
```bash
pip install -r requirements.txt
python customer_segmentation.py --data Mall_Customers.csv
```

## Outputs
The `outputs/` directory contains:
- `customer_segments.csv` — customer records with K-Means cluster labels
- `customer_clusters.png` — 2D cluster visualisation
- `cluster_selection.png` — elbow and silhouette plots
- `cluster_profile.csv` — average income and spending score by cluster
- `customer_segments_with_dbscan.csv` — optional DBSCAN comparison

## Method
The two numeric features are standardised before clustering. Candidate values of k are compared using inertia and silhouette score. The script selects the k with the highest silhouette score among the tested values. Review the plots and cluster profiles before interpreting the segments.

## Important
Cluster numbers are arbitrary labels, not rankings. Describe each segment using its income and spending profile rather than assuming that Cluster 0 is inherently better than Cluster 1.


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
