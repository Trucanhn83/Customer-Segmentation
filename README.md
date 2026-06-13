# Customer Segmentation — Online Retail Analysis (Project 2)

End-to-end analytics on an online retail transactions dataset — from data cleaning and EDA to **RFM-based customer segmentation with KMeans**, plus a broad set of advanced techniques: predictive modeling, association rule mining, recommender systems, churn prediction, and more.

> Project for **DAB303 – Marketing Analytics** (Project 2).

---

## 1. Overview

The core objective is **customer segmentation** — dividing customers into actionable groups using **RFM (Recency, Frequency, Monetary)** analysis to support targeted marketing. The notebook also explores the dataset end-to-end with EDA, aggregation, visualization, and a range of advanced modeling tasks.

## 2. Data

- **Source:** `Sales_data.csv` (Online Retail transactions, `ISO-8859-1` encoding)
- **Size:** 541,909 rows × 8 columns
- **Columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- **Scale:** 4,070 unique products · 4,373 unique customers · United Kingdom dominates (3,951 unique customers and most revenue)

> ⚠️ `Sales_data.csv` is **not included** in this repo. Download the dataset and update the file path in the *How to Run* section.
> Note: ~135K rows have a missing `CustomerID` (filled with `0` as a placeholder), which limits customer-level models.

## 3. Project Structure

```
.
├── Project 2.ipynb               # Main notebook (code + analysis)
├── Project 2.html                # HTML export of the notebook
├── Customer Segmentation.pptx    # Results presentation (17 slides)
├── Customer Segmentation.pdf     # PDF version of the presentation
├── Sales_data.csv                # Dataset (add it yourself)
└── README.md
```

## 4. Workflow

**1. Data Preprocessing & Cleaning** — load data; handle missing values (`CustomerID` → `0`, `Description` → `"Unknown"`); convert `InvoiceDate` to datetime; add `TotalPrice = UnitPrice × Quantity`.

**2. Exploratory Data Analysis** — unique products & customers, top products by quantity, top country by customers, `TotalPrice` distribution (highly right-skewed).

**3. Data Aggregation** — total sales per country, highest-sales month, average unit price per product, total quantity per customer.

**4. Data Visualization** — sales by country (bar), sales trend over time (line, strong Nov–Dec seasonality), UnitPrice vs Quantity (scatter), correlation heatmap (Quantity–TotalPrice ≈ 0.89).

**5. Advanced Analysis** — outlier detection (box plots), initial RFM exploration, monthly trend of top-5 products.

**6. Feature Engineering** — extract `Year`/`Month`/`Day`/`Hour`; create `ReturnFlag` (1 if `Quantity < 0`).

**7. Customer Segmentation (KMeans)** — build the **RFM matrix**, normalize with `StandardScaler`, choose **k via the Elbow method (≈3 clusters)**, and interpret segments (e.g., recently-active low spenders vs lapsed groups).

**8. Predictive Analytics** — classification of `ReturnFlag` (RandomForest) and regression of `TotalPrice` (RandomForestRegressor); features chosen to avoid data leakage.

**9–13. Advanced Techniques**
- **Association Rule Mining** — Apriori for frequently-bought-together products; deeper rules filtered by high lift & confidence (market basket analysis).
- **Pareto Analysis (80/20)** — the ~20% of products/customers driving ~80% of revenue.
- **Time-Series Anomalies** — rolling mean ± 3σ (7-day and 30-day windows).
- **Recommender Systems** — popularity baseline, item-based collaborative filtering (cosine), ALS matrix factorization, and a **Neural Collaborative Filtering** model (Keras).

**14–16. Churn & Optimization** — churn labeling (Recency > 90 days), churn classification, plus hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`) and a `VotingClassifier` ensemble.

## 5. Key Results & Insights

- **Segmentation:** RFM + KMeans yields ~3 actionable segments; most customers are low-frequency, recent buyers, differentiated mainly by recency and spend.
- **Return classification:** RandomForest — **accuracy ≈ 0.998**, precision 0.94, recall 0.97, F1 0.96 (note: the dataset is highly imbalanced toward non-returns).
- **TotalPrice regression:** **R² ≈ 0.917** (MAE ≈ 0.002, RMSE ≈ 0.040).
- **Recommendations:** Neural Collaborative Filtering reaches **validation AUC ≈ 0.846**.
- **Churn prediction:** moderate performance (Class 1 recall ≈ 0.78, precision ≈ 0.55), constrained by the large share of missing `CustomerID`.
- **Business signals:** strong **Nov–Dec holiday seasonality**; UK dominates revenue; a small share of products/customers drives most revenue (Pareto).

## 6. Requirements

- Python 3.x

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend scipy tensorflow
```

## 7. How to Run

1. Place `Sales_data.csv` in the project folder.
2. Open `Project 2.ipynb` (Jupyter Notebook / JupyterLab / VS Code).
3. **Update the data path** in the CSV-reading cell — replace the original absolute path:

   ```python
   sales = pd.read_csv('Sales_data.csv', encoding='ISO-8859-1')   # instead of the original D:\... path
   ```

4. Run the cells in order from top to bottom.

## 8. Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` (KMeans, StandardScaler, RandomForest, GridSearch, VotingClassifier) · `mlxtend` (Apriori) · `scipy` · `tensorflow` / `keras` (Neural Collaborative Filtering)

