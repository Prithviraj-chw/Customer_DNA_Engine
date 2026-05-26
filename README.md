# 🧬 Customer DNA Engine

> Unsupervised ML pipeline that discovers hidden behavioral personas inside e-commerce transaction data — and recommends targeted promotions for each segment.

---

## What it does

Takes raw retail transaction data and automatically groups customers into distinct behavioral segments using **RFM feature engineering** + **K-Means Clustering** — no labels, no manual rules. A Streamlit frontend lets you input any customer's RFM values and instantly get their segment + a targeted promotion recommendation.

**The 3 discovered segments:**

| Segment | Profile | Promotion |
|---|---|---|
| 😶 Lost Customer | High recency, low frequency, low spend | "We miss you — 25% off to come back" |
| 🤝 Loyal Regular | Balanced RFM, core customer base | "Early access to new arrivals" |
| 🐋 VIP Wholesaler | Very recent, extremely high frequency & spend | "Dedicated account manager + bulk discount" |

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, matplotlib, seaborn
- **ML** — K-Means Clustering, DBSCAN, StandardScaler
- **Frontend** — Streamlit
- **Model persistence** — joblib

---

## Project Structure

```
Customer DNA Engine/
├── clustering.ipynb       # Full ML pipeline — EDA, cleaning, feature engineering, clustering
├── app.py                 # Streamlit frontend
├── kmeans_model.pkl       # Trained K-Means model (K=3)
├── scaler.pkl             # Fitted StandardScaler
└── README.md
```

> ⚠️ The dataset (`online_retail_II.csv`) is not included due to file size.  
> Download **"Online Retail II UCI"** by Miyabon from [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and place it in the root folder.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/Customer-DNA-Engine.git
cd Customer-DNA-Engine
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

**3. Download the dataset**  
Get `online_retail_II.csv` from Kaggle (link above) and place it in the root folder.

**4. Run the notebook**  
Open `clustering.ipynb` in Jupyter or PyCharm and run all cells. This will regenerate `kmeans_model.pkl` and `scaler.pkl`.

**5. Launch the app**
```bash
streamlit run app.py
```

---

## ML Pipeline

```
Raw Data (1M+ rows)
        ↓
EDA & Cleaning
(drop nulls, remove cancellations, filter negatives)
        ↓
Feature Engineering — RFM
(one row per customer: Recency, Frequency, Monetary)
        ↓
StandardScaler
(normalize all features to the same scale)
        ↓
Elbow Method + Silhouette Score
(mathematically determine K=3)
        ↓
K-Means Clustering
(assign segment labels 0, 1, 2)
        ↓
Segment Naming + Promotion Engine
```

---

## Key Findings

- The dataset contains **~5,878 unique customers** after cleaning
- A tiny group of **VIP Wholesalers** (Cluster 2) drives the majority of revenue — avg spend £173,123 per customer
- The largest group is **Loyal Regulars** (Cluster 1) — your reliable everyday buyers
- DBSCAN was tested but produced 9 irregular clusters due to density variation — K-Means with K=3 gave cleaner, more interpretable results

---

## Dataset

**UCI Online Retail II** — Real UK-based online retail transactions (Dec 2009 – Dec 2011)  
Source: [Kaggle — Miyabon](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

---

*Built as part of Sheryians AI School — Data Science playlist (Parts 1 & 4)*
