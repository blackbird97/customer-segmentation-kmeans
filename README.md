# Customer Segmentation Using K-Means Clustering

## 📌 Project Overview

This project performs **Customer Segmentation Analysis** using the **K-Means Clustering algorithm** on the Mall Customers dataset.  

The objective is to group customers based on demographic and spending behavior to support **data-driven marketing strategy** and improve business decision-making.

This project demonstrates practical skills in:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Selection
- Data Preprocessing
- Unsupervised Machine Learning
- Model Evaluation (Elbow Method)
- Data Visualization
- Business Insight Interpretation

---

## 🎯 Business Problem

Many businesses treat customers as a single segment, which can reduce marketing effectiveness.

This project answers:

- How can customers be segmented based on income and spending behavior?
- Which groups represent high-value customers?
- How can segmentation improve marketing strategy?

---

## 📂 Dataset Information

Dataset: **Mall Customers Dataset**

Features used:

- `Age`
- `Annual Income (k$)`
- `Spending Score (1–100)`

Column renaming for cleaner analysis:

```python
df.rename(columns={
    'Annual Income (k$)' : 'Annual_Income',
    'Spending Score (1-100)' : 'Spending_Score'
}, inplace=True)
🧪 Methodology
1️⃣ Data Understanding
df.info()
df.describe()
Checked data types

Reviewed statistical distribution

Verified dataset quality

2️⃣ Feature Selection
X = df[['Age', 'Annual_Income', 'Spending_Score']]
Selected relevant numerical variables for clustering.

3️⃣ Data Preprocessing (Feature Scaling)
K-Means is distance-based, so scaling is required:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
4️⃣ Determining Optimal Clusters (Elbow Method)
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)
The optimal number of clusters was determined at K = 3.

5️⃣ Model Training
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
6️⃣ Data Visualization
sns.scatterplot(
    x=df['Annual_Income'],
    y=df['Spending_Score'],
    hue=df['Cluster'],
    palette='Set2'
)
Visualization shows clear customer segmentation patterns.

📊 Cluster Interpretation
Cluster analysis based on average values:

cluster_summary = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean()
cluster_summary
Example Business Interpretation
Cluster 0 – High Income, High Spending

Premium customers

Strategy: Loyalty programs & exclusive offers

Cluster 1 – High Income, Low Spending

High potential customers

Strategy: Personalized promotions

Cluster 2 – Lower Income, High Spending

Price-sensitive active buyers

Strategy: Discount-based campaigns

🛠 Tools & Technologies
Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook

💡 Key Skills Demonstrated
Data Cleaning & Preparation

Exploratory Data Analysis (EDA)

Feature Scaling

K-Means Clustering

Elbow Method

Data Visualization

Business Insight Translation

🚀 Business Impact
Customer segmentation enables:

Targeted marketing campaigns

Budget optimization

Customer retention improvement

Data-driven strategic planning

✅ Conclusion
This project demonstrates how unsupervised machine learning can generate actionable business insights.

K-Means clustering successfully segments customers into meaningful groups that support strategic marketing decisions.
