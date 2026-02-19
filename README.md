# 🛍 Customer Segmentation Using K-Means Clustering

---

## 📌 Project Overview

This project performs **Customer Segmentation Analysis** using the **K-Means Clustering algorithm** on the **Mall Customers dataset**.

The objective is to group customers based on demographic characteristics and spending behavior to support **data-driven marketing strategies** and improve business decision-making.

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

Many businesses treat customers as a single homogeneous segment, which can reduce marketing effectiveness and ROI.

This project answers:

- How can customers be segmented based on income and spending behavior?  
- Which groups represent high-value customers?  
- How can segmentation improve marketing strategy?  

---

## 📂 Dataset Information

**Dataset:** Mall Customers Dataset  

### Features Used

- `Age`
- `Annual Income (k$)`
- `Spending Score (1–100)`

### Column Renaming

```python
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income',
    'Spending Score (1-100)': 'Spending_Score'
}, inplace=True)
```

---

## 🧪 Methodology

### 1️⃣ Data Understanding

```python
df.info()
df.describe()
```

Performed:

- Data type verification  
- Statistical distribution review  
- Missing value inspection  
- Dataset quality validation  

---

### 2️⃣ Feature Selection

```python
X = df[['Age', 'Annual_Income', 'Spending_Score']]
```

Selected relevant numerical variables for clustering.

---

### 3️⃣ Data Preprocessing (Feature Scaling)

Because K-Means is distance-based, feature scaling is required.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 4️⃣ Determining Optimal Clusters (Elbow Method)

```python
from sklearn.cluster import KMeans

inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)
```

The optimal number of clusters was determined at:

**K = 3**

---

### 5️⃣ Model Training

```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

Cluster labels were added to the dataset.

---

### 6️⃣ Data Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    x=df['Annual_Income'],
    y=df['Spending_Score'],
    hue=df['Cluster'],
    palette='Set2'
)

plt.title("Customer Segmentation")
plt.show()
```

The visualization reveals distinct customer segments.

---

## 📊 Cluster Interpretation

```python
cluster_summary = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean()
cluster_summary
```

### 🔍 Business Interpretation

#### 🟢 Cluster 0 – High Income, High Spending
- Premium customers  
- High lifetime value  

**Strategy:**  
- Loyalty programs  
- VIP offers  
- Exclusive campaigns  

---

#### 🔵 Cluster 1 – High Income, Low Spending
- High purchasing power  
- Untapped potential  

**Strategy:**  
- Personalized promotions  
- Cross-selling  
- Targeted remarketing  

---

#### 🟣 Cluster 2 – Lower Income, High Spending
- Active buyers  
- More price-sensitive  

**Strategy:**  
- Discount campaigns  
- Bundling strategy  
- Flash sales  

---

## 🛠 Tools & Technologies

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 💡 Key Skills Demonstrated

- Data Cleaning & Preparation  
- Exploratory Data Analysis  
- Feature Scaling  
- K-Means Clustering  
- Elbow Method  
- Data Visualization  
- Business Insight Translation  

---

## 🚀 Business Impact

Customer segmentation enables:

- Targeted marketing campaigns  
- Budget optimization  
- Improved customer retention  
- Data-driven strategic planning  

---

## ✅ Conclusion

This project demonstrates how unsupervised machine learning can generate actionable business insights.

K-Means clustering successfully segments customers into meaningful groups that support strategic marketing decisions.

---

## 📌 Future Improvements

- Use Silhouette Score for validation  
- Compare with Hierarchical Clustering  
- Apply PCA for better visualization  
- Deploy model as API for real-world CRM integration  

---

**Author:** Mohammad Azizul Bazarun
**Role:** Data Analyst | Machine Learning Enthusiast  
**Project Type:** Unsupervised Machine Learning  
