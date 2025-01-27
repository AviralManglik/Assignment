import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure all required modules are installed
try:
    import micropip
    micropip.install(["pandas", "numpy", "scikit-learn", "seaborn", "matplotlib"])
except ModuleNotFoundError:
    print("micropip is not available in this environment. Please install the required packages manually.")

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge datasets
customer_transactions = transactions.merge(customers, on='CustomerID', how='inner')

# Feature engineering: Aggregate transaction data
customer_summary = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',   # Total transaction value per customer
    'Quantity': 'sum',     # Total quantity purchased
    'ProductID': 'nunique', # Number of unique products purchased
    'Region': 'first'      # Customer region
}).reset_index()

# One-hot encode 'Region'
customer_summary = pd.get_dummies(customer_summary, columns=['Region'], drop_first=True)

# Normalize features
scaler = StandardScaler()
features = customer_summary.drop(columns=['CustomerID'])
scaled_features = scaler.fit_transform(features)

# K-Means Clustering
optimal_clusters = range(2, 11)
db_scores = []

for n_clusters in optimal_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    db_index = davies_bouldin_score(scaled_features, cluster_labels)
    db_scores.append(db_index)

# Plot DB Index
plt.plot(optimal_clusters, db_scores, marker='o', linestyle='-', color='blue')
plt.title('Davies-Bouldin Index for Different Cluster Sizes')
plt.xlabel('Number of Clusters')
plt.ylabel('DB Index')
plt.show()

# Optimal number of clusters
optimal_n = optimal_clusters[np.argmin(db_scores)]
kmeans = KMeans(n_clusters=optimal_n, random_state=42)
customer_summary['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters
sns.pairplot(customer_summary, hue='Cluster', palette='viridis')
plt.title('Cluster Visualization')
plt.show()

# Print cluster information
print(f"Optimal Number of Clusters: {optimal_n}")
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Save cluster assignments
customer_summary[['CustomerID', 'Cluster']].to_csv("Customer_Segments.csv", index=False)
print("Customer segmentation completed. Results saved to Customer_Segments.csv.")
