import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge datasets for combined information
customer_transactions = transactions.merge(customers, on='CustomerID', how='inner')

# Feature engineering: Aggregate transaction data
customer_summary = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',   # Total transaction value per customer
    'Quantity': 'sum',     # Total quantity purchased
    'ProductID': 'nunique', # Number of unique products purchased
    'Region': 'first'      # Customer region
}).reset_index()

# One-hot encoding for the 'Region' feature
customer_summary = pd.get_dummies(customer_summary, columns=['Region'], drop_first=True)

# Normalize features
scaler = StandardScaler()
features = customer_summary.drop(columns=['CustomerID'])
scaled_features = scaler.fit_transform(features)

# Compute similarity matrix
similarity_matrix = cosine_similarity(scaled_features)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_summary['CustomerID'], columns=customer_summary['CustomerID'])

# Function to find top N similar customers
def get_top_similar(customers_df, customer_id, top_n=3):
    similar_customers = customers_df[customer_id].sort_values(ascending=False)[1:top_n+1]
    return [(idx, round(score, 2)) for idx, score in similar_customers.items()]

# Generate lookalike results for the first 20 customers
lookalike_results = {}
for customer_id in customer_summary['CustomerID'][:20]:
    lookalike_results[customer_id] = get_top_similar(similarity_df, customer_id)

# Convert lookalike results to the required CSV format
lookalike_df = pd.DataFrame({
    "CustomerID": lookalike_results.keys(),
    "Lookalikes": [
        [{"CustomerID": pair[0], "Score": pair[1]} for pair in lookalike_results[cid]]
        for cid in lookalike_results
    ]
})

# Save results to CSV
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike model completed. Results saved to Lookalike.csv.")
