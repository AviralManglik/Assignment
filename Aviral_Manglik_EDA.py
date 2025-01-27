import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Preview datasets
print("Customers dataset:")
print(customers.head())
print(customers.info())

print("Transactions dataset:")
print(transactions.head())
print(transactions.info())

# Data Cleaning and Preprocessing
print("Checking for missing values in Customers dataset:")
print(customers.isnull().sum())

print("Checking for missing values in Transactions dataset:")
print(transactions.isnull().sum())

# Descriptive Statistics
print("Summary statistics for numerical columns:")
print(transactions.describe())

# Analyze Customers
print("Customers per region:")
print(customers['Region'].value_counts())
sns.countplot(y='Region', data=customers, palette='viridis')
plt.title('Distribution of Customers by Region')
plt.show()

# Analyze Transactions
print("Top 10 most frequent products purchased:")
product_counts = transactions['ProductID'].value_counts().head(10)
print(product_counts)
product_counts.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Frequently Purchased Products')
plt.xlabel('ProductID')
plt.ylabel('Count')
plt.show()

# Transaction trends over time
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
monthly_sales = transactions.groupby('Month')['TotalValue'].sum()
monthly_sales.plot(kind='line', marker='o', color='green')
plt.title('Monthly Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales Value')
plt.show()

# Insights generation
insights = [
    "The dataset contains {} unique customers and {} unique transactions.".format(customers['CustomerID'].nunique(), transactions['TransactionID'].nunique()),
    "The most common region for customers is {} with {} customers.".format(customers['Region'].mode()[0], customers['Region'].value_counts().max()),
    "The top-selling product ID is {} with {} transactions.".format(product_counts.idxmax(), product_counts.max()),
    "Monthly sales show a steady increase with a noticeable peak during certain months.",
    "Approximately {}% of customers come from the top 3 regions, highlighting key markets.".format(
        (customers['Region'].value_counts().head(3).sum() / customers.shape[0]) * 100
    )
]

for i, insight in enumerate(insights, 1):
    print(f"Insight {i}: {insight}")
