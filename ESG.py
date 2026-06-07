"""
ESG Transparency & Accountability
S&P 500 ESG Risk Analysis — Data Cleaning, Clustering, and Visualization
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Data Loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "SP_500_ESG_Risk_Ratings.csv"))

# ── Data Cleaning ─────────────────────────────────────────────────────────────

# Standardize column names
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Convert Full_Time_Employees to numeric
df['Full_Time_Employees'] = df['Full_Time_Employees'].str.replace(',', '', regex=False)
df['Full_Time_Employees'] = pd.to_numeric(df['Full_Time_Employees'], errors='coerce')

# Extract numeric value from ESG_Risk_Percentile
df['ESG_Risk_Percentile'] = df['ESG_Risk_Percentile'].str.extract(r'(\d+)')
df['ESG_Risk_Percentile'] = pd.to_numeric(df['ESG_Risk_Percentile'], errors='coerce')

# Map controversy level to numeric score
controversy_map = {
    'Low Controversy Level': 1,
    'Moderate Controversy Level': 2,
    'Significant Controversy Level': 3,
    'High Controversy Level': 4,
    'Severe Controversy Level': 5
}
df['Controversy_Level_Score'] = df['Controversy_Level'].map(controversy_map)
df['Controversy_Score'] = pd.to_numeric(df['Controversy_Score'], errors='coerce')

# Drop rows with missing core ESG scores
required_columns = ['Total_ESG_Risk_score', 'Environment_Risk_Score', 'Social_Risk_Score', 'Governance_Risk_Score']
df.dropna(subset=required_columns, inplace=True)

# Drop less relevant columns
df.drop(columns=['Address', 'Description'], errors='ignore', inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# ── Exploratory Analysis ──────────────────────────────────────────────────────

# Correlation matrix
plt.figure()
corr_matrix = df[['Total_ESG_Risk_score', 'Environment_Risk_Score',
                  'Social_Risk_Score', 'Governance_Risk_Score', 'Controversy_Score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of ESG Risk Components")
plt.tight_layout()
plt.show()

# Average ESG scores by sector
sector_avg = df.groupby('Sector')[['Total_ESG_Risk_score', 'Environment_Risk_Score',
                                   'Social_Risk_Score', 'Governance_Risk_Score']].mean().sort_values('Total_ESG_Risk_score', ascending=False)
sector_avg.plot(kind='bar', figsize=(14, 6), colormap='viridis')
plt.title("Average ESG Risk Scores by Sector")
plt.ylabel("Average Score")
plt.xlabel("Sector")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ── Clustering & PCA ──────────────────────────────────────────────────────────

# Select and scale features
features = ['Environment_Risk_Score', 'Social_Risk_Score', 'Governance_Risk_Score', 'Controversy_Score']
df_cluster = df.dropna(subset=features).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[features])

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster['ESG_Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_cluster['PCA1'] = pca_result[:, 0]
df_cluster['PCA2'] = pca_result[:, 1]

# PCA cluster plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cluster, x='PCA1', y='PCA2', hue='ESG_Cluster',
cat > ~/ESG-Transparency-Accountability/requirements.txt << 'EOF'
pandas
seaborn
matplotlib
scikit-learn
