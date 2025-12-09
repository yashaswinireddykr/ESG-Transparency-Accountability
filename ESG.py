#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 15:05:43 2025

@author: yashaswinireddykr
"""

#import libraries
import pandas as pd
 
#load the data
df=pd.read_csv("SP_500_ESG_Risk_Ratings.csv")

#standardize columns names
df.columns=df.columns.str.strip().str.replace(' ', '_') 

#clean column (converting to numeric)
df['Full_Time_Employees'] = df['Full_Time_Employees'].str.replace(',', '', regex=False)
df['Full_Time_Employees'] = pd.to_numeric(df['Full_Time_Employees'], errors='coerce')

#extract numeric value 
df['ESG_Risk_Percentile'] = df['ESG_Risk_Percentile'].str.extract(r'(\d+)')
df['ESG_Risk_Percentile'] = pd.to_numeric(df['ESG_Risk_Percentile'], errors='coerce')

#mapping 'controversy_level' to numeric value
controversy_map = {
    'Low Controversy Level': 1,
    'Moderate Controversy Level': 2,
    'Significant Controversy Level': 3,
    'High Controversy Level': 4,
    'Severe Controversy Level': 5
}
df['Controversy_Level_Score'] = df['Controversy_Level'].map(controversy_map)
df['Controversy_Score'] = pd.to_numeric(df['Controversy_Score'], errors='coerce') 

#drop rows with missing core ESG scores
required_columns = ['Total_ESG_Risk_score', 'Environment_Risk_Score', 'Social_Risk_Score', 'Governance_Risk_Score']
df.dropna(subset=required_columns, inplace=True)

#dropping less relevant columns
df.drop(columns=['Address', 'Description'], errors='ignore', inplace=True)

#reset index
df.reset_index(drop=True, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#correlation matrix for ESG core components
plt.figure()
corr_matrix = df[['Total_ESG_Risk_score', 'Environment_Risk_Score',
                  'Social_Risk_Score', 'Governance_Risk_Score', 'Controversy_Score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of ESG Risk Components")
plt.tight_layout()
plt.show()

#select ESG features
features = ['Environment_Risk_Score', 'Social_Risk_Score', 'Governance_Risk_Score', 'Controversy_Score']
df_cluster = df.dropna(subset=features).copy()

#standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[features])

#apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster['ESG_Cluster'] = kmeans.fit_predict(X_scaled)

#apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_cluster['PCA1'] = pca_result[:, 0]
df_cluster['PCA2'] = pca_result[:, 1]

#plot PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cluster, x='PCA1', y='PCA2', hue='ESG_Cluster', palette='Set2')
plt.title("PCA: ESG Clusters Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

#group by sector and calculate average ESG scores
sector_avg = df.groupby('Sector')[['Total_ESG_Risk_score', 'Environment_Risk_Score',
                                   'Social_Risk_Score', 'Governance_Risk_Score']].mean().sort_values('Total_ESG_Risk_score', ascending=False)

#plot bar chart
sector_avg.plot(kind='bar', figsize=(14, 6), colormap='viridis')
plt.title("Average ESG Risk Scores by Sector")
plt.ylabel("Average Score")
plt.xlabel("Sector")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#save cleaned data
df_cluster.to_csv("Cleaned_ESG_Ratings.csv", index=False)

#preview cleaned data
print(df_cluster.head())
