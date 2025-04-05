# %% [markdown]
# ### Importing Packages

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# %% [markdown]
# ### Loading the dataset

# %%
df = pd.read_csv("credit_card_customer_data.csv")
df.head()

# %% [markdown]
# ### Preliminary Analysis

# %%
df.shape

# %%
df.info()

# %% [markdown]
# inference: <br>
# * There are 660 rows and 10 columns. 
# * There are no null values in the dataset
# * Since every value is numerical, there is no need to convert any categorical variables.

# %% [markdown]
# ### Dropping unnecessary features

# %%
features = df.drop(columns = ["Sl_No", "Customer Key"])

# %% [markdown]
# ### Standardizing the features

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# %% [markdown]
# ### Finding optimal number of clusters

# %%
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# %% [markdown]
# ### Plot Elbow Method and Silhouette Score

# %%
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')

plt.tight_layout()
plt.show() 

# %% [markdown]
# Inference: <br>
# * Since the silhouetter score is highest at k=3, we can choose 3 clusters.
# * The elbow method also has a curve at k=3, which supports the choice of 3 clusters.

# %%
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# %%
final_silhouette = silhouette_score(X_scaled, df['Cluster'])
print(f"Final Silhouette Score: {final_silhouette:.2f}")

# %%
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title('K-Means Clustering Visualization (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# %% [markdown]
# Inference <br>
# * The groups were grouoed into three fairly good groups. 
# * The silhouette score is 0.52 which is a good score.
# * The elbow method suggests that three clusters are optimal.


