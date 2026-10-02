# Week 3 - Unsupervised Learning and Clustering Analysis
# Dataset: Titanic cleaned dataset from Week 1
# Reproduces the report analysis and charts.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

DATA = "titanic_cleaned.csv"
OUT = "charts"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# Do not use PassengerId or Survived for clustering.
# Survived is retained only for post-clustering descriptive profiling.
features = ["Age", "Fare", "Pclass", "SibSp", "Parch",
            "FamilySize", "IsAlone", "CabinKnown"]

X = df[features].copy()

# Standardization is important because variables have different scales.
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Candidate k values: 2 through 8
k_values = range(2, 9)
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(Xs)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(Xs, labels))

# Selected k = 5 for an interpretable solution in the elbow region.
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(Xs)

print("K-Means silhouette score:", silhouette_score(Xs, df["Cluster"]))
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster profile:")
print(df.groupby("Cluster")[features + ["Survived"]].mean().round(2))

# Hierarchical clustering using Ward linkage
agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
agg_labels = agg.fit_predict(Xs)
print("\nHierarchical silhouette score:", silhouette_score(Xs, agg_labels))

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xs)

plt.figure(figsize=(8.5, 6))
for c in sorted(df["Cluster"].unique()):
    idx = df["Cluster"].values == c
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=18, alpha=0.6, label=f"Cluster {c}")
plt.title("K-Means Clusters in 2D PCA Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_pca_clusters.png"), bbox_inches="tight")
plt.close()

# Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.axvline(k, linestyle="--", label="Selected k = 5")
plt.title("K-Means Elbow Analysis")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_elbow_curve.png"), bbox_inches="tight")
plt.close()

# Silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouettes, marker="o")
plt.axvline(k, linestyle="--", label="Selected k = 5")
plt.title("Silhouette Score Across Candidate Cluster Counts")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_silhouette_scores.png"), bbox_inches="tight")
plt.close()

# Hierarchical dendrogram
Z = linkage(Xs, method="ward")
plt.figure(figsize=(11, 5.5))
dendrogram(Z, truncate_mode="lastp", p=25)
plt.title("Hierarchical Clustering Dendrogram (Truncated View)")
plt.xlabel("Merged groups / sample index")
plt.ylabel("Ward distance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_hierarchical_dendrogram.png"), bbox_inches="tight")
plt.close()
