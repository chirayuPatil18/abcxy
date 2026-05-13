import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
X = load_iris().data
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
labels = model.predict(X)
silhouette_score(X, labels)
model
model = dendrogram(linkage(X, method='ward'))
plt.show(model)