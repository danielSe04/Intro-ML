import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hier
from sklearn.cluster import AgglomerativeClustering

def plot_data_using_scatter_plot():
    data_x = data[:, 0]
    data_y = data[:, 1]
    plt.figure(figsize=(6.4,4.8), dpi=100)
    plt.scatter(data_x, data_y)
    plt.title("Scatter plot - original data")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.tight_layout()
    plt.savefig(sys.stdout.buffer)

def plot_dendrogram(linkage_measure: str, calc_thresholds: bool):
    linkage_matrix = hier.linkage(data, method=linkage_measure, metric="euclidean")
    fig, ax = plt.subplots(figsize=(6.4,4.8), dpi=100)
    dendro = hier.dendrogram(linkage_matrix, ax=ax)
    ax.set_title(f"Dendrogram - {linkage_measure} measure")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Dissimilarity")
    ax.set_xticks([])
    if calc_thresholds:
        if linkage_measure == "single":
            lines = [0.13, 0.14, 0.15]
        elif linkage_measure == "average":
            lines = [0.26, 0.3, 0.4]
        elif linkage_measure == "complete":
            lines = [0.55, 0.63, 0.75]
        elif linkage_measure == "ward":
            lines = [1.1, 2, 3]
        else:
            lines = []
        for line in lines:
            ax.axhline(y= line, color = "black", linestyle="--")
    fig.savefig(sys.stdout.buffer)

def agglomerative_clustering(measure: str, k: int):
    a = AgglomerativeClustering(n_clusters=k, linkage=measure).fit(data)
    fitted_data = []
    labels = a.labels_
    #for i in range(k):
     #   points = [data[j] for j in range(len(data)) if i == a.labels_[j]]
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors="k")
    plt.title(f"Clustering results for {k} clusters, using '{measure}' measure")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.savefig(sys.stdout.buffer)
    plt.close()

data = [] 
with open("data_clustering.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append([float(line[0]), float(line[1])])
data = np.array(data)