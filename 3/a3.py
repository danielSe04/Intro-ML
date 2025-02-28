import csv
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def region_query(D, P, eps: float) -> list:
    neighbor_points = []
    for i,point in enumerate(D):
        if math.sqrt((point[0]-P[0])**2 + (point[1]-P[1])**2) <= eps:
            neighbor_points.append(i)
    return neighbor_points

def expand_cluster(D, P, neighbour_pts, cluster_index, eps, min_pts):
    D[P] = (D[P][0], D[P][1], cluster_index)
    neighbour_queue = deque(neighbour_pts)
    neighbours_checked = set(neighbour_pts)
    while neighbour_queue:
        point = neighbour_queue.popleft()
        if D[point][2] == -1:
            D[point] = (D[point][0], D[point][1], cluster_index)
            continue
        if D[point][2] == -2:
            D[point] = (D[point][0], D[point][1], cluster_index)
            neighbour_pts_expanded = region_query(D, D[point], eps)
            if len(neighbour_pts_expanded) >= min_pts:
                neighbour_pts_expanded = [pt for pt in neighbour_pts_expanded if not pt in neighbours_checked]
                neighbours_checked.update(neighbour_pts_expanded)
                neighbour_queue.extend(neighbour_pts_expanded)
    return D
    
def DBSCAN(D, eps: float, MinPts: int):
    cluster_index = 0
    for i, data_point in enumerate(D):
        if data_point[2] == -2:
            D[i] = (data_point[0], data_point[1], -1)
            neighbour_pts = region_query(D, data_point, eps)
            if len(neighbour_pts) < MinPts:
                continue
            cluster_index += 1
            D = expand_cluster(D, i, neighbour_pts, cluster_index, eps, MinPts)
    return [point[2] for point in D]

def plot_db_scan(D, eps, k):
    types = DBSCAN(D, eps, k)
    x = [point[0] for point in D]
    y = [point[1] for point in D]


    scatter = plt.scatter(x,y, c=types, cmap="viridis", edgecolors="k")
    handles, labels = scatter.legend_elements()
    legend = plt.legend(handles=handles, labels=labels, title="Clusters")
    plt.gca().add_artist(legend)
    plt.title(f"DBSCAN clustering with MinPt={k},eps={eps}")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.savefig(sys.stdout.buffer)
    plt.close()


data = [] 
with open("data_clustering.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append((float(line[0]), float(line[1]), -2))
#data = np.array(data)
#plot_db_scan(data, 0.04, 3)
