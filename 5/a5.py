import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math
np.random.seed(1740844038)
np.set_printoptions(precision=5, suppress=True)

CLASS_SIZE = 50

def plot_data():
    plt.scatter(data[0][:, 0], data[0][:, 1], color='red')
    plt.scatter(data[1][:, 0], data[1][:, 1], color='blue')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig("plot.png")
    plt.close()

def find_closest_prototype(prototypes, point):
    minimum = float('inf')
    index = -1
    for k in range(len(prototypes)):
        dist = (point[0] - prototypes[k][0][0]) ** 2 + (point[1] - prototypes[k][0][1]) ** 2
        if dist < minimum:
            minimum = dist
            index = k
    return index, minimum

def linear_vector_quantization(num_prototypes, learning_rate, max_epoch):
    prototypes = []
    misclassifications = []
    errors = []
    # Generate prototypes
    for i, data_class in enumerate(data):
        random_indices = np.random.choice(len(data_class), size=num_prototypes)
        for index in random_indices:
            prototypes.append([data_class[index], i])
    prototype_trace = [[p[0] for p in prototypes]]
    previous_labels = []
    for epoch in range(max_epoch):
        misclassifications_epoch = 0
        labels = []
        point_set = np.random.permutation(data.reshape(2*CLASS_SIZE, -1))
        for i,point in enumerate(point_set):
            index, minimum = find_closest_prototype(prototypes, point)
            sign = 1 if prototypes[index][1] == (int)(i / CLASS_SIZE) else -1

            if sign == -1: misclassifications_epoch += 1
            if not isinstance(prototypes[index][0], np.ndarray):
                print("error")
            prototypes[index][0] += learning_rate * sign * (point-prototypes[index][0])
            labels.append(prototypes[index][1])
        prototype_trace.append([[p[0][0], p[0][1]] for p in prototypes])
        misclassifications.append(misclassifications_epoch)
        # if error does not change anymore, return

    predicted_labels = []
    for point in data.reshape(2*CLASS_SIZE, -1):
        index, _ = find_closest_prototype(prototypes, point)
        predicted_labels.append(prototypes[index][1] + 1)

    return prototype_trace, predicted_labels, misclassifications

data = []
with open("data_lvq.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append([float(line[0]), float(line[1])])
data = np.array([data[:CLASS_SIZE], data[CLASS_SIZE:2*CLASS_SIZE]])
prototypes, labels, misclassifications = linear_vector_quantization(1, 0.002, 100)

print(prototypes)
print(misclassifications)