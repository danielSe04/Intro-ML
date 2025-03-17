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
    plt.savefig(sys.stdout.buffer)
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
    data_1D = data.reshape(2*CLASS_SIZE, -1)
    # Generate prototypes
    for i, data_class in enumerate(data):
        random_indices = np.random.choice(len(data_class), size=num_prototypes, replace=False)
        for index in random_indices:
            prototypes.append([data_class[index], i+1])
    prototype_trace = [[p[0] for p in prototypes]]
    for epoch in range(max_epoch):
        permutation = np.random.permutation(len(data_1D))
        data_permutation = data_1D[permutation]
        label_permutation = data_labels[permutation]
        for i,point in enumerate(data_permutation):
            index, minimum = find_closest_prototype(prototypes, point)
            sign = 1 if prototypes[index][1] == label_permutation[i] else -1
            prototypes[index][0] = prototypes[index][0] + learning_rate*sign*(point-prototypes[index][0])
        prototype_trace.append([[p[0][0], p[0][1]] for p in prototypes])
        misclassifications_epoch = 0
        for i,point in enumerate(data_permutation):
            index, _ = find_closest_prototype(prototypes, point)
            misclassifications_epoch += 0 if prototypes[index][1] == label_permutation[i] else 1
        misclassifications.append(misclassifications_epoch)
        # if error does not change anymore, return

    predicted_labels = []
    for point in data.reshape(2*CLASS_SIZE, -1):
        index, _ = find_closest_prototype(prototypes, point)
        predicted_labels.append(prototypes[index][1])

    return np.array(prototype_trace), np.array(predicted_labels), np.array(misclassifications)

def plot_error_rate(num_prototypes, learning_rate, max_epoch):
    _, _, errors = linear_vector_quantization(num_prototypes, learning_rate, max_epoch)
    print(errors)
    error_percentage = errors/100
    print(error_percentage)
    plt.plot(range(len(errors)), error_percentage)
    plt.xlabel('Epoch')
    plt.ylabel('The error rate in %')
    plt.title('Learning curve')
    plt.savefig(sys.stdout.buffer)
    plt.close()

def plot_trajectory(num_prototypes, learning_rate, max_epoch):
    prototype_trace, labels, _ = linear_vector_quantization(num_prototypes, learning_rate, max_epoch)
    data_1D = data_1D = data.reshape(2*CLASS_SIZE, -1)
    data_1 = np.array([data_point for i, data_point in enumerate(data_1D) if labels[i] == 1])
    data_2 = np.array([data_point for i, data_point in enumerate(data_1D) if labels[i] == 2])
    plt.figure()
    plt.scatter(data_1[:, 0], data_1[:, 1], color='red')
    plt.scatter(data_2[:, 0], data_2[:, 1], color='blue')
    print(prototype_trace[-1])
    plt.scatter(prototype_trace[-1, :num_prototypes, 0], prototype_trace[-1, :num_prototypes, 1], color='red',marker='*',s=200)
    plt.scatter(prototype_trace[-1, num_prototypes:, 0], prototype_trace[-1, num_prototypes:, 1], color='blue',marker='*',s=200)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig(sys.stdout.buffer)
    plt.close()

data = []
with open("data_lvq.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append([float(line[0]), float(line[1])])
data = np.array([data[:CLASS_SIZE], data[CLASS_SIZE:2*CLASS_SIZE]])
data_labels = np.array([(int) (i / CLASS_SIZE) + 1 for i in range(2*CLASS_SIZE)])

#plot_data()

# prototypes, labels, misclassifications = linear_vector_quantization(1, 0.002, 100)
# print(prototypes)
# print(labels)
# print(misclassifications)

#plot_error_rate(1, 0.002, 100)

#plot_trajectory(1, 0.002, 100)