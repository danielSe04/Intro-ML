import csv
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(1740844038)
np.set_printoptions(precision=5, suppress=True)


def vector_quantization(k: int, learning_rate: float, max_epoch: int):
    prototypes = data[np.random.randint(len(data), size=k)]
    errors = []
    updated_prototypes = [np.array(prototypes.copy())]
    r = int(random.random() * len(data))
    for epoch in range(max_epoch):
        error = 0.0
        point_set = np.random.permutation(data)
        for point in point_set:
            minimum = float('inf')
            index = -1
            for k in range(len(prototypes)):
                dist = math.sqrt((point[0] - prototypes[k][0]) ** 2 + (point[1] - prototypes[k][1]) ** 2)
                if dist < minimum:
                    minimum = dist
                    index = k
            prototypes[index] += learning_rate * (point - prototypes[index])
        updated_prototypes.append(np.array(prototypes.copy()))
        # Calculate error
        for point in point_set:
            minimum = float('inf')
            for k in range(len(prototypes)):
                dist = math.sqrt((point[0] - prototypes[k][0]) ** 2 + (point[1] - prototypes[k][1]) ** 2)
                if dist < minimum:
                    minimum = dist
            error += minimum**2
        errors.append(error)
    return [np.array(prot) for prot in updated_prototypes], np.array(errors)

def plot_vq(k: int, learning_rate: float, max_epoch: int):
    prototypes, errors = vector_quantization(k, learning_rate, max_epoch)
    colors = ['red','blue', 'yellow', 'green']
    plt.scatter(data[:,0], data[:,1], edgecolors='k')
    for i in range(k):
        prototype_x = [p[i][0] for p in prototypes[:-1]]
        prototype_y = [p[i][1] for p in prototypes[:-1]]
        print(len(prototype_x), len(prototype_y))
        plt.scatter(prototype_x, prototype_y, c=colors[i%len(colors)])
        plt.plot(prototype_x, prototype_y, c=colors[i%len(colors)])
    plt.title("Trajectory Of Prototypes")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(sys.stdout.buffer)
    plt.close()

def plot_vq_error(HVQerror_k, max_epoch: int):
    plt.plot(HVQerror_k)
    plt.title("Quantization Error Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Quantization Error")
    plt.savefig(sys.stdout.buffer)
    plt.close()

data = []
with open("simplevqdata.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append([float(line[0]), float(line[1])])
data = np.array(data)
# prototype_trace, HVQ_trace = vector_quantization(k=2, learning_rate=0.1, max_epoch=100)
# print(prototype_trace)
# print(np.round(HVQ_trace, decimals = 5))
# plot_vq_error(HVQ_trace, max_epoch=100)
#plot_vq(k=2, learning_rate=0.1, max_epoch=100)