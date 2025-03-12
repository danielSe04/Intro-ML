import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(1740844038)
np.set_printoptions(precision=5, suppress=True)


def vector_quantization(k: int, learning_rate: float, max_epoch: int, randomInit = True):
    if randomInit:
        prototypes = data[np.random.randint(len(data), size=k)]
    else:
        prototypes = data[:k]
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

def plot_vq(k: int, learning_rate: float, max_epoch: int, j: int):
    prototypes, errors = vector_quantization(k, learning_rate, max_epoch, True)
    colors = ['red','blue', 'yellow', 'green']
    plt.scatter(data[:,0], data[:,1], edgecolors='k')
    for i in range(k):
        prototype_x = [p[i][0] for p in prototypes[:-1]]
        prototype_y = [p[i][1] for p in prototypes[:-1]]
        plt.scatter(prototype_x, prototype_y, c=colors[i%len(colors)])
        plt.plot(prototype_x, prototype_y, c=colors[i%len(colors)])
    plt.title("Trajectory Of Prototypes")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(os.path.join("plot_" + str(j) + ".png"))
    plt.close()
    print(errors[-1])
    return prototypes, errors 

def plot_vq_error(HVQerror_k, max_epoch: int, i: int):
    plt.plot(HVQerror_k)
    plt.title("Quantization Error Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Quantization Error")
    plt.savefig(os.path.join("plot_error_" + str(i) + ".png"))
    plt.close()

data = []
with open("simplevqdata.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append([float(line[0]), float(line[1])])
data = np.array(data)

learning_rates = [0.5, 0.1, 0.01, 0.001, 0.0001]
# for i, learning_rate in enumerate(learning_rates):
#     if i == 0:
#         prototype_trace, HVQ_trace = plot_vq(k=4, learning_rate=learning_rate, max_epoch=100, j=i)
#         plot_vq_error(HVQ_trace, 100, i)
ks = range(1,6)
errors = []
for k in ks:
    _, HVQ_trace = vector_quantization(k, 0.001, 100)
    errors.append(HVQ_trace[-1])

plt.plot(ks, errors)
plt.title("Quantization Error for Varying Amount of Prototypes")
plt.xlabel("Number of Prototypes")
plt.ylabel("Quantization Error")
plt.savefig(os.path.join("plot.png"))
plt.close()