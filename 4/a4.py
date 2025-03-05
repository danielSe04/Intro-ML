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
            error += minimum
        errors.append(error)
        updated_prototypes.append(np.array(prototypes.copy()))
    return np.array(updated_prototypes), np.array(errors)

data = []
with open("simplevqdata.csv", "r") as f:
    csv_file = csv.reader(f)
    for line in csv_file:
        data.append([float(line[0]), float(line[1])])
data = np.array(data)
prototype_trace, HVQ_trace = vector_quantization(k=2, learning_rate=0.1, max_epoch=100)
print(prototype_trace)
print(np.round(HVQ_trace, decimals = 5))