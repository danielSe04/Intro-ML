import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy


def normalize_data(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    mean = np.mean(x)
    sd = np.std(x)
    return (x-mean)/sd

def calculate_eigen(z):
    # Take co-variance matrix C of z
    cov = np.cov(z)
    # Calculate eignevalues and eigenvector of C
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvalues, eigenvectors

#def dimension_reduced_data(x,y,d,perplexity,random_state):

def eigen_value_profile(x, d):
    
    _, values, _ = pca(x, d)

    x = range(1, len(values) + 1)
    plt.plot(x,values)

    plt.xlabel("Index eigen-value")
    plt.ylabel("Eigen-value")
    plt.legend(title="Eigen-value Profile of the Dataset",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(sys.stdout.buffer)



#def input_data_sample():

# Output: Principal components, eigen-values, reduced version of data set
def pca(x, d):
    z = normalize_data(x)
    eigenvalues, eigenvectors = calculate_eigen(z)
    # Sort and take d highest eigenvalues
    eig_map = {eigenvalues[i]: eigenvectors[i] for i in range(len(eigenvalues))}
    eig_map_sorted = dict(sorted(eig_map.items(), reverse=True))
    values = np.array(list(eig_map_sorted.keys()))[0:d]
    vectors = np.array(list(eig_map_sorted.values()))[0:d]
    # Reduce the dimensionality of Z
    reduced_z = vectors @ z
    return vectors, values, reduced_z

dict_in = scipy.io.loadmat("COIL20.mat")
x_coords = np.array(dict_in['X'])