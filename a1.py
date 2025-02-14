import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy


def normalize_data(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    return (x-mean)/sd

def calculate_eigen(z):
    # Take co-variance matrix C of z
    cov = np.cov(z, rowvar=False)
    # Calculate eignevalues and eigenvector of C
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvalues, eigenvectors

#def dimension_reduced_data(x,y,d,perplexity,random_state):

def eigen_value_profile(x, d):
    
    z = normalize_data(x)
    eigenvalues, eigenvectors = calculate_eigen(z)
    # Sort and take d highest eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    values = eigenvalues[sorted_indices]

    x = range(1, len(values) + 1)
    plt.plot(x,values, color='purple')

    plt.xlabel("Index eigen-value")
    plt.ylabel("Eigen-value")
    plt.title("Eigen-value Profile of the Dataset")
    plt.tight_layout()
    plt.savefig(sys.stdout.buffer)

#def input_data_sample():

# Output: Principal components, eigen-values, reduced version of data set
def pca(x, d):
    z = normalize_data(x)
    eigenvalues, eigenvectors = calculate_eigen(z)

    eig_pairs = sorted(zip(eigenvalues, eigenvectors.T), key=lambda x: x[0], reverse=True)
    # Sort and take d highest eigenvalues
    values = np.array([p[0] for p in eig_pairs])
    vectors = np.array([p[1] for p in eig_pairs])[:d,:].T
    # Reduce the dimensionality of Z
    reduced_z = z @ vectors
    return vectors, values, reduced_z

dict_in = scipy.io.loadmat("COIL20.mat")
x_coords = np.array(dict_in['X'])