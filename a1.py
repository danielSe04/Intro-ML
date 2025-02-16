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
    _, values, _ = pca(x, d)

    x = range(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(6.4,4.8), dpi=100)
    ax.plot(x,values, color='purple')

    ax.set_xlabel("Index eigen-value")
    ax.set_ylabel("Eigen-value")
    ax.set_title("Eigen-value Profile of the Dataset")
    fig.savefig(sys.stdout.buffer)

def input_data_sample():
    plt.figure(figsize=(6.4,4.8), dpi=100)
    plt.title("Input data sample as an image")
    plt.imshow(np.reshape(x_coords[0,:],(32,32)))
    plt.tight_layout()
    plt.savefig(sys.stdout.buffer)

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