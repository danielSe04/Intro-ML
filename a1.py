import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy
import sklearn

'''
This file contains functions for dimensionality reduction and eigen-value profiling of the dataset.
Authors: Daniel Seidel & Mihnea Pasere
'''

'''
This function takes in one argument which is a numpy array and returns a normalized version of the array.
The normalized version is calculated by subtracting the mean of the array from each element 
and dividing each element by the standard deviation of the array.
'''
def normalize_data(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    return (x-mean)/sd
'''
This function takes in as argument a normalized set of data, which is a numpy array.
It then computes the covariance matrix of the data and returns the eigenvalues and eigenvectors of that matrix.
'''
def calculate_eigen(z):
    # Take co-variance matrix C of z
    cov = np.cov(z, rowvar=False)
    # Calculate eignevalues and eigenvector of C
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvalues, eigenvectors

'''
The function below realizes the t-SNE visualization of the dataset.
It also reduces the data x to d dimensions using PCA.
'''
def dimension_reduced_data(x, y, d, perplexity, random_state):
    _, _, z = pca(x, d)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    z_2d = tsne.fit_transform(z)
    plt.figure(figsize=(6.4,4.8), dpi=100)
    for i, point in enumerate(z_2d):
        plt.plot(point[0], point[1], marker = "o", label=f"Class {(int) (i / 72)}")
    plt.legend(
        title="Object ID",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.savefig("Plot.png")

'''
This function outputs the distribution of the eigenvalues of the dataset.
'''
def eigen_value_profile(x, d):
    _, values, _ = pca(x, d)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(values, color='purple')

    ax.set_xlabel("Index eigen-value")
    ax.set_ylabel("Eigen-value")
    ax.set_title("Eigen-value Profile of the Dataset")
    fig.savefig(sys.stdout.buffer)

'''
Takes the first image from the dataset in its array form and plots it as an image.
'''
def input_data_sample():
    plt.title("Input data sample as an image")
    plt.imshow(np.reshape(x_coords[0,:],(32,32)))
    plt.savefig(sys.stdout.buffer)

'''
The function is used to implement the standard PCA algorithm, reducing the dimension of x to d.
This function outputs three things: Principal components, eigenvalues, reduced version of data set
It uses previously defined functions to normalize the data, compute eigenvalues and eigenvectors, and then sort them.
Finally, it reduces the dimensionality of the data by matrix multiplication with the first d eigenvectors.
'''
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

'''
Defines the input data.
'''
dict_in = scipy.io.loadmat("COIL20.mat")
x_coords = np.array(dict_in['X'])
y_coords = np.array(dict_in['Y'])