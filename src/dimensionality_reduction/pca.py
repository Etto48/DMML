import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

def cov(data: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    if scipy.sparse.issparse(data):
        A = scipy.sparse.lil_matrix(data)
        mean = data.mean(axis=-1)
        for i in range(A.shape[-1]):
            A[:, i] -= mean
        return A @ A.T / (A.shape[1] - 1)
    else:
        A = data
        mean = np.mean(data, axis=-1)
        for i in range(A.shape[-1]):
            A[:, i] -= mean
        return A @ A.T / (A.shape[1] - 1)

def pca(data: np.ndarray | scipy.sparse.spmatrix, n_components: int, sparse=False) -> np.ndarray:
    if sparse:
        A = data - np.mean(data, axis=0)
        covariance = cov(data)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(covariance, k=n_components)
        E = np.real_if_close(eigvecs)
        return E.T @ A
    else:    
        A = data - np.mean(data, axis=0)
        covariance = cov(data)
        eigvals, eigvecs = np.linalg.eig(covariance)
        indices = np.argsort(eigvals)[::-1]
        E = np.real_if_close(eigvecs[:, indices[:n_components]])
        return E.T @ A

if __name__ == '__main__':
    dataset = np.random.rand(5, 100)
    plt.subplot(1, 2, 1)
    plt.scatter(dataset[0], dataset[1])
    new_data = pca(dataset, 2)
    plt.subplot(1, 2, 2)
    plt.scatter(new_data[0], new_data[1])
    plt.show()
    
    

