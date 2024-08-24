import numpy as np
import matplotlib.pyplot as plt

def cov(data: np.ndarray) -> np.ndarray:
    A = data
    mean = np.mean(data, axis=-1)
    for i in range(A.shape[-1]):
        A[:, i] -= mean
    return A @ A.T / (A.shape[1] - 1)

def pca(data: np.ndarray, n_components: int) -> np.ndarray:
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
    
    

