from typing import Literal
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm

def spectral_projection(dataset: np.ndarray, num_dimensions: int, similarity_function: Literal["exp", "nn"] = "exp", sparse=True) -> np.ndarray:
    N = dataset.shape[0]
    A = scipy.sparse.lil_matrix((N, N)) if sparse else np.zeros((N, N))
    dataset_z = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    if similarity_function == "nn":
        neighbor_matrix = np.zeros((N, N))
        for i in range(N):
            indices = [j for j in range(N)]
            indices.sort(key=lambda j: np.linalg.norm(dataset_z[i] - dataset_z[j]))
            for k, j in enumerate(indices):
                neighbor_matrix[i, j] = k
                
        def index_similarity_func(i: int, j: int) -> float:
            value = np.exp(-neighbor_matrix[i, j]**2 * 0.01)
            if value < 1e-2:
                return 0
            return value
        
    elif similarity_function == "exp":
        def index_similarity_func(i: int, j: int) -> float:
            if isinstance(dataset_z, scipy.sparse.spmatrix):
                value = np.exp(-scipy.sparse.linalg.norm(dataset_z[i] - dataset_z[j])**2 * 10)
            else:
                value = np.exp(-np.linalg.norm(dataset_z[i] - dataset_z[j])**2 * 10)
            if value < 1e-2:
                return 0
            return value
    
    if similarity_function == "nn":
        for i in tqdm(range(N), "Calculating similarity matrix"):
            for j in range(N):
                A[i, j] = index_similarity_func(i, j)
    elif similarity_function == "exp":
        for i in tqdm(range(N), "Calculating similarity matrix"):
            for j in range(i,N):
                A[i, j] = index_similarity_func(i, j)
                A[j, i] = A[i, j]
    if sparse:       
        D = scipy.sparse.lil_matrix((N, N))
        for i in tqdm(range(N), "Calculating D"):
            D[i, i] = np.sum(A[i, :])
        D = scipy.sparse.csc_matrix(D)
        D = scipy.sparse.linalg.inv(D)
        for i in tqdm(range(N), "Calculating D^-0.5"):
            D[i, i] = np.sqrt(D[i, i])
    else:
        D = np.zeros((N, N))
        for i in tqdm(range(N), "Calculating D"):
            D[i, i] = np.sum(A[i, :])
        D = np.linalg.inv(D) ** 0.5 
    # normalized Laplacian matrix
    L = D @ A @ D
    
    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k=min(num_dimensions+1, N-1), which="LM")
    else:
        eigvals, eigvecs = scipy.linalg.eigh(L)
    
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    # sort eigenvectors by eigenvalues
    indices = np.argsort(eigvals)[::-1]
    if eigvals[indices[0]] == 1:
        offset = 1
    else:
        offset = 0
    indices = indices[offset:offset+num_dimensions]
    return eigvecs[:, indices]
    
if __name__ == "__main__":
    from src.clustering.datasets import random_dataset, demo_dataset
    from src.clustering.k_means import k_means
    from src.utils.colors import RANDOM_COLORS
    import matplotlib.pyplot as plt
    N_DIM = 2
    N_CLUSTERS = 2
    #dataset_og = np.array(random_dataset(500, 0.05, [0 for _ in range(N_DIM)], [1 for _ in range(N_DIM)], n_clusters=N_CLUSTERS))
    dataset_og = np.array(demo_dataset(1000, "two_moons"))
    #dataset_og = np.array(demo_dataset(1000, "smiley"))
    #dataset_og = np.array(demo_dataset(1000, "clusters"))
    dataset_proj = spectral_projection(dataset_og, 2)
    
    clusters = k_means(dataset_proj, N_CLUSTERS)
    colors = RANDOM_COLORS
    
    fig = plt.figure()
    if dataset_og.shape[1] == 2:
        ax = fig.add_subplot(1,2,1)
        ax.set_title("Original dataset")
        for i, cluster in enumerate(clusters):
            cluster = np.array(list(cluster))
            ax.scatter(dataset_og[list(cluster), 0], dataset_og[list(cluster), 1], label=f"Cluster {i}", color=colors[i])
    elif dataset_og.shape[1] > 2:
        ax = fig.add_subplot(1,2,1, projection="3d")
        ax.set_title("Original dataset")
        for i, cluster in enumerate(clusters):
            cluster = np.array(list(cluster))
            ax.scatter(dataset_og[list(cluster), 0], dataset_og[list(cluster), 1], dataset_og[list(cluster), 2], label=f"Cluster {i}", color=colors[i])
    else:
        raise ValueError("Dataset must have at least 2 dimensions")
    ax = fig.add_subplot(1,2,2)
    ax.set_title("Projected dataset")
    for i, cluster in enumerate(clusters):
        cluster = np.array(list(cluster))
        cluster_data = dataset_proj[list(cluster), :]
        ax.scatter(cluster_data[:,0], cluster_data[:,1], label=f"Cluster {i}", color=colors[i])
    
    plt.show()
    
    