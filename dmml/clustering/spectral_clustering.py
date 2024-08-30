import sys
sys.path.append(f'{__file__}/../../../')
import numpy as np

def spectral_projection(dataset: np.ndarray, num_dimensions: int) -> np.ndarray:
    N = len(dataset)
    A = np.zeros((N, N)) # similarity matrix
    
    dataset_z = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    
    neighbor_matrix = np.zeros((N, N))
    for i in range(N):
        indices = [j for j in range(N)]
        indices.sort(key=lambda j: np.linalg.norm(dataset_z[i] - dataset_z[j]))
        for k, j in enumerate(indices):
            neighbor_matrix[i, j] = k
            
    def index_similarity_func(i: int, j: int) -> float:
        return np.exp(-neighbor_matrix[i, j]**2 * 0.01)
    for i in range(N):
        for j in range(N):
            A[i, j] = index_similarity_func(i, j)
    D = np.diag(np.sum(A, axis=1)) # degree matrix
    # normalized Laplacian matrix
    L = np.linalg.inv(D)**0.5 @ A @ np.linalg.inv(D)**0.5
    #L = D - A # unnormalized Laplacian matrix
    
    eigvals, eigvecs = np.linalg.eig(L)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    # sort eigenvectors by eigenvalues
    indices = np.argsort(eigvals)[::-1]
    indices = indices[1:1+num_dimensions]
    return eigvecs[:, indices]
    
if __name__ == "__main__":
    from dmml.clustering.datasets import random_dataset, demo_dataset
    from dmml.clustering.k_means import k_means
    from dmml.utils.colors import RANDOM_COLORS
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
    
    