from dmml.outlier_detection.angle_based import var_cosine_similarity
import numpy as np

def angle_based_clustering_tendency(dataset: list[list[float]], num_samples: int) -> float:
    N = len(dataset)
    N_dim = len(dataset[0])
    
    ranges = [[dataset[0][i], dataset[0][i]] for i in range(N_dim)]
    for i in range(N):
        for j in range(N_dim):
            if ranges[j][0] > dataset[i][j]:
                ranges[j][0] = dataset[i][j]
            if ranges[j][1] < dataset[i][j]:
                ranges[j][1] = dataset[i][j]
    
    p = [np.random.randint(0, N) for _ in range(N)]
    q = [[np.random.uniform(ranges[j][0], ranges[j][1]) for j in range(N_dim)] for _ in range(N)]
    
    random_measures = num_samples
    
    p_similarity = [var_cosine_similarity(np.array(dataset), i, random_measures) for i in p]
    q_similarity = [var_cosine_similarity(np.array(dataset), np.array(i), random_measures) for i in q]
    
    a = sum(p_similarity) / (sum(p_similarity) + sum(q_similarity))
    
    return a

if __name__ == "__main__":
    from dmml.clustering.datasets import random_dataset, demo_dataset
    from dmml.clustering.plotting import plot_dataset
    import matplotlib.pyplot as plt
    
    #dataset = random_dataset(500, 0.05, [0, 0], [10, 10])
    dataset = demo_dataset(500, "three_lines")
    uniform = demo_dataset(500, "uniform")
    h_d = angle_based_clustering_tendency(dataset, 100)
    h_u = angle_based_clustering_tendency(uniform, 100)
    plt.subplot(1, 2, 1)
    plot_dataset(dataset)
    plt.title(f"ABCT test: {h_d}")
    plt.subplot(1, 2, 2)
    plot_dataset(uniform)
    plt.title(f"ABCT test: {h_u}")
    plt.show()
    