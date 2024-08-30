import numpy as np

def dist(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    return sum((a[i] - b[i])**2 for i in range(len(a)))**0.5

def nearest_neighbour(dataset: list[list[float]], point: int | list[float]) -> int:
    if isinstance(point, int):
        i = point
        point = dataset[i]
    else:
        i = -1
    min_dist = float("inf")
    nearest = None
    for j in range(len(dataset)):
        if i != j:
            d = dist(point, dataset[j])
            if d < min_dist:
                min_dist = d
                nearest = j
    assert nearest is not None
    return nearest

def hopkins_test(dataset: list[list[float]], num_samples: int) -> float:
    N = len(dataset)
    N_dim = len(dataset[0])
    ranges = [[dataset[0][i], dataset[0][i]] for i in range(N_dim)]
    for i in range(N):
        for j in range(N_dim):
            if ranges[j][0] > dataset[i][j]:
                ranges[j][0] = dataset[i][j]
            if ranges[j][1] < dataset[i][j]:
                ranges[j][1] = dataset[i][j]
    
    n = num_samples
    p = [np.random.randint(0, N) for _ in range(n)]
    q = [[np.random.uniform(ranges[j][0], ranges[j][1]) for j in range(N_dim)] for _ in range(n)]
    
    min_p_distances = [dist(dataset[nearest_neighbour(dataset, i)], dataset[i]) for i in p]
    min_q_distances = [dist(dataset[nearest_neighbour(dataset, i)], i) for i in q]
    
    h = sum(min_q_distances) / (sum(min_p_distances) + sum(min_q_distances))
    
    return h

if __name__ == "__main__":
    from dmml.clustering.datasets import random_dataset, demo_dataset
    from dmml.clustering.plotting import plot_dataset
    import matplotlib.pyplot as plt
    
    #dataset = random_dataset(500, 0.05, [0, 0], [10, 10])
    dataset = demo_dataset(500, "three_lines")
    uniform = demo_dataset(500, "uniform")
    h_d = hopkins_test(dataset, 50)
    h_u = hopkins_test(uniform, 50)
    plt.subplot(1, 2, 1)
    plot_dataset(dataset)
    plt.title(f"Hopkins test: {h_d}")
    plt.subplot(1, 2, 2)
    plot_dataset(uniform)
    plt.title(f"Hopkins test: {h_u}")
    plt.show()
    