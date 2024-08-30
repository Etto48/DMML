from dmml.utils.colors import RANDOM_COLORS, avg_colors
from matplotlib import pyplot as plt

def plot_dataset(dataset, color_function=None, ax=None):
    ax = ax or plt.gca()
    if color_function is not None:
        colors = [color_function(i, x) for i, x in enumerate(dataset)]
        ax.scatter([x[0] for x in dataset], [x[1] for x in dataset], c=colors)
    else:
        ax.scatter([x[0] for x in dataset], [x[1] for x in dataset])
        
def plot_clusters(dataset: list[list[float]], clusters: list[set[int]], sparse_adjacency: list[list[tuple[int, float]]] = None, ax=None):
    ax = ax or plt.gca()
    colors = RANDOM_COLORS
    color_lookup = {}
    for i, subset in enumerate(clusters):
        if len(subset) == 0:
            continue
        color_hash = list(subset)[0] % len(colors)
        for j in subset:
            color_lookup[j] = colors[color_hash]

    plot_dataset(dataset, lambda i,x: color_lookup.get(i, 'black'), ax)
    if sparse_adjacency is not None:
        for i, knn in enumerate(sparse_adjacency):
            for j, distance in knn:
                color = avg_colors(avg_colors(color_lookup[i], color_lookup[j]), "#000000")
                alpha = 1 / (distance + 1)
                ax.plot([dataset[i][0], dataset[j][0]], [dataset[i][1], dataset[j][1]], color, alpha=alpha, lw=0.5)
    
if __name__ == "__main__":
    from dmml.clustering.datasets import demo_dataset
    dataset, labels = demo_dataset(500, "smiley", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.show()