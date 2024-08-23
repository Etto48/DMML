from random_dataset import random_dataset, plot_dataset, dist, demo_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

def dist_to_weight(dist: float) -> float:
    return 1 / (dist + 1)

def knn_to_nx_graph(knn_list, indices: set[int]) -> nx.Graph:
    nx_graph = nx.Graph()
    for i, knn in enumerate(knn_list):
        if i in indices:
            nx_graph.add_node(i)
            for j, distance in knn:
                if j in indices:
                    nx_graph.add_edge(i, j, weight=dist_to_weight(distance), distance=distance)
    return nx_graph

def str_to_color(color: str) -> tuple[int, int, int]:
    assert color[0] == "#"
    return tuple(int(color[i+1:i+3], 16) for i in (0, 2, 4))

def color_to_str(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"

def avg_colors(color_1: str, color_2: str) -> str:
    c1 = str_to_color(color_1)
    c2 = str_to_color(color_2)
    return color_to_str(tuple((c1[i] + c2[i]) // 2 for i in range(3)))

def rgb_to_hsv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    if delta == 0:
        return 0, 0, max_val
    if max_val == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif max_val == g:
        hue = 60 * (((b - r) / delta) + 2)
    else:
        hue = 60 * (((r - g) / delta) + 4)
    saturation = delta / max_val
    value = max_val / 255
    return int(hue), int(saturation), int(value)

def random_color() -> str:
    r = 0
    g = 0
    b = 0
    while True: 
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        h, s, v = rgb_to_hsv((r, g, b))
        if s > 0.3 and v > 0.3:
            break
    return color_to_str((r, g, b))

RANDOM_COLORS = [random_color() for _ in range(100)]

def plot_clusters(dataset, clusters, knn_list):
    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FF8000",
        "#FF0080",
        "#8000FF",
        "#80FF00",
        "#0080FF",
        "#00FF80",
        "#FF80FF",
        "#80FFFF",
        "#FFFF80",
        "#FF8080",
        "#80FF80",
        "#8080FF",
    ]
    colors = RANDOM_COLORS
    color_lookup = {}
    for i, subset in enumerate(clusters):
        if len(subset) == 0:
            print("Empty cluster")
            continue
        color_hash = list(subset)[0] % len(colors)
        for j in subset:
            color_lookup[j] = colors[color_hash]

    plot_dataset(dataset, lambda i,x: color_lookup.get(i, 'black'))
    for i, knn in enumerate(knn_list):
        for j, _ in knn:
            color = avg_colors(avg_colors(color_lookup.get(i, "#000000"), color_lookup.get(j, "#000000")), "#000000")
            plt.plot([dataset[i][0], dataset[j][0]], [dataset[i][1], dataset[j][1]], color, alpha=0.5, lw=0.5)
    plt.show()
    
def split_graph_spectral(graph: nx.Graph, indices: set[int], at_all_costs: bool = False) -> tuple[set[int], set[int]]:
    
    if len(indices) < 2:
        return indices, set()
    
    subgraph: nx.Graph = graph.subgraph(indices)
    
    source = np.random.choice(list(indices))
    target = np.random.choice(list(indices - {source}))
    
    flow, (partition_a, partition_b) = nx.minimum_cut(subgraph, source, target, capacity='weight')
    
    if (len(partition_a) * 3 > len(indices) and len(partition_b) * 3 > len(indices)) or \
        (at_all_costs and len(partition_a) * 4 > len(indices) and len(partition_b) * 4 > len(indices)) or \
        flow == 0:
        return partition_a, partition_b
    
    vertices = list(subgraph.nodes)
    laplacian = nx.laplacian_matrix(subgraph, weight="weight").toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    sorted_indices = np.argsort(eigenvalues)
    fiedler_vector = eigenvectors[:, sorted_indices[1]]
    
    partition_a = [v for i,v in enumerate(vertices) if fiedler_vector[i] < 0]
    partition_b = [v for i,v in enumerate(vertices) if fiedler_vector[i] >= 0]
    
    if len(partition_a) == 0:
        return set(partition_b), set()
    else:
        return set(partition_a), set(partition_b)

def relative_interconnectivity(graph: nx.Graph, cluster_a: set[int], cluster_b: set[int]) -> float:
    cut_size = nx.cut_size(graph, cluster_a, cluster_b, weight='weight')
    a_a, a_b = split_graph_spectral(graph, cluster_a)
    min_cut_size_a = nx.cut_size(graph, a_a, a_b, weight='weight')
    b_a, b_b = split_graph_spectral(graph, cluster_b)
    min_cut_size_b = nx.cut_size(graph, b_a, b_b, weight='weight')
        
    if min_cut_size_a + min_cut_size_b == 0:
        if cut_size == 0:
            return 0
        else:
            return np.inf
        
    return cut_size * 2 / (min_cut_size_a + min_cut_size_b)
    

def chameleon(dataset, K, min_size, min_clusters, min_ri, interactive=False):

    knn_list = []

    for i,d in enumerate(tqdm(dataset, "Calculating KNN")):
        knn = []
        for j,other in enumerate(dataset):
            if i != j:
                knn.append((j, dist(d, other)))
                knn.sort(key=lambda x: x[1])
                knn = knn[:K]
        knn_list.append(knn)
    
    subsets_list = []
    nx_graph = knn_to_nx_graph(knn_list, set(range(len(dataset))))
    set_a, set_b = split_graph_spectral(nx_graph, set(range(len(dataset))), at_all_costs=True)
        
    if len(set_b) > 0:
        subsets_list.append([set_a, set_b])
    else:
        subsets_list.append([set_a])

    while True:
        new_subsets = []
        end = True
        for subset in subsets_list[-1]:
            set_a, set_b = split_graph_spectral(nx_graph, subset)
            new_subsets.append(set_a)
            if len(set_b) > 0:
                new_subsets.append(set_b)
                end = False  
        biggest_subset_size = max(len(subset) for subset in new_subsets)
        if biggest_subset_size < min_size:
            break
        if end:
            break
        subsets_list.append(new_subsets)    

    clusters = [subset for subset in subsets_list[-1]]    
    
    if interactive:
        plt.ion()
    merge_operations = max(len(clusters) - min_clusters, 0)
    for i in tqdm(range(merge_operations), "Merging clusters"):
        if interactive:
            plt.clf()
            plot_clusters(dataset, clusters, knn_list)
            plt.pause(0.1)
        clusters_to_merge = None
        best_interconnectivity = 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                interconnectivity = relative_interconnectivity(nx_graph, clusters[i], clusters[j])
                if interconnectivity > best_interconnectivity and interconnectivity > min_ri:
                    best_interconnectivity = interconnectivity
                    clusters_to_merge = (i, j)
                    if best_interconnectivity == np.inf or ((len(clusters[i]) == 1 or len(clusters[j]) == 1) and interconnectivity > min_ri):
                        break
        if clusters_to_merge is not None:
            new_clusters = [clusters[i] for i in range(len(clusters)) if i not in clusters_to_merge]
            new_clusters.append(clusters[clusters_to_merge[0]] | clusters[clusters_to_merge[1]])
            clusters = new_clusters
        else:
            break
    if interactive:
        plt.ioff()
        plot_clusters(dataset, clusters, knn_list)
        plt.show()
    return clusters

if __name__ == "__main__":
    dataset = random_dataset(500, 0.1, 0.05, [0, 0], [10, 10])
    K = 5
    MIN_SIZE = 20
    MIN_CLUSTERS = 2
    MIN_RI = 0.3
    clusters = chameleon(dataset, K, MIN_SIZE, MIN_CLUSTERS, MIN_RI, interactive=True)