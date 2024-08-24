import sys
sys.path.append(f'{__file__}/../../../')
from dmml.utils.colors import RANDOM_COLORS, str_to_rgb, rgb_to_name
from dmml.clustering.plotting import plot_clusters
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
    
def split_graph_spectral(graph: nx.Graph, indices: set[int], at_all_costs: bool = False) -> tuple[set[int], set[int]]:
    
    if len(indices) < 2:
        return indices, set()
    
    subgraph: nx.Graph = graph.subgraph(indices)
    
    source = np.random.choice(list(indices))
    target = np.random.choice(list(indices - {source}))
    
    flow, (partition_a, partition_b) = nx.minimum_cut(subgraph, source, target, capacity='weight')
    
    minimum_percentage = 0.3
    minimum_percentage_at_all_costs = 0.25
    
    if (len(partition_a) > len(indices) * minimum_percentage and len(partition_b) > len(indices) * minimum_percentage) or \
        (at_all_costs and len(partition_a) > len(indices) * minimum_percentage_at_all_costs and len(partition_b) > len(indices) * minimum_percentage_at_all_costs) or \
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
                    if best_interconnectivity == np.inf:
                        break
        if clusters_to_merge is not None:
            new_clusters = [clusters[i] for i in range(len(clusters)) if i not in clusters_to_merge]
            new_clusters.append(clusters[clusters_to_merge[0]] | clusters[clusters_to_merge[1]])
            clusters = new_clusters
        else:
            break
    if interactive:
        plt.ioff()
        plt.clf()
        plot_clusters(dataset, clusters, knn_list)
        for i, cluster in enumerate(clusters):
            cluster_color = RANDOM_COLORS[list(cluster)[0] % len(RANDOM_COLORS)]
            cluster_color_name = rgb_to_name(*str_to_rgb(cluster_color))
            print(f"Cluster {i}: {len(cluster)} elements {cluster_color_name}")
        plt.show()
    return clusters

if __name__ == "__main__":
    from dmml.clustering.datasets import random_dataset, dist
    dataset = random_dataset(500, 0.05, [0, 0], [10, 10])
    K = 5
    MIN_SIZE = 20
    MIN_CLUSTERS = 2
    MIN_RI = 0.4
    clusters = chameleon(dataset, K, MIN_SIZE, MIN_CLUSTERS, MIN_RI, interactive=True)