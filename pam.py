import numpy as np
import pandas as pd
import networkx as nx


# preprocessing #

#build a networkx graph
def build_graph(df: pd.DataFrame, is_directed: bool) -> nx.Graph:
    return nx.from_pandas_edgelist(
        df, 'FromNodeId', 'ToNodeId',
        create_using=(nx.DiGraph() if is_directed else nx.Graph())
    )

#keep only largest connected component since distance cannot be calculated for disconnected components
def restrict_to_largest_component(G: nx.Graph, is_directed: bool) -> nx.Graph:
    if is_directed:
        largest_connected_component = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_connected_component = max(nx.connected_components(G), key=len)

    return G.subgraph(largest_connected_component).copy()

#precomputes shortest path distance matrix once, so cost computation is an array lookup
def build_distance_matrix(G: nx.Graph, node_list: list) -> np.ndarray:
    N = len(node_list)
    index = {node: i for i, node in enumerate(node_list)}
    distance_matrix = np.full((N, N), np.inf)

    for source, lengths in nx.all_pairs_shortest_path_length(G):
        i = index[source]
        for target, distance in lengths.items():
            distance_matrix[i, index[target]] = distance

    return distance_matrix


# algorithm #

#randomly choose k objects as initial seeds, without replacement to avoid duplicate start points
def initialize_medoids(n: int, k: int, random_state: int = None) -> list:
    rng = np.random.default_rng(random_state)
    return list(rng.choice(n, size=k, replace=False))

#assign each object to nearest medoid
def assign_to_nearest_medoid(distance_matrix: np.ndarray, medoid_index: list) -> np.ndarray:
    medoid_distances = distance_matrix[medoid_index, :]
    assignments = medoid_distances.argmin(axis=0)

    return assignments

#total clustering cost for current set of medoids
def compute_total_cost(distance_matrix: np.ndarray, medoid_index: list) -> float:
    medoid_distances = distance_matrix[medoid_index, :]
    nearest_distances = medoid_distances.min(axis=0)
    total_cost = nearest_distances.sum()

    return total_cost

# considers swapping medoid for non medoid object and computes resulting cost
def find_best_swap(distance_matrix: np.ndarray, medoid_index: list, current_cost: float):
    N = distance_matrix.shape[0]
    non_medoid_index = [i for i in range(N) if i not in medoid_index]
    best_change = 0.0 #negative value indicates improvement
    best_swap = None #stores (position to replace, replacement)

    for medoid_position, medoid_to_remove in enumerate(medoid_index):
        #each point's cost if current_medoid is removed and not replaced
        remaining_medoids = [medoid for medoid in medoid_index if medoid != medoid_to_remove]
        if remaining_medoids:
            remaining_distances = distance_matrix[remaining_medoids, :]
            nearest_remaining_distance = remaining_distances.min(axis=0)
        else:
            nearest_remaining_distance = np.full(N, np.inf)
            
        #each point's cost if current_medoid is replaced
        candidate_distances = distance_matrix[non_medoid_index, :] #stores (number of non medoids, number of objects)
        new_assignment_distances = np.minimum(nearest_remaining_distance[None, :], candidate_distances)
        candidate_total_costs = new_assignment_distances.sum(axis=1)                         

        #finds replacement lowering total cost most
        j = int(np.argmin(candidate_total_costs))
        change = candidate_total_costs[j] - current_cost
        if change < best_change:
            best_change = change
            best_swap = (medoid_position, non_medoid_index[j])

    return best_swap, best_change

#replace medoid at given position with new candidate medoid
def apply_swap(medoid_index: list, swap: tuple) -> list:
    position, new_index = swap
    medoid_index = medoid_index.copy()
    medoid_index[position] = new_index
    return medoid_index

#repeatedly find best swap until convergence
def run_pam(distance_matrix: np.ndarray, k: int, max_iterations: int = 100, random_state: int = None):
    N = distance_matrix.shape[0]
    medoid_index = initialize_medoids(N, k, random_state)
    cost = compute_total_cost(distance_matrix, medoid_index)
    history = [cost]

    for _ in range(max_iterations):
        best_swap, best_change = find_best_swap(distance_matrix, medoid_index, cost)
        if best_swap is None:
            break #converged
        medoid_index = apply_swap(medoid_index, best_swap)
        cost = cost + best_change
        history.append(cost)

    return medoid_index, cost, history


#(Node, Medoid) dataframe assignment table
def build_cluster_dataframe(node_list: list, distance_matrix: np.ndarray, medoid_index: list) -> pd.DataFrame:
    assignments = assign_to_nearest_medoid(distance_matrix, medoid_index)
    medoid_nodes = [node_list[i] for i in medoid_index]
    rows = []

    for i in range(len(node_list)):
        node = node_list[i]
        cluster = medoid_nodes[assignments[i]]
        rows.append((node, cluster))

    return pd.DataFrame(rows, columns=['Node', 'Medoid'])

#entry point
def k_medoid_pam(df: pd.DataFrame, is_directed: bool, k: int,
                  max_iterations: int = 100, restrict_to_largest_connected_component: bool = True,
                  random_state: int = None):
    """
    Perform K-Medoids clustering using the Partitioning Around Medoids (PAM)
    algorithm on the given DataFrame.

    Args:
        df (DataFrame): DataFrame with columns 'FromNodeId' and 'ToNodeId'
        is_directed (bool): whether to build directed or undirected graph
        k (int): number of clusters
        max_iterations (int): cap on number of swap iterations
        restrict_to_largest_connected_component (bool): drops all but largest 
        connected component before clustering
        random_state (int): seed for initial medoid selection

    Returns:
        clusters: DataFrame with columns 'Node' and 'Medoid'
        cost: final total cost
        history: list of cost values per iteration (for convergence plots)

    """
    G = build_graph(df, is_directed)

    if restrict_to_largest_connected_component:
        G = restrict_to_largest_component(G, is_directed)

    node_list = list(G.nodes())
    N = len(node_list)

    if k >= N:
        raise ValueError(f"k={k} must be smaller than number of nodes={N}")

    distance_matrix = build_distance_matrix(G, node_list)

    if np.isinf(distance_matrix).any():
        raise ValueError(
            "This graph is not fully connected. Set restrict_to_largest_connected_component=True."
        )

    medoid_index, cost, history = run_pam(distance_matrix, k, max_iterations, random_state)
    clusters = build_cluster_dataframe(node_list, distance_matrix, medoid_index)

    return clusters, cost, history
