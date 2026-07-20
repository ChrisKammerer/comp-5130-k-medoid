from typing import Callable
import random
import pandas as pd
import numpy as np
import networkx as nx


def KMedoid(df: pd.DataFrame, k: int, max_iter: int = 100) -> pd.DataFrame:
    """
    Perform K-Medoids clustering on the given DataFrame.

    Args:
        df (DataFrame): DataFrame with columns 'FromNodeId' and 'ToNodeId'
        k (int): number of clusters
        max_iter (int): maximum number of iterations

    Returns:
        DataFrame: DataFrame with columns 'FromNodeId', 'ToNodeId', and 'ClusterId'
    """
    ...

def k_medoid_clarans(
        df: pd.DataFrame,
        is_directed: bool,
        k: int,
        l: int = 100,
        m: int = 1
    ) ->  pd.DataFrame:
    """
    Perform K-Medoids clustering using the CLARANS algorithm on the given DataFrame.

    Args:
        df (DataFrame): DataFrame with columns 'FromNodeId' and 'ToNodeId'
        is_directed (bool): input if graph is directed or undirected, affects which distance algorithm is used
        k (int): number of clusters
        l (int): number randomized neighbors selected
        m (int): number of algorithm repeats

    Returns:
        DataFrame with columns 'Medoid' and 'Node'

    """
    G = nx.from_pandas_edgelist(
        df,
        'FromNodeId',
        'ToNodeId',
        create_using=(nx.DiGraph() if is_directed else nx.Graph())
    )
    nodes = list(G.nodes())
    best_medoids = None
    best_cost = float('inf')

    for _ in range(m):
        current_medoids = set(random.sample(nodes, k))
        current_cost = _cost(G, current_medoids, nodes)
        j = 0
        while j < l:
            old = random.choice(list(current_medoids))
            new = random.choice([n for n in nodes if n not in current_medoids])
            candidate = (current_medoids - {old}) | {new}
            candidate_cost = _cost(G, candidate, nodes)
            if candidate_cost < current_cost:
                current_medoids, current_cost = candidate, candidate_cost
                j = 0
            else:
                j += 1
        if current_cost < best_cost:
            best_medoids, best_cost = current_medoids, current_cost
    clusters = _assign_clusters(G, best_medoids, nodes)
    _bfs_cache.clear()
    return clusters, best_cost

_bfs_cache = {}
def _dist_from(G: nx.Graph, node: int) -> dict:
    if node not in _bfs_cache:
        _bfs_cache[node] = nx.single_source_shortest_path_length(G, node)
    return _bfs_cache[node]

def _distance(G: nx.Graph, u: int, v: int):
    PENALTY = 100000
    d = _dist_from(G, u)
    return d.get(v, PENALTY) # penalty if node is unreachable

def _cost(G: nx.Graph, medoids: dict, nodes: dict) -> int:
    total = 0
    for n in nodes:
        total += min(_distance(G, m, n) for m in medoids)
    return total

def _assign_clusters(G: nx.Graph, medoids: set, nodes: list) -> pd.DataFrame:
    """Assign each node to its nearest medoid, recording each as a (Medoid, Node) row."""
    rows = [(min(medoids, key=lambda m: _distance(G, m, n)), n) for n in nodes]
    return pd.DataFrame(rows, columns=['Medoid', 'Node'])
    