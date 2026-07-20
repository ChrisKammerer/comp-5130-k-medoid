from typing import Callable
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
        DataFrame: DataFrame with columns 'FromNodeId', 'ToNodeId', and 'ClusterId'

    """
    medoids = {}
    distance_function = None
    G = nx.from_pandas_edgelist(
        df,
        'FromNodeId',
        'ToNodeId',
        create_using=(nx.DiGraph() if is_directed else nx.Graph())
    )


_bfs_cache = {}
def _dist_from(G: nx.Graph, node: int) -> dict:
    if node not in _bfs_cache:
        _bfs_cache[node] = nx.single_source_shortest_path_length(G, node)
    return _bfs_cache[node]

def _distance(G: nx.Graph, u: int, v: int):
    PENALTY = 100000
    d = _dist_from(G, u)
    return d.get(v, PENALTY) # penalty if node is unreachable
    