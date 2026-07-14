import pandas as pd


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