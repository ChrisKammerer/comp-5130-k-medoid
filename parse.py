import pandas as pd


df_directed = pd.read_csv("datasets/CA-GrQc.txt", sep="\t", comment="#", header=None, names=['FromNodeId', 'ToNodeId'])

df_undirected = pd.read_csv("datasets/com-dblp.ungraph.txt", sep="\t", comment="#", header=None, names=['FromNodeId', 'ToNodeId'])