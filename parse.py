import pandas as pd
import kmedoid

df_directed = pd.read_csv("datasets/CA-GrQc.txt", sep="\t", comment="#", header=None, names=['FromNodeId', 'ToNodeId'])
df_undirected = pd.read_csv("datasets/com-dblp.ungraph.txt", sep="\t", comment="#", header=None, names=['FromNodeId', 'ToNodeId'])

print(kmedoid.k_medoid_clarans(df_directed, True, k=5, l=2, m=1))