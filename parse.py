import time
import pandas as pd
import kmedoid
import pam

df_directed = pd.read_csv("datasets/CA-GrQc.txt", sep="\t", comment="#", header=None, names=['FromNodeId', 'ToNodeId'])
df_undirected = pd.read_csv("datasets/com-dblp.ungraph.txt", sep="\t", comment="#", header=None, names=['FromNodeId', 'ToNodeId'])

print(kmedoid.k_medoid_clarans(df_directed, True, k=5, l=2, m=1))

def pam_results(df):
    start_time = time.time()
    clusters, cost, history = pam.k_medoid_pam(df, is_directed=False, k=5, random_state=0)
    end_time = time.time()

    print("\nPAM Performance:")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    print(f"Swaps to converge: {len(history) - 1}")
    print(f"Initial cost: {history[0]}")
    print(f"Final cost: {cost}")

    improvement = history[0] - cost
    improvement_percent = (improvement / history[0]) * 100

    print(f"Cost reduction: {improvement}")
    print(f"Percent improvement: {improvement_percent:.2f}%")

    cluster_sizes = clusters["Medoid"].value_counts()

    print("\nPAM Cluster Summary:")
    print(f"{'Medoid':<10}{'Cluster Size':>10}")

    for medoid, size in cluster_sizes.items():
        print(f"{medoid:<10}{size:>10}")

pam_results(df_directed)