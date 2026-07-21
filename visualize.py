import time
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

import pam
import kmedoid

sns.set_theme(style="whitegrid")

def generate_visualizations_pipeline(dataset_path: str, is_directed: bool, dataset_label: str, file_suffix: str):
    print("Loading dataset...")
    df_raw = pd.read_csv(
        dataset_path, 
        sep="\t", 
        comment="#", 
        header=None, 
        names=['FromNodeId', 'ToNodeId']
    )

    # Pre-process graph to largest connected component
    G_full = pam.build_graph(df_raw, is_directed=is_directed)
    G_clean = pam.restrict_to_largest_component(G_full, is_directed=is_directed)
    
    # select edges for cluster processing
    df_clean = nx.to_pandas_edgelist(G_clean, source='FromNodeId', target='ToNodeId')

    print("Running PAM")
    t0_pam = time.perf_counter()
    clusters_pam, cost_pam, history_pam = pam.k_medoid_pam(
        df_clean, 
        is_directed=is_directed, 
        k=5, 
        random_state=0
    )
    time_pam = time.perf_counter() - t0_pam

    print("Running CLARANS")
    t0_clarans = time.perf_counter()
    clusters_clarans, cost_clarans = kmedoid.k_medoid_clarans(
        df_clean, 
        is_directed=is_directed, 
        k=5, 
        l=20, 
        m=2
    )
    time_clarans = time.perf_counter() - t0_clarans

    print(f"PAM: Cost = {cost_pam:.1f}, Time = {time_pam:.2f}s")
    print(f"CLARANS: Cost = {cost_clarans:.1f}, Time = {time_clarans:.2f}s")

    # PAM Cost Convergence per Iteration graph
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(history_pam)), history_pam, marker='o', color='#2b5c8f', linewidth=2)
    plt.title(f"PAM Iterative Cost Reduction ({dataset_label}, k=5)", fontsize=12, fontweight='bold')
    plt.xlabel("Swap Iteration", fontsize=10)
    plt.ylabel("Total Graph Distance Cost", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"pam_convergence_{file_suffix}.png", dpi=300)
    plt.close()

    # Cluster Size Distribution Comparison
    plt.figure(figsize=(9, 4.5))
    counts_pam = clusters_pam['Medoid'].value_counts().reset_index()
    counts_pam.columns = ['Medoid', 'Count']
    counts_pam['Algorithm'] = 'PAM'
    counts_pam['Medoid_Label'] = "PAM-" + counts_pam['Medoid'].astype(str)

    counts_clarans = clusters_clarans['Medoid'].value_counts().reset_index()
    counts_clarans.columns = ['Medoid', 'Count']
    counts_clarans['Algorithm'] = 'CLARANS'
    counts_clarans['Medoid_Label'] = "CLARANS-" + counts_clarans['Medoid'].astype(str)

    combined_counts = pd.concat([counts_pam, counts_clarans])

    sns.barplot(
        data=combined_counts, 
        x='Medoid_Label', 
        y='Count', 
        hue='Algorithm', 
        palette='Set2', 
        dodge=False
    )
    plt.title(f"Cluster Size Distribution: PAM vs. CLARANS ({dataset_label})", fontsize=12, fontweight='bold')
    plt.xlabel("Medoid ID", fontsize=10)
    plt.ylabel("Assigned Node Count", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"cluster_distribution_{file_suffix}.png", dpi=300)
    plt.close()

    # Comparison Table pam_vs_clarans_tradeoff.png
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.axis('tight')

    speedup = time_pam / time_clarans if time_clarans > 0 else 0
    cost_diff_pct = ((cost_clarans - cost_pam) / cost_pam) * 100

    table_data = [
        ["Execution Time (s)", f"{time_pam:.2f} s", f"{time_clarans:.2f} s", f"CLARANS ({speedup:.1f}x faster)"],
        ["Total Clustering Cost", f"{cost_pam:,.1f}", f"{cost_clarans:,.1f}", f"PAM ({cost_diff_pct:.1f}% lower)"]
    ]
    
    col_labels = ["Metric", "PAM", "CLARANS", "Advantage"]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2b5c8f')
        else:
            cell.set_facecolor('#f7f9fb' if row % 2 == 0 else '#ffffff')

    plt.title(f"Algorithm Performance Summary: PAM vs. CLARANS ({dataset_label})", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"pam_vs_clarans_tradeoff_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # network clusters graph
    print(f"network_clusters_{file_suffix}.png being created")
    medoids = list(clusters_pam['Medoid'].unique())
    
    nodes_to_draw = set(medoids)
    for m in medoids:
        if m in G_clean:
            neighbors = list(G_clean.neighbors(m))[:25]
            nodes_to_draw.update(neighbors)
            
    subG = G_clean.subgraph(nodes_to_draw)
    node_to_medoid = dict(zip(clusters_pam['Node'], clusters_pam['Medoid']))
    
    pos = nx.spring_layout(subG, k=0.3, seed=42)
    
    plt.figure(figsize=(10, 8))
    
    nx.draw_networkx_edges(subG, pos, alpha=0.2, edge_color='gray')
    cmap = plt.get_cmap('tab10')
    
    for idx, m in enumerate(medoids):
        # Find nodes belonging to this specific medoid/cluster
        cluster_nodes = [
            n for n in subG.nodes() 
            if n not in medoids and node_to_medoid.get(n) == m
        ]
        
        if cluster_nodes:
            nx.draw_networkx_nodes(
                subG, pos, 
                nodelist=cluster_nodes, 
                node_size=45, 
                node_color=[cmap(idx % 10)], 
                label=f"Cluster {m}",
                alpha=0.85
            )
    
    # Mediods marked by stars
    nx.draw_networkx_nodes(
        subG, pos, 
        nodelist=medoids, 
        node_size=280, 
        node_color='red', 
        node_shape='*', 
        edgecolors='black', 
        linewidths=1.2,
    )
    
    plt.title(f"PAM Medoids & Local Neighborhoods ({dataset_label} Subsample)", fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Clusters & Medoids", frameon=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"network_clusters_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graphs generated for {dataset_label}.")

def generate_directed_visualizations():
    # CA-GrQc
    generate_visualizations_pipeline(
        dataset_path="datasets/CA-GrQc.txt",
        is_directed=True,
        dataset_label="CA-GrQc Directed",
        file_suffix="directed"
    )

def generate_undirected_visualizations():
    # undirected graph w/ com-dblp
    generate_visualizations_pipeline(
        dataset_path="datasets/com-dblp.ungraph.txt",
        is_directed=False,
        dataset_label="com-DBLP Undirected",
        file_suffix="undirected"
    )

if __name__ == "__main__":
    # directed graph visualization run
    generate_directed_visualizations()
    
    # undirected graph visualization run, uncomment when ready.
    
    # generate_undirected_visualizations()