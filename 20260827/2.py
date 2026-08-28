"""
visualize_hi2_14.py
使用 NetworkX 加载并可视化 HI-II-14 人类蛋白质相互作用网络

HI-II-14 数据来源：
  Rolland et al., Cell 2014
  CCSB Human Interactome Database: http://interactome-atlas.org
  格式：制表符分隔文件，每行一对 Ensembl Gene ID

用法：
  python visualize_hi2_14.py                          # 使用本地文件
  python visualize_hi2_14.py --data_file HI-II-14.tsv # 指定文件路径
  python visualize_hi2_14.py --demo                   # 无数据时生成演示网络
"""

import os
import argparse
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境使用 Agg 后端
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from collections import Counter

# ============================================================
# 全局配置
# ============================================================
OUTPUT_DIR = "hi2_14_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配色方案
COLORS = {
    'hub':          '#E74C3C',   # Hub 蛋白（度 ≥ 10）
    'medium':       '#F39C12',   # 中等连接蛋白（度 5~9）
    'low':          '#3498DB',   # 低连接蛋白（度 1~4）
    'isolated':     '#BDC3C7',   # 孤立节点（度 = 0）
    'edge_default': '#D5D8DC',   # 默认边颜色
    'edge_highlight':'#E74C3C',  # 高亮边
    'bg':           '#FAFAFA',   # 背景色
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor':   COLORS['bg'],
    'font.size':        11,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
})


# ============================================================
# 第 1 部分：数据加载与图构建
# ============================================================

def load_hi2_14(filepath: str) -> nx.Graph:
    """
    从 TSV 文件加载 HI-II-14 数据，构建 NetworkX 无向图。

    HI-II-14 文件格式：
      - 制表符分隔（TSV）
      - 每行两个 Ensembl Gene ID，表示一对相互作用的蛋白
      - 可能有表头行（以 # 开头）

    Args:
        filepath: TSV 文件路径

    Returns:
        G: NetworkX 无向图
    """
    G = nx.Graph()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            cols = line.split('\t')
            if len(cols) < 2:
                continue

            protein_a = cols[0].strip()
            protein_b = cols[1].strip()

            # 跳过表头行
            if protein_a.lower() in ('protein_a', 'protein1',
                                      'interactor_a', 'gene_a'):
                continue

            # 跳过自环（蛋白与自身相互作用）
            if protein_a == protein_b:
                continue

            G.add_edge(protein_a, protein_b)

    return G


def generate_demo_network() -> nx.Graph:
    """
    生成一个模拟 HI-II-14 特征的演示网络。
    使用 Barabási-Albert 优先连接模型，模拟蛋白质网络的无标度特性。
    """
    # HI-II-14 约有 ~4000 个蛋白、~14000 条边
    # BA 模型参数：m=3 表示每个新节点连接 3 个已有节点
    G = nx.barabasi_albert_graph(500, 3, seed=42)

    # 将节点重命名为类似 Ensembl ID 的格式
    mapping = {n: f"ENSG{n:011d}" for n in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    return G


# ============================================================
# 第 2 部分：网络分析
# ============================================================

def analyze_network(G: nx.Graph) -> dict:
    """
    计算 PPI 网络的核心拓扑指标。

    Returns:
        stats: 包含各指标的字典
    """
    degrees = [d for _, d in G.degree()]
    degree_dict = dict(G.degree)

    stats = {
        'num_nodes':       G.number_of_nodes(),
        'num_edges':       G.number_of_edges(),
        'density':         nx.density(G),
        'avg_degree':      np.mean(degrees) if degrees else 0,
        'max_degree':      max(degrees) if degrees else 0,
        'min_degree':      min(degrees) if degrees else 0,
        'median_degree':   np.median(degrees) if degrees else 0,
        'num_components':  nx.number_connected_components(G),
    }

    # 最大连通子图
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)
    stats['largest_cc_nodes'] = G_lcc.number_of_nodes()
    stats['largest_cc_edges'] = G_lcc.number_of_edges()

    # 聚类系数（仅在最大连通子图上计算，避免孤立节点干扰）
    if G_lcc.number_of_nodes() > 0:
        stats['avg_clustering'] = nx.average_clustering(G_lcc)
    else:
        stats['avg_clustering'] = 0.0

    # 度分布统计
    degree_counts = Counter(degrees)
    stats['degree_distribution'] = degree_counts

    # Hub 蛋白（度排名前 20）
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    stats['top_hubs'] = top_hubs

    return stats


def print_stats(stats: dict):
    """打印网络统计信息"""
    print("\n" + "=" * 50)
    print("  HI-II-14 网络拓扑统计")
    print("=" * 50)
    print(f"  蛋白数（节点）:     {stats['num_nodes']:>8,}")
    print(f"  互作数（边）:       {stats['num_edges']:>8,}")
    print(f"  网络密度:           {stats['density']:>8.6f}")
    print(f"  平均度:             {stats['avg_degree']:>8.2f}")
    print(f"  最大度:             {stats['max_degree']:>8}")
    print(f"  中位数度:           {stats['median_degree']:>8.1f}")
    print(f"  连通分量数:         {stats['num_components']:>8}")
    print(f"  最大连通子图节点:   {stats['largest_cc_nodes']:>8,}")
    print(f"  最大连通子图边数:   {stats['largest_cc_edges']:>8,}")
    print(f"  平均聚类系数:       {stats['avg_clustering']:>8.4f}")
    print("=" * 50)

    print("\n  Top 20 Hub 蛋白（连接度最高）:")
    print(f"  {'排名':>4}  {'蛋白 ID':<16}  {'度':>6}")
    print(f"  {'─' * 4}  {'─' * 16}  {'─' * 6}")
    for rank, (protein_id, degree) in enumerate(stats['top_hubs'], 1):
        print(f"  {rank:>4}  {protein_id:<16}  {degree:>6}")


# ============================================================
# 第 3 部分：可视化
# ============================================================

def visualize_full_network(G: nx.Graph, stats: dict,
                           max_nodes: int = 300,
                           save_name: str = "01_full_network.png"):
    """
    可视化 1：完整 PPI 网络拓扑。

    对于大规模网络（节点数 > max_nodes），自动提取最大连通子图进行可视化。
    节点大小按度（连接数）缩放，颜色按度分档。
    Hub 蛋白（Top 10）标注 ID。
    """
    print(f"\n[可视化 1] 完整 PPI 网络拓扑")

    # 如果图太大，取最大连通子图
    if G.number_of_nodes() > max_nodes:
        largest_cc = max(nx.connected_components(G), key=len)
        G_vis = G.subgraph(largest_cc).copy()
        print(f"  网络较大（{G.number_of_nodes()} 节点），"
              f"取最大连通子图（{G_vis.number_of_nodes()} 节点）进行可视化")
    else:
        G_vis = G.copy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # ---- 布局算法 ----
    # 小图用 Kamada-Kawai（全局最优布局）
    # 大图用 spring_layout（力导向布局，更快）
    if G_vis.number_of_nodes() > 150:
        pos = nx.spring_layout(G_vis, k=1.0, iterations=100, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G_vis)

    # ---- 节点大小和颜色（按度分档）----
    degrees = dict(G_vis.degree)
    node_sizes = []
    node_colors = []

    for node in G_vis.nodes():
        deg = degrees[node]
        if deg >= 10:
            node_colors.append(COLORS['hub'])
            node_sizes.append(200 + deg * 8)
        elif deg >= 5:
            node_colors.append(COLORS['medium'])
            node_sizes.append(100 + deg * 6)
        elif deg >= 1:
            node_colors.append(COLORS['low'])
            node_sizes.append(60 + deg * 4)
        else:
            node_colors.append(COLORS['isolated'])
            node_sizes.append(40)

    # ---- 绘制边 ----
    nx.draw_networkx_edges(
        G_vis, pos,
        edge_color=COLORS['edge_default'],
        width=0.3,
        alpha=0.2,
        ax=ax,
    )

    # ---- 绘制节点 ----
    nx.draw_networkx_nodes(
        G_vis, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='#555555',
        linewidths=0.3,
        alpha=0.85,
        ax=ax,
    )

    # ---- 标注 Hub 蛋白（Top 10）----
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    hub_labels = {}
    for node, deg in sorted_nodes[:10]:
        hub_labels[node] = f"{node}\n(deg={deg})"

    nx.draw_networkx_labels(
        G_vis, pos,
        labels=hub_labels,
        font_size=7,
        font_color='#333333',
        font_weight='bold',
        ax=ax,
    )

    # ---- 图例 ----
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS['hub'], markersize=12,
               label=f'Hub (degree ≥ 10)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS['medium'], markersize=10,
               label=f'Medium (degree 5-9)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS['low'], markersize=8,
               label=f'Low (degree 1-4)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=10, framealpha=0.9)

    ax.set_title(
        f"HI-II-14 Human Protein-Protein Interaction Network\n"
        f"Nodes: {stats['num_nodes']:,} | Edges: {stats['num_edges']:,} | "
        f"Avg Degree: {stats['avg_degree']:.1f} | "
        f"Clustering: {stats['avg_clustering']:.4f}",
        fontsize=14, fontweight='bold',
    )
    ax.axis('off')

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_hub_neighborhood(G: nx.Graph, stats: dict,
                                save_name: str = "02_hub_neighborhood.png"):
    """
    可视化 2：Top Hub 蛋白的邻域子图。

    选取连接度最高的 3 个 Hub 蛋白，展示它们各自的 1-hop 邻域。
    """
    print(f"\n[可视化 2] Hub 蛋白邻域子图")

    top_hubs = stats['top_hubs'][:3]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for idx, (hub_id, hub_degree) in enumerate(top_hubs):
        ax = axes[idx]

        if hub_id not in G:
            ax.set_title(f"{hub_id}\n(not in network)")
            ax.axis('off')
            continue

        # 提取 hub 的 1-hop 邻域子图
        neighbors = list(G.neighbors(hub_id))
        subgraph_nodes = [hub_id] + neighbors
        G_sub = G.subgraph(subgraph_nodes).copy()

        # 布局：hub 在中心，邻居环绕
        pos = nx.circular_layout(G_sub)
        pos[hub_id] = (0, 0)  # hub 放在中心

        # 节点颜色
        node_colors = []
        node_sizes = []
        for node in G_sub.nodes():
            if node == hub_id:
                node_colors.append(COLORS['hub'])
                node_sizes.append(600)
            else:
                node_colors.append(COLORS['low'])
                node_sizes.append(200)

        # 绘制边
        nx.draw_networkx_edges(
            G_sub, pos,
            edge_color=COLORS['edge_highlight'],
            width=1.5,
            alpha=0.5,
            ax=ax,
        )

        # 绘制节点
        nx.draw_networkx_nodes(
            G_sub, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors='#333333',
            linewidths=1.5,
            alpha=0.9,
            ax=ax,
        )

        # 标注 hub 蛋白
        hub_label = {hub_id: f"{hub_id}\n(deg={hub_degree})"}
        nx.draw_networkx_labels(
            G_sub, pos,
            labels=hub_label,
            font_size=8,
            font_weight='bold',
            font_color='white',
            ax=ax,
        )

        ax.set_title(
            f"Hub: {hub_id}\n"
            f"Degree: {hub_degree} | Neighbors: {len(neighbors)}",
            fontsize=11, fontweight='bold',
        )
        ax.axis('off')

    fig.suptitle("Top Hub Proteins and Their 1-Hop Neighborhoods",
                 fontsize=15, fontweight='bold', y=1.02)

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_degree_distribution(G: nx.Graph, stats: dict,
                                   save_name: str = "03_degree_distribution.png"):
    """
    可视化 3：度分布直方图（对数坐标）。

    蛋白质相互作用网络通常呈现无标度（scale-free）特性，
    度分布近似幂律分布 P(k) ~ k^(-γ)。
    """
    print(f"\n[可视化 3] 度分布")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    degrees = [d for _, d in G.degree()]
    degree_counts = Counter(degrees)

    # ---- 左图：线性坐标直方图 ----
    ax = axes[0]
    sorted_degrees = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in sorted_degrees]

    ax.bar(sorted_degrees, counts, color=COLORS['low'],
           edgecolor='#555', alpha=0.8, width=0.8)
    ax.set_xlabel('Node Degree (k)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Degree Distribution (Linear Scale)', fontweight='bold')
    ax.axvline(x=stats['avg_degree'], color=COLORS['hub'],
               linestyle='--', linewidth=2, label=f"Avg = {stats['avg_degree']:.1f}")
    ax.legend(fontsize=10)

    # ---- 右图：对数坐标（检测幂律特性）----
    ax = axes[1]
    # 过滤掉度为 0 的节点
    nonzero_degrees = [d for d in sorted_degrees if d > 0]
    nonzero_counts = [degree_counts[d] for d in nonzero_degrees]

    ax.scatter(nonzero_degrees, nonzero_counts, color=COLORS['hub'],
               s=40, alpha=0.7, edgecolors='#333', linewidths=0.5)
    ax.set_xlabel('Node Degree (k)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Degree Distribution (Log-Log Scale)', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 添加幂律拟合线（简单线性回归）
    if len(nonzero_degrees) > 2:
        log_k = np.log10(nonzero_degrees)
        log_p = np.log10(nonzero_counts)
        coeffs = np.polyfit(log_k, log_p, 1)
        fit_line = np.poly1d(coeffs)
        k_fit = np.logspace(np.log10(min(nonzero_degrees)),
                            np.log10(max(nonzero_degrees)), 100)
        ax.plot(k_fit, 10**fit_line(np.log10(k_fit)),
                color='#333', linewidth=2, linestyle='--',
                label=f'P(k) ~ k^({coeffs[0]:.2f})')
        ax.legend(fontsize=10)

    fig.suptitle("HI-II-14 Degree Distribution Analysis",
                 fontsize=14, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_connected_components(G: nx.Graph, stats: dict,
                                    save_name: str = "04_connected_components.png"):
    """
    可视化 4：连通分量分析。

    展示最大连通子图和各孤立小组件。
    """
    print(f"\n[可视化 4] 连通分量分析")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- 左图：最大连通子图 ----
    ax = axes[0]
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()

    if G_lcc.number_of_nodes() > 300:
        # 对大子图取核心（度最高的节点）
        degrees_lcc = dict(G_lcc.degree)
        top_nodes = sorted(degrees_lcc.items(), key=lambda x: x[1], reverse=True)[:300]
        top_node_ids = [n for n, _ in top_nodes]
        G_lcc_vis = G_lcc.subgraph(top_node_ids).copy()
    else:
        G_lcc_vis = G_lcc

    pos = nx.spring_layout(G_lcc_vis, k=0.8, iterations=80, seed=42)

    degrees_vis = dict(G_lcc_vis.degree)
    node_sizes = [50 + degrees_vis[n] * 5 for n in G_lcc_vis.nodes()]
    node_colors = [COLORS['hub'] if degrees_vis[n] >= 10 else
                   COLORS['medium'] if degrees_vis[n] >= 5 else
                   COLORS['low'] for n in G_lcc_vis.nodes()]

    nx.draw_networkx_edges(G_lcc_vis, pos, edge_color='#DDD',
                           width=0.3, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(G_lcc_vis, pos, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#555',
                           linewidths=0.3, alpha=0.85, ax=ax)

    ax.set_title(
        f"Largest Connected Component\n"
        f"Nodes: {G_lcc.number_of_nodes():,} | Edges: {G_lcc.number_of_edges():,}",
        fontsize=12, fontweight='bold',
    )
    ax.axis('off')

    # ---- 右图：连通分量大小分布 ----
    ax = axes[1]
    cc_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)

    ax.bar(range(len(cc_sizes)), cc_sizes,
           color=COLORS['low'], edgecolor='#555', alpha=0.8)
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Component Size (Number of Nodes)')
    ax.set_title(
        f'Connected Component Sizes\n'
        f'Total: {len(cc_sizes)} components',
        fontsize=12, fontweight='bold',
    )
    ax.set_yscale('log')

    # 标注前 5 个分量的大小
    for i in range(min(5, len(cc_sizes))):
        ax.annotate(f'{cc_sizes[i]}', (i, cc_sizes[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')

    fig.suptitle("HI-II-14 Connected Component Analysis",
                 fontsize=14, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_l3_path_examples(G: nx.Graph,
                                save_name: str = "05_l3_path_examples.png"):
    """
    可视化 5：L3 路径示例。

    从网络中提取若干 L3 路径（u → w1 → w2 → v），
    在 PPI 网络上下文中高亮展示。

    L3 路径定义：
      4 个不同节点 u, w1, w2, v
      3 条边：u-w1, w1-w2, w2-v
      正样本：u 和 v 之间存在直接边
      负样本：u 和 v 之间不存在直接边
    """
    print(f"\n[可视化 5] L3 路径示例")

    # 提取正样本和负样本 L3 路径
    positive_paths = []
    negative_paths = []
    max_samples = 3

    nodes = list(G.nodes())
    random.seed(42)
    random.shuffle(nodes)

    for u in nodes:
        if len(positive_paths) >= max_samples and len(negative_paths) >= max_samples:
            break

        for w1 in G.neighbors(u):
            if len(positive_paths) >= max_samples and len(negative_paths) >= max_samples:
                break
            for w2 in G.neighbors(w1):
                if w2 == u:
                    continue
                for v in G.neighbors(w2):
                    if v == u or v == w1:
                        continue

                    path = (u, w1, w2, v)

                    if G.has_edge(u, v):
                        if len(positive_paths) < max_samples:
                            positive_paths.append(path)
                    else:
                        if len(negative_paths) < max_samples:
                            negative_paths.append(path)

    all_paths = positive_paths + negative_paths
    if not all_paths:
        print("  未找到足够的 L3 路径")
        return

    # 收集路径涉及的所有节点和边
    path_nodes = set()
    path_edges_pos = set()
    path_edges_neg = set()

    for i, (u, w1, w2, v) in enumerate(all_paths):
        path_nodes.update([u, w1, w2, v])
        edges = {(u, w1), (w1, w2), (w2, v)}
        if i < len(positive_paths):
            path_edges_pos.update(edges)
        else:
            path_edges_neg.update(edges)

    # 提取包含路径节点的子图
    context_nodes = set(path_nodes)
    for node in path_nodes:
        if node in G:
            context_nodes.update(G.neighbors(node))

    G_sub = G.subgraph(context_nodes).copy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    pos = nx.spring_layout(G_sub, k=1.5, iterations=80, seed=42)

    # 分类节点
    l3_nodes = list(path_nodes)
    other_nodes = [n for n in G_sub.nodes() if n not in path_nodes]

    # 绘制普通边
    all_path_edges = path_edges_pos | path_edges_neg
    normal_edges = [(u, v) for u, v in G_sub.edges()
                    if (u, v) not in all_path_edges and (v, u) not in all_path_edges]
    nx.draw_networkx_edges(G_sub, pos, edgelist=normal_edges,
                           edge_color='#E0E0E0', width=0.8, alpha=0.5, ax=ax)

    # 绘制正样本 L3 路径边（红色虚线）
    pos_edges = [(u, v) for u, v in path_edges_pos]
    nx.draw_networkx_edges(G_sub, pos, edgelist=pos_edges,
                           edge_color='#2ECC71', width=3,
                           style='dashed', alpha=0.8, ax=ax)

    # 绘制负样本 L3 路径边（蓝色虚线）
    neg_edges = [(u, v) for u, v in path_edges_neg]
    nx.draw_networkx_edges(G_sub, pos, edgelist=neg_edges,
                           edge_color='#E74C3C', width=3,
                           style='dashed', alpha=0.8, ax=ax)

    # 绘制普通节点
    nx.draw_networkx_nodes(G_sub, pos, nodelist=other_nodes,
                           node_size=120, node_color='#E8E8E8',
                           edgecolors='#AAA', linewidths=0.5, alpha=0.6, ax=ax)

    # 绘制 L3 路径节点（大圆圈高亮）
    nx.draw_networkx_nodes(G_sub, pos, nodelist=l3_nodes,
                           node_size=400, node_color='#A8D8EA',
                           edgecolors='#333', linewidths=1.5, alpha=0.95, ax=ax)

    # 标注 L3 路径节点角色
    role_labels = {}
    for i, (u, w1, w2, v) in enumerate(all_paths):
        prefix = "P" if i < len(positive_paths) else "N"
        role_labels[u] = f"{prefix}{i+1}:u"
        role_labels[w1] = f"{prefix}{i+1}:w₁"
        role_labels[w2] = f"{prefix}{i+1}:w₂"
        role_labels[v] = f"{prefix}{i+1}:v"

    nx.draw_networkx_labels(G_sub, pos, labels=role_labels,
                            font_size=7, font_weight='bold', ax=ax)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#A8D8EA',
               markersize=12, markeredgecolor='#333', label='L3 path node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E8E8E8',
               markersize=10, markeredgecolor='#AAA', label='Context node'),
        Line2D([0], [0], color='#2ECC71', linewidth=3, linestyle='--',
               label='Positive L3 path (u-v connected)'),
        Line2D([0], [0], color='#E74C3C', linewidth=3, linestyle='--',
               label='Negative L3 path (u-v not connected)'),
        Line2D([0], [0], color='#E0E0E0', linewidth=1, label='Network edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=10, framealpha=0.9)

    ax.set_title(
        "L3 Paths in HI-II-14 PPI Network\n"
        "(Green: Positive samples | Red: Negative samples)",
        fontsize=14, fontweight='bold',
    )
    ax.axis('off')

    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="HI-II-14 PPI 网络可视化（NetworkX）"
    )
    parser.add_argument(
        "--data_file", type=str, default="data/raw/HI-II-14.tsv",
        help="HI-II-14 TSV 文件路径（默认: HI-II-14.tsv）"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="无数据时使用演示网络（Barabási-Albert 模型）"
    )
    parser.add_argument(
        "--max_nodes", type=int, default=300,
        help="网络可视化最大节点数（默认: 300）"
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  HI-II-14 蛋白质相互作用网络可视化")
    print("  使用 NetworkX + Matplotlib")
    print("=" * 60)

    # ---- 步骤 1：加载数据 ----
    print(f"\n[步骤 1] 加载 PPI 数据")

    if args.demo or not os.path.exists(args.data_file):
        if not args.demo:
            print(f"  文件不存在: {args.data_file}")
            print(f"  请从 http://interactome-atlas.org 下载 HI-II-14 数据")
        print(f"\n  [演示模式] 使用 Barabási-Albert 模型生成模拟网络")
        G = generate_demo_network()
        print(f"  模拟网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    else:
        G = load_hi2_14(args.data_file)
        print(f"  已加载: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # ---- 步骤 2：网络分析 ----
    print(f"\n[步骤 2] 网络拓扑分析")
    stats = analyze_network(G)
    print_stats(stats)

    # ---- 步骤 3：可视化 ----
    print(f"\n[步骤 3] 生成可视化图")

    visualize_full_network(G, stats, max_nodes=args.max_nodes)
    visualize_hub_neighborhood(G, stats)
    visualize_degree_distribution(G, stats)
    visualize_connected_components(G, stats)
    visualize_l3_path_examples(G)

    # ---- 完成 ----
    print(f"\n{'=' * 60}")
    print(f"  全部完成！输出目录: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()