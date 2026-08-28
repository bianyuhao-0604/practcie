"""
graph_build_and_visualize.py
将 PPI 数据集构建为图对象并进行可视化。

用法:
    python graph_build_and_visualize.py --data_file data/raw/HI-II-14.tsv --mode all
"""

import os
import argparse
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import torch
from collections import defaultdict

# ============================================================
# 全局配置
# ============================================================
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配色方案
COLORS = {
    'node_default':    '#A8D8EA',
    'node_u':          '#FF6B6B',   # 起始蛋白 u
    'node_v':          '#4ECDC4',   # 目标蛋白 v
    'node_w1':         '#FFD93D',   # 中间蛋白 w1
    'node_w2':         '#6C5CE7',   # 中间蛋白 w2
    'node_vp0':        '#FF8A5C',   # 中心虚拟节点 v_P0
    'node_vpi':        '#A8E6CF',   # 路径虚拟节点 v_Pi
    'edge_default':    '#CCCCCC',
    'edge_l3':         '#FF6B6B',   # L3 路径边
    'edge_active':     '#2ECC71',   # 门控激活的边
    'edge_inactive':   '#E0E0E0',   # 门控抑制的边
    'bg':              '#FAFAFA',
    'positive':        '#2ECC71',
    'negative':        '#E74C3C',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['bg'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


# ============================================================
# 第 1 部分：从原始数据构建图
# ============================================================

def load_ppi_network(tsv_path: str) -> nx.Graph:
    """
    从 TSV 文件加载 PPI 网络，构建为 networkx 无向图。
    
    输入格式（HI-II-14.tsv）:
        protein_A    protein_B
        ENSG0001     ENSG0002
        ...
    
    输出:
        G: networkx.Graph 对象
    """
    G = nx.Graph()
    
    with open(tsv_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            
            protein_a = cols[0].strip()
            protein_b = cols[1].strip()
            
            # 跳过表头
            if protein_a.lower() in ('protein_a', 'protein1', 'interactor_a'):
                continue
            
            # 跳过自环
            if protein_a == protein_b:
                continue
            
            G.add_edge(protein_a, protein_b)
    
    print(f"[构建完成] {tsv_path}")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数:   {G.number_of_edges()}")
    
    return G


def load_shs_data(tsv_path: str) -> nx.MultiGraph:
    """
    从 SHS27k/SHS148k 文件加载，构建为带边属性的多重图。
    
    与 HI-II-14 的区别:
    - 同一对蛋白之间可能有多种交互类型（多重边）
    - 每条边带有 action 类型和 score
    """
    G = nx.MultiGraph()
    
    with open(tsv_path, 'r') as f:
        header = f.readline()  # 读取表头
        col_names = header.strip().split('\t')
        
        # 自动检测列索引
        col_map = {}
        for i, name in enumerate(col_names):
            name_lower = name.lower().strip()
            if name_lower in ('protein1', 'protein_a'):
                col_map['p1'] = i
            elif name_lower in ('protein2', 'protein_b'):
                col_map['p2'] = i
            elif name_lower in ('action', 'type'):
                col_map['action'] = i
            elif name_lower in ('score', 'combined_score'):
                col_map['score'] = i
    
    with open(tsv_path, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 2:
                continue
            
            p1 = cols[col_map.get('p1', 0)].strip()
            p2 = cols[col_map.get('p2', 1)].strip()
            action = cols[col_map.get('action', 2)].strip() if 'action' in col_map else 'binding'
            score = float(cols[col_map.get('score', 3)]) if 'score' in col_map else 1.0
            
            if p1 == p2:
                continue
            
            G.add_edge(p1, p2, action=action, score=score)
    
    print(f"[构建完成] {tsv_path}")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数:   {G.number_edges()}")
    
    return G


# ============================================================
# 第 2 部分：L3 路径提取
# ============================================================

def extract_l3_paths(G: nx.Graph, max_samples: int = 50,max_attempts=10000) -> dict:
    """
    从 PPI 网络中提取 L3 路径样本。
    
    L3 路径: u → w1 → w2 → v（4 个不同节点，3 条边）
    
    正样本: u 和 v 在 G 中存在直接边
    负样本: u 和 v 在 G 中不存在直接边
    
    Returns:
        {
            'positive': [(u, w1, w2, v), ...],
            'negative': [(u, w1, w2, v), ...],
        }
    """
    positive_paths = []
    negative_paths = []
    nodes = list(G.nodes())
    random.shuffle(nodes)
    edges_set = set()
    for u, v in G.edges():
        edges_set.add((u, v))
        edges_set.add((v, u))
    
    count = 0
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
                    
                    count += 1
                    if count > 500000:  # 防止遍历过久
                        break
                if count > 500000:
                    break
            if count > 500000:
                break
    
    print(f"[L3 路径提取] 正样本: {len(positive_paths)}, 负样本: {len(negative_paths)}")
    
    return {
        'positive': positive_paths,
        'negative': negative_paths,
    }


# ============================================================
# 第 3 部分：可视化函数
# ============================================================

def visualize_full_network(G: nx.Graph, title: str = "PPI Network",
                           max_nodes: int = 200, save_name: str = "01_full_network.png"):
    """
    可视化 1：完整 PPI 网络（大规模时取最大连通子图）。
    
    对于大规模网络（>max_nodes 节点），自动提取最大连通子图进行可视化。
    """
    print(f"\n[可视化 1] 完整 PPI 网络")
    
    # 如果图太大，取最大连通子图
    if G.number_of_nodes() > max_nodes:
        largest_cc = max(nx.connected_components(G), key=len)
        G_vis = G.subgraph(largest_cc).copy()
        print(f"  图太大（{G.number_of_nodes()} 节点），取最大连通子图（{G_vis.number_of_nodes()} 节点）")
    else:
        G_vis = G.copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # 布局算法
    if G_vis.number_of_nodes() > 100:
        pos = nx.spring_layout(G_vis, k=0.8, iterations=80, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G_vis)
    
    # 计算节点度（用于节点大小）
    degrees = dict(G_vis.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 400 * (degrees[n] / max_degree) for n in G_vis.nodes()]
    
    # 绘制节点
    nx.draw_networkx_nodes(
        G_vis, pos,
        node_size=node_sizes,
        node_color=COLORS['node_default'],
        edgecolors='#555555',
        linewidths=0.5,
        alpha=0.85,
        ax=ax,
    )
    
    # 绘制边
    nx.draw_networkx_edges(
        G_vis, pos,
        edge_color=COLORS['edge_default'],
        width=0.5,
        alpha=0.3,
        ax=ax,
    )
    
    # 标注高度数节点（hub 蛋白）
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    hub_labels = {node: f"{node[:12]}\n(deg={deg})" for node, deg in top_hubs}
    nx.draw_networkx_labels(
        G_vis, pos,
        labels=hub_labels,
        font_size=7,
        font_color='#333333',
        ax=ax,
    )
    
    ax.set_title(f"{title}\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}",
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_l3_paths_in_context(G: nx.Graph, paths: dict,
                                   save_name: str = "02_l3_paths_in_network.png"):
    """
    可视化 2：在 PPI 网络子图中高亮 L3 路径。
    
    选取若干正/负样本 L3 路径，展示它们在原始网络中的位置。
    """
    print(f"\n[可视化 2] L3 路径在网络中的位置")
    
    # 选取最多 3 条正样本和 3 条负样本
    pos_paths = paths['positive'][:3]
    neg_paths = paths['negative'][:3]
    all_paths = pos_paths + neg_paths
    
    if not all_paths:
        print("  无 L3 路径可可视化")
        return
    
    # 收集路径涉及的所有节点和边
    path_nodes = set()
    path_edges = set()
    path_edge_set_pos = set()
    path_edge_set_neg = set()
    
    for i, (u, w1, w2, v) in enumerate(all_paths):
        nodes_in_path = {u, w1, w2, v}
        path_nodes.update(nodes_in_path)
        
        edges_in_path = {(u, w1), (w1, w2), (w2, v)}
        path_edges.update(edges_in_path)
        
        if i < len(pos_paths):
            path_edge_set_pos.update(edges_in_path)
        else:
            path_edge_set_neg.update(edges_in_path)
    
    # 提取包含路径节点的子图（1-hop 邻域）
    context_nodes = set(path_nodes)
    for node in path_nodes:
        if node in G:
            context_nodes.update(G.neighbors(node))
    
    G_sub = G.subgraph(context_nodes).copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    pos = nx.spring_layout(G_sub, k=1.2, iterations=60, seed=42)
    
    # 分类节点
    l3_node_list = list(path_nodes)
    other_node_list = [n for n in G_sub.nodes() if n not in path_nodes]
    
    # 绘制普通边
    normal_edges = [(u, v) for u, v in G_sub.edges() 
                    if (u, v) not in path_edges and (v, u) not in path_edges]
    nx.draw_networkx_edges(G_sub, pos, edgelist=normal_edges,
                           edge_color='#DDDDDD', width=0.8, alpha=0.5, ax=ax)
    
    # 绘制正样本 L3 路径边（红色虚线）
    pos_edges = [(u, v) for u, v in path_edges if (u, v) in path_edge_set_pos]
    nx.draw_networkx_edges(G_sub, pos, edgelist=pos_edges,
                           edge_color=COLORS['positive'], width=2.5,
                           style='dashed', alpha=0.8, ax=ax)
    
    # 绘制负样本 L3 路径边（蓝色虚线）
    neg_edges = [(u, v) for u, v in path_edges if (u, v) in path_edge_set_neg]
    nx.draw_networkx_edges(G_sub, pos, edgelist=neg_edges,
                           edge_color=COLORS['negative'], width=2.5,
                           style='dashed', alpha=0.8, ax=ax)
    
    # 绘制普通节点
    nx.draw_networkx_nodes(G_sub, pos, nodelist=other_node_list,
                           node_size=150, node_color='#E0E0E0',
                           edgecolors='#AAAAAA', linewidths=0.5, alpha=0.6, ax=ax)
    
    # 绘制 L3 路径节点（大圆圈高亮）
    nx.draw_networkx_nodes(G_sub, pos, nodelist=l3_node_list,
                           node_size=400, node_color=COLORS['node_default'],
                           edgecolors='#333333', linewidths=1.5, alpha=0.95, ax=ax)
    
    # 标注 L3 路径节点角色
    role_labels = {}
    for i, (u, w1, w2, v) in enumerate(all_paths):
        role_labels[u] = f"u{i+1}"
        role_labels[w1] = f"w1_{i+1}"
        role_labels[w2] = f"w2_{i+1}"
        role_labels[v] = f"v{i+1}"
    
    nx.draw_networkx_labels(G_sub, pos, labels=role_labels,
                            font_size=8, font_weight='bold', ax=ax)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['node_default'],
               markersize=12, markeredgecolor='#333', label='L3 path node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E0E0E0',
               markersize=10, markeredgecolor='#AAA', label='Context node'),
        Line2D([0], [0], color=COLORS['positive'], linewidth=2.5, linestyle='--',
               label='Positive L3 path'),
        Line2D([0], [0], color=COLORS['negative'], linewidth=2.5, linestyle='--',
               label='Negative L3 path'),
        Line2D([0], [0], color='#DDDDDD', linewidth=1, label='Network edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    ax.set_title("L3 Paths in PPI Network Context\n(Dashed: L3 path edges | Solid: network edges)",
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_l3_path_structures(paths: dict,
                                  save_name: str = "03_l3_path_structures.png"):
    """
    可视化 3：正/负 L3 路径样本的独立图结构表示。
    
    每条 L3 路径画成独立的小图，标注节点角色和边方向。
    """
    print(f"\n[可视化 3] L3 路径样本结构")
    
    pos_paths = paths['positive'][:3]
    neg_paths = paths['negative'][:3]
    
    n_pos = len(pos_paths)
    n_neg = len(neg_paths)
    n_cols = max(n_pos, n_neg, 1)
    
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 5 * 2))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    # 节点角色颜色映射
    role_colors = {
        0: COLORS['node_u'],    # u: 红色
        1: COLORS['node_w1'],   # w1: 黄色
        2: COLORS['node_w2'],   # w2: 紫色
        3: COLORS['node_v'],    # v: 青色
    }
    role_labels = {0: 'u', 1: 'w₁', 2: 'w₂', 3: 'v'}
    
    # 绘制正样本
    for col, path in enumerate(pos_paths):
        ax = axes[0, col]
        _draw_single_l3_path(ax, path, is_positive=True, title=f"Positive Sample {col+1}")
    
    # 填充空位
    for col in range(n_pos, n_cols):
        axes[0, col].axis('off')
    
    # 绘制负样本
    for col, path in enumerate(neg_paths):
        ax = axes[1, col]
        _draw_single_l3_path(ax, path, is_positive=False, title=f"Negative Sample {col+1}")
    
    # 填充空位
    for col in range(n_neg, n_cols):
        axes[1, col].axis('off')
    
    # 行标签
    axes[0, 0].set_ylabel("Positive\n(u,v 有直接边)", fontsize=12, fontweight='bold',
                           color=COLORS['positive'], rotation=0, labelpad=80, va='center')
    axes[1, 0].set_ylabel("Negative\n(u,v 无直接边)", fontsize=12, fontweight='bold',
                           color=COLORS['negative'], rotation=0, labelpad=80, va='center')
    
    fig.suptitle("L3 Path Sample Structures (u → w₁ → w₂ → v)",
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def _draw_single_l3_path(ax, path, is_positive=True, title=""):
    """绘制单条 L3 路径的小图"""
    u, w1, w2, v = path
    
    # 创建有向图
    G_path = nx.DiGraph()
    G_path.add_nodes_from([0, 1, 2, 3])
    G_path.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # 线性布局
    pos = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    
    # 节点颜色
    node_colors = [COLORS[i] for i in range(4)]
    role_colors = {
        0: COLORS['node_u'],
        1: COLORS['node_w1'],
        2: COLORS['node_w2'],
        3: COLORS['node_v'],
    }
    role_labels = {0: 'u', 1: 'w₁', 2: 'w₂', 3: 'v'}

    # 节点颜色
    node_colors = [role_colors[i] for i in range(4)]
    # 绘制节点
    nx.draw_networkx_nodes(
        G_path, pos,
        node_size=800,
        node_color=node_colors,
        edgecolors='#333333',
        linewidths=2,
        ax=ax,
    )
    
    # 绘制有向边（带箭头）
    for src, dst in G_path.edges():
        arrow = FancyArrowPatch(
            pos[src], pos[dst],
            arrowstyle='->', mutation_scale=20,
            linewidth=2.5,
            color=COLORS['positive'] if is_positive else COLORS['negative'],
            alpha=0.8,
        )
        ax.add_patch(arrow)
    
    # 标注节点角色
    labels = {i: role_labels[i] for i in range(4)}
    nx.draw_networkx_labels(G_path, pos, labels, font_size=14, font_weight='bold', ax=ax)
    
    # 标注蛋白 ID
    id_labels = {0: u[:10], 1: w1[:10], 2: w2[:10], 3: v[:10]}
    pos_below = {k: (v[0], v[1] - 0.25) for k, v in pos.items()}
    nx.draw_networkx_labels(G_path, pos_below, id_labels, font_size=7, font_color='#666', ax=ax)
    
    # 标题和状态
    status = "y = 1" if is_positive else "y = 0"
    status_color = COLORS['positive'] if is_positive else COLORS['negative']
    ax.set_title(f"{title}\n{status}", fontsize=11, fontweight='bold', color=status_color)
    ax.axis('off')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.8, 0.6)


def visualize_prompt_graph(K: int = 4, gates: list = None,
                            save_name: str = "04_prompt_graph_gating.png"):
    """
    可视化 4：初始提示图 Gpre 和门控筛选后的 GF。
    
    左右两列对比：
    - 左：Gpre（所有 K 条路径激活）
    - 右：GF（门控筛选后，部分路径被抑制）
    """
    print(f"\n[可视化 4] 提示图结构与门控筛选")
    
    if gates is None:
        # 默认示例：路径 1 和 3 激活，路径 2 和 4 抑制
        gates = [1, 0, 1, 0]
    if len(gates) < K:
    # 补全到 K 个，缺省用 0
       gates = (gates + [0] * K)[:K]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax, title, show_gating in [(ax1, "Initial Prompt Graph (Gpre)", False),
                                    (ax2, "Final Prompt Graph (GF)", True)]:
        G_prompt = nx.DiGraph()
        
        # 节点: u(0), v(1), v_P0(2), v_P1(3), ..., v_PK(K+2)
        num_nodes = K + 3
        G_prompt.add_nodes_from(range(num_nodes))
        
        # 布局: u 在左，v 在右，虚拟节点在中间
        pos = {}
        pos[0] = (0, 0)       # u
        pos[1] = (4, 0)       # v
        pos[2] = (2, 0)       # v_P0（中心）
        
        for i in range(K):
            angle = np.pi / 2 + (i / (K - 1)) * np.pi if K > 1 else np.pi / 2
            px = 2 + 1.2 * np.cos(angle)
            py = 1.2 * np.sin(angle)
            pos[3 + i] = (px, py)
        
        # 绘制所有边
        for i in range(K):
            pi = 3 + i  # v_Pi 的节点索引
            
            if show_gating and gates[i] == 0:
                # 被抑制的路径：灰色虚线
                edge_color = COLORS['edge_inactive']
                edge_width = 1.0
                edge_style = 'dotted'
                alpha = 0.4
            else:
                # 激活的路径：彩色实线
                edge_color = COLORS['edge_active'] if not show_gating or gates[i] == 1 else COLORS['edge_inactive']
                edge_width = 2.0
                edge_style = 'solid'
                alpha = 0.8
            
            # u → v_Pi
            arrow1 = FancyArrowPatch(
                pos[0], pos[pi],
                arrowstyle='->', mutation_scale=15,
                linewidth=edge_width, color=edge_color,
                linestyle=edge_style, alpha=alpha,
                connectionstyle='arc3,rad=0.1',
            )
            ax.add_patch(arrow1)
            
            # v_Pi → v_P0
            arrow2 = FancyArrowPatch(
                pos[pi], pos[2],
                arrowstyle='->', mutation_scale=15,
                linewidth=edge_width, color=edge_color,
                linestyle=edge_style, alpha=alpha,
            )
            ax.add_patch(arrow2)
            
            # v_Pi → v
            arrow3 = FancyArrowPatch(
                pos[pi], pos[1],
                arrowstyle='->', mutation_scale=15,
                linewidth=edge_width, color=edge_color,
                linestyle=edge_style, alpha=alpha,
                connectionstyle='arc3,rad=-0.1',
            )
            ax.add_patch(arrow3)
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        for n in range(num_nodes):
            if n == 0:
                node_colors.append(COLORS['node_u'])
                node_sizes.append(700)
            elif n == 1:
                node_colors.append(COLORS['node_v'])
                node_sizes.append(700)
            elif n == 2:
                node_colors.append(COLORS['node_vp0'])
                node_sizes.append(600)
            else:
                node_colors.append(COLORS['node_vpi'])
                node_sizes.append(500)
        
        nx.draw_networkx_nodes(
            G_prompt, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors='#333333',
            linewidths=1.5,
            alpha=0.95,
            ax=ax,
        )
        
        # 标注节点
        labels = {0: 'u', 1: 'v', 2: 'v_P0'}
        for i in range(K):
            gate_str = f"\ng={gates[i]}" if show_gating else ""
            labels[3 + i] = f"v_P{i+1}{gate_str}"
        
        nx.draw_networkx_labels(G_prompt, pos, labels, font_size=10, font_weight='bold', ax=ax)
        
        # 统计
        if show_gating:
            active = sum(gates)
            edge_count = active * 3
            ax.set_title(f"{title}\nActive paths: {active}/{K} | Edges: {edge_count}/{3*K}",
                         fontsize=13, fontweight='bold')
        else:
            ax.set_title(f"{title}\nAll {K} paths active | Edges: {3*K}",
                         fontsize=13, fontweight='bold')
        
        ax.axis('off')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1.5, 2.5)
    
    fig.suptitle("GNNgpt Gating Mechanism: Gpre → GF",
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_gnnpre_dataflow(save_name: str = "05_gnnpre_dataflow.png"):
    """
    可视化 5：GNNpre/GIN 处理 GF 的完整数据流。
    
    展示从输入图 → GIN 消息传递 → Sum Pooling → 分类输出的流程。
    """
    print(f"\n[可视化 5] GNNpre/GIN 数据流")
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    
    # ---- 子图 1：输入图 GF ----
    ax = axes[0]
    G_in = nx.DiGraph()
    G_in.add_edges_from([(0, 2), (2, 1), (0, 3), (3, 2), (3, 1)])
    pos_in = {0: (0, 1), 1: (3, 1), 2: (1.5, 1), 3: (1.5, 0)}
    
    nx.draw_networkx_nodes(G_in, pos_in, node_size=500,
                           node_color=[COLORS['node_u'], COLORS['node_v'],
                                       COLORS['node_vp0'], COLORS['node_vpi']],
                           edgecolors='#333', linewidths=1.5, ax=ax)
    for src, dst in G_in.edges():
        arrow = FancyArrowPatch(pos_in[src], pos_in[dst],
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color=COLORS['edge_active'])
        ax.add_patch(arrow)
    nx.draw_networkx_labels(G_in, pos_in, {0: 'u', 1: 'v', 2: 'v_P0', 3: 'v_P1'},
                            font_size=10, font_weight='bold', ax=ax)
    ax.set_title("Input: Final\nPrompt Graph GF", fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # ---- 子图 2：GIN 消息传递 ----
    ax = axes[1]
    # 画 3 层 GIN 的示意
    layer_y = [2.5, 1.5, 0.5]
    layer_labels = ['Layer 1', 'Layer 2', 'Layer 3']
    
    for i, (y, label) in enumerate(zip(layer_y, layer_labels)):
        # 4 个节点
        for j, (x, name) in enumerate([(0, 'u'), (1, 'w₁'), (2, 'w₂'), (3, 'v')]):
            circle = plt.Circle((x, y), 0.2, color=COLORS['node_default'],
                                ec='#333', linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, f'h_{name}^({i+1})', ha='center', va='center',
                    fontsize=7, fontweight='bold', zorder=4)
        
        # 层间连接
        if i > 0:
            for j in range(4):
                ax.annotate('', xy=(j, y + 0.2), xytext=(j, layer_y[i-1] - 0.2),
                            arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
        
        ax.text(-0.8, y, label, fontsize=9, fontweight='bold', va='center', color='#666')
    
    ax.set_title("GIN Message\nPassing (L layers)", fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(-1.2, 4)
    ax.set_ylim(0, 3)
    
    # ---- 子图 3：Sum Pooling ----
    ax = axes[2]
    # 多个节点汇聚到一个
    for j, (x, name) in enumerate([(0, 'u'), (1, 'w₁'), (2, 'w₂'), (3, 'v')]):
        circle = plt.Circle((x, 2), 0.2, color=COLORS['node_default'],
                            ec='#333', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, 2, f'h_{name}', ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=4)
        
        # 箭头指向中心
        ax.annotate('', xy=(1.5, 0.7), xytext=(x, 1.8),
                    arrowprops=dict(arrowstyle='->', color=COLORS['edge_active'], lw=2))
    
    # 汇聚结果
    circle = plt.Circle((1.5, 0.5), 0.3, color=COLORS['node_vp0'],
                        ec='#333', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(1.5, 0.5, 'h_G', ha='center', va='center',
            fontsize=11, fontweight='bold', zorder=4)
    
    ax.text(1.5, -0.2, 'Σ (Sum Pooling)', ha='center', fontsize=10,
            fontweight='bold', color='#666')
    
    ax.set_title("Graph-Level\nReadOut", fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(-0.8, 4)
    ax.set_ylim(-0.5, 2.8)
    
    # ---- 子图 4：分类输出 ----
    ax = axes[3]
    # h_G → Linear → Sigmoid → ỹ_pre
    elements = [
        (0.5, 2.0, 'h_G', COLORS['node_vp0']),
        (0.5, 1.2, 'Linear', '#DDD'),
        (0.5, 0.4, 'σ(z)', '#DDD'),
    ]
    
    for x, y, text, color in elements:
        if text == 'h_G':
            circle = plt.Circle((x, y), 0.25, color=color, ec='#333', linewidth=2, zorder=3)
            ax.add_patch(circle)
        else:
            rect = plt.Rectangle((x - 0.4, y - 0.15), 0.8, 0.3,
                                  facecolor=color, ec='#333', linewidth=1.5, zorder=3)
            ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)
    
    # 箭头
    for y_start, y_end in [(1.75, 1.35), (1.05, 0.55)]:
        ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    # 输出值
    ax.text(0.5, -0.2, 'ỹ_pre = 0.86', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['positive'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=COLORS['positive']))
    
    ax.set_title("Binary\nClassification", fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, 2.6)
    
    # 总标题
    fig.suptitle("GNNpre/GIN Data Flow: Input Graph → GIN Layers → Sum Pooling → PPI Score",
                 fontsize=15, fontweight='bold', y=1.02)
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def visualize_degree_distribution(G: nx.Graph,
                                   save_name: str = "06_network_statistics.png"):
    """
    可视化 6：网络统计特征（度分布、连通分量等）。
    """
    print(f"\n[可视化 6] 网络统计特征")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ---- 度分布 ----
    ax = axes[0, 0]
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=50, color=COLORS['node_default'], edgecolor='#555', alpha=0.8)
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Count')
    ax.set_title('Degree Distribution', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # ---- 连通分量大小分布 ----
    ax = axes[0, 1]
    cc_sizes = [len(c) for c in nx.connected_components(G)]
    cc_sizes.sort(reverse=True)
    ax.bar(range(len(cc_sizes[:50])), cc_sizes[:50],
           color=COLORS['node_vpi'], edgecolor='#555', alpha=0.8)
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Component Size')
    ax.set_title(f'Connected Components (total: {len(cc_sizes)})', fontweight='bold')
    
    # ---- 最短路径长度分布（最大连通子图）----
    ax = axes[1, 0]
    largest_cc = max(nx.connected_components(G), key=len)
    G_cc = G.subgraph(largest_cc)
    
    if G_cc.number_of_nodes() <= 5000:
        # 采样计算
        sample_nodes = random.sample(list(G_cc.nodes()), min(200, G_cc.number_of_nodes()))
        path_lengths = []
        for u in sample_nodes:
            lengths = nx.single_source_shortest_path_length(G_cc, u)
            path_lengths.extend(lengths.values())
        
        ax.hist(path_lengths, bins=30, color=COLORS['node_w1'], edgecolor='#555', alpha=0.8)
        ax.set_xlabel('Shortest Path Length')
        ax.set_ylabel('Count')
        ax.set_title('Shortest Path Length Distribution (sampled)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Graph too large\nfor path analysis',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Shortest Path Length Distribution', fontweight='bold')
    
    # ---- 基本统计信息 ----
    ax = axes[1, 1]
    ax.axis('off')
    
    density = nx.density(G)
    avg_degree = np.mean(degrees)
    avg_clustering = nx.average_clustering(G)
    
    stats_text = (
        f"Network Statistics\n"
        f"{'─' * 30}\n"
        f"Nodes:              {G.number_of_nodes():>10,}\n"
        f"Edges:              {G.number_of_edges():>10,}\n"
        f"Density:            {density:>10.6f}\n"
        f"Avg Degree:         {avg_degree:>10.2f}\n"
        f"Max Degree:         {max(degrees):>10}\n"
        f"Min Degree:         {min(degrees):>10}\n"
        f"Avg Clustering:     {avg_clustering:>10.4f}\n"
        f"Connected Comp.:    {len(cc_sizes):>10}\n"
        f"Largest Comp.:      {cc_sizes[0]:>10}\n"
    )
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))
    ax.set_title('Summary', fontweight='bold')
    
    fig.suptitle("PPI Network Statistical Properties", fontsize=15, fontweight='bold')
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PPI 图数据集构建与可视化")
    parser.add_argument("--data_file", type=str, default="data/raw/HI-II-14.tsv",
                        help="PPI 网络 TSV 文件路径")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "network", "l3path", "prompt", "dataflow", "stats"],
                        help="可视化模式")
    parser.add_argument("--max_nodes", type=int, default=200,
                        help="网络可视化最大节点数")
    parser.add_argument("--max_paths", type=int, default=20,
                        help="每种样本最大提取路径数")
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 60)
    print("PPI 图数据集构建与可视化")
    print("=" * 60)
    
    # ---- 步骤 1：构建图 ----
    print(f"\n[步骤 1] 从 {args.data_file} 构建 PPI 网络图")
    
    if not os.path.exists(args.data_file):
        print(f"错误: 文件不存在 {args.data_file}")
        print("请先下载 HI-II-14 数据，或使用 --data_file 指定路径")
        
        # 生成示例数据进行演示
        print("\n[演示] 使用随机生成的示例 PPI 网络")
        G = nx.barabasi_albert_graph(500, 3, seed=42)
        G = nx.relabel_nodes(G, {n: f"ENSG{n:06d}" for n in G.nodes()})
        print(f"  示例网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    else:
        G = load_ppi_network(args.data_file)
    
    # ---- 步骤 2：提取 L3 路径 ----
    print(f"\n[步骤 2] 提取 L3 路径样本")
    paths = extract_l3_paths(G, max_samples=args.max_paths)
    
    # ---- 步骤 3：可视化 ----
    print(f"\n[步骤 3] 生成可视化图")
    
    if args.mode in ("all", "network"):
        visualize_full_network(G, max_nodes=args.max_nodes)
    
    if args.mode in ("all", "l3path"):
        visualize_l3_paths_in_context(G, paths)
        visualize_l3_path_structures(paths)
    
    if args.mode in ("all", "prompt"):
        visualize_prompt_graph(K=4, gates=[1, 0, 1, 0])
    
    if args.mode in ("all", "dataflow"):
        visualize_gnnpre_dataflow()
    
    if args.mode in ("all", "stats"):
        visualize_degree_distribution(G)
    
    print(f"\n{'=' * 60}")
    print(f"全部完成！输出目录: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")
    
    # 列出生成的文件
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()