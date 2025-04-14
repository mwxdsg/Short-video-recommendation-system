# -*- coding: utf-8 -*-

# 视频聚类分析模块 - 基于观看用户相似性

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
from tqdm import tqdm


def create_video_viewer_matrix(videos, users):
    """
    创建视频-观众矩阵，用于计算视频之间的相似性
    """
    # 创建视频ID到索引的映射
    video_to_idx = {video.url: i for i, video in enumerate(videos)}

    # 初始化矩阵
    matrix = np.zeros((len(videos), len(users)))

    # 填充矩阵
    for j, user in enumerate(users):
        for url, _ in user.watched_videos:
            if url in video_to_idx:
                i = video_to_idx[url]
                matrix[i, j] = 1

    return matrix, video_to_idx


def cluster_videos(videos, users, n_clusters=None):
    """
    基于相似观众对视频进行聚类
    """
    print("Creating video-viewer matrix...")
    matrix, video_to_idx = create_video_viewer_matrix(videos, users)

    # 检查并处理零向量
    # 找出非零行（至少有一个用户观看过的视频）
    non_zero_rows = np.where(matrix.sum(axis=1) > 0)[0]

    if len(non_zero_rows) < 2:
        print("Warning: Not enough videos with viewers for clustering")
        return []

    # 只保留非零行进行聚类
    filtered_matrix = matrix[non_zero_rows]

    # 如果没有指定聚类数，自动确定最佳聚类数
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(filtered_matrix, max_clusters=min(20, len(filtered_matrix) // 5))

    print(f"Clustering {len(filtered_matrix)} videos into {n_clusters} groups...")

    # 使用层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )

    # 对矩阵进行聚类
    clusters = clustering.fit_predict(filtered_matrix)

    # 整理结果
    result = []
    idx_to_video = {i: url for url, i in video_to_idx.items()}

    # 创建映射表：原始索引 -> 聚类结果
    cluster_map = {}
    for i, orig_idx in enumerate(non_zero_rows):
        cluster_map[orig_idx] = clusters[i]

    for cluster_id in range(n_clusters):
        cluster_videos = []
        for orig_idx, c in cluster_map.items():
            if c == cluster_id:
                url = idx_to_video[orig_idx]
                video_obj = next((v for v in videos if v.url == url), None)
                if video_obj:
                    cluster_videos.append((url, len(video_obj.viewers), video_obj.like_count))

        # 按观看人数排序
        cluster_videos.sort(key=lambda x: x[1], reverse=True)
        result.append({
            'cluster_id': cluster_id,
            'size': len(cluster_videos),
            'videos': cluster_videos,  # 保存所有视频，不再截取前10个
            'display_videos': cluster_videos[:10]  # 用于显示的前10个视频
        })

    # 按簇大小排序
    result.sort(key=lambda x: x['size'], reverse=True)
    return result


def determine_optimal_clusters(matrix, max_clusters=20):
    """
    使用轮廓系数确定最佳聚类数
    """
    if len(matrix) < 3:  # 数据太少，无法进行聚类
        return 1

    max_k = min(max_clusters, len(matrix) - 1)

    if max_k <= 2:
        return max_k

    scores = []

    # 为了提高效率，尝试2到max_k之间的几个值
    ks = list(range(2, max_k + 1, max(1, max_k // 10)))
    if max_k not in ks:
        ks.append(max_k)

    for k in tqdm(ks, desc="Finding optimal clusters"):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='cosine',
            linkage='average'
        )

        try:
            cluster_labels = clustering.fit_predict(matrix)
            # 计算轮廓系数
            silhouette_avg = silhouette_score(matrix, cluster_labels, metric='cosine')
            scores.append((k, silhouette_avg))
        except Exception as e:
            # 如果计算失败，分配一个低分
            print(f"Warning: Silhouette calculation failed for k={k}: {str(e)}")
            scores.append((k, -1))

    # 选择轮廓系数最高的k值
    if not scores:
        return 2  # 默认值

    # 从有效分数中选择最佳k
    valid_scores = [(k, s) for k, s in scores if s > -1]
    if valid_scores:
        best_k, _ = max(valid_scores, key=lambda x: x[1])
    else:
        # 如果所有分数都无效，使用默认值
        best_k = 2

    return best_k


# 添加一个函数用于查看某个聚类的所有视频
def view_cluster_videos(clusters, cluster_id):
    """
    查看指定聚类中的所有视频

    参数:
    - clusters: 聚类结果列表
    - cluster_id: 要查看的聚类ID

    返回:
    - 该聚类的所有视频列表
    """
    for cluster in clusters:
        if cluster['cluster_id'] == cluster_id:
            return cluster['videos']
    return []


# 修改报告保存函数，包含所有视频信息
def save_video_clusters_report(clusters, filename="video_clusters.xlsx"):
    """
    保存视频聚类结果到Excel
    """
    wb = pd.ExcelWriter(filename)

    # 创建摘要表
    summary_data = []
    for c in clusters:
        summary_data.append({
            'Cluster ID': c['cluster_id'],
            'Size': c['size'],
            'Top Videos': ', '.join([v[0][:50] + '...' if len(v[0]) > 50 else v[0] for v in c['display_videos'][:3]])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(wb, sheet_name='Clusters Summary', index=False)

    # 为每个聚类创建详细表
    all_videos = []
    for c in clusters:
        for v in c['videos']:  # 使用所有视频而非前10个
            all_videos.append({
                'Cluster ID': c['cluster_id'],
                'Video URL': v[0],
                'View Count': v[1],
                'Like Count': v[2]
            })

    details_df = pd.DataFrame(all_videos)
    details_df.to_excel(wb, sheet_name='Cluster Details', index=False)

    wb.close()
    print(f"Video clusters report saved to {filename}")
    print(f"完整聚类信息已保存到文件，可通过Excel查看每个聚类的所有视频")
def plot_video_clusters(clusters, filename="video_clusters.png"):
    """
    绘制视频聚类大小分布图
    """
    if not clusters:
        print("Warning: No clusters to plot")
        return

    sizes = [c['size'] for c in clusters]
    ids = [f"Cluster {c['cluster_id']}" for c in clusters]

    plt.figure(figsize=(12, 6))
    plt.bar(ids, sizes)
    plt.title('Video Clusters by Size')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Video clusters chart saved to {filename}")
