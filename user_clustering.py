# -*- coding: utf-8 -*-

# 用户聚类分析模块 - 基于观看兴趣相似性

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
from tqdm import tqdm


def create_user_interest_matrix(users, all_categories):
    """
    创建用户-兴趣矩阵，用于计算用户之间的相似性
    """
    # 创建类别到索引的映射
    category_to_idx = {cat: i for i, cat in enumerate(all_categories)}

    # 初始化矩阵
    matrix = np.zeros((len(users), len(all_categories)))

    # 填充矩阵 - 使用用户的兴趣偏好
    for i, user in enumerate(users):
        for category, weight in user.interests.items():
            if category in category_to_idx:
                j = category_to_idx[category]
                matrix[i, j] = weight

    return matrix, category_to_idx


def analyze_user_video_preferences(users, videos):
    """
    分析用户实际观看行为中的视频类别偏好
    """
    # 创建URL到视频对象的映射
    url_to_video = {v.url: v for v in videos}

    # 统计每个用户观看的视频类别
    user_category_counts = []
    for user in users:
        category_count = Counter()
        for url, _ in user.watched_videos:
            if url in url_to_video and url_to_video[url].category:
                category_count[url_to_video[url].category] += 1

        # 只保留频次最高的类别
        user_category_counts.append(category_count)

    return user_category_counts


def cluster_users(users, videos, n_clusters=None):
    """
    基于相似兴趣对用户进行聚类
    """
    # 获取所有唯一类别
    all_categories = set()
    for video in videos:
        if video.category:
            all_categories.add(video.category)
    all_categories = list(all_categories)

    print("Creating user-interest matrix...")
    matrix, category_to_idx = create_user_interest_matrix(users, all_categories)

    # 检查并处理零向量
    # 找出非零行（至少有一个兴趣类别的用户）
    non_zero_rows = np.where(matrix.sum(axis=1) > 0)[0]

    if len(non_zero_rows) < 2:
        print("Warning: Not enough users with interests for clustering")
        return []

    # 只保留非零行进行聚类
    filtered_matrix = matrix[non_zero_rows]

    # 获取用户实际观看数据
    user_category_counts = analyze_user_video_preferences(users, videos)

    # 如果没有指定聚类数，自动确定最佳聚类数
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(filtered_matrix, max_clusters=min(20, len(filtered_matrix) // 50))

    print(f"Clustering {len(filtered_matrix)} users into {n_clusters} groups...")

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
    # 创建映射表：原始用户索引 -> 聚类结果
    cluster_map = {}
    for i, orig_idx in enumerate(non_zero_rows):
        cluster_map[orig_idx] = clusters[i]

    for cluster_id in range(n_clusters):
        cluster_users = []
        top_categories = Counter()

        for orig_idx, c in cluster_map.items():
            if c == cluster_id:
                user = users[orig_idx]

                # 收集该用户观看过的类别
                for cat, count in user_category_counts[orig_idx].items():
                    top_categories[cat] += count

                # 记录用户及其观看和点赞数据
                cluster_users.append((
                    user.user_id,
                    len(user.watched_videos),
                    len(user.liked_videos)
                ))

        # 按观看视频数量排序
        cluster_users.sort(key=lambda x: x[1], reverse=True)

        result.append({
            'cluster_id': cluster_id,
            'size': len(cluster_users),
            'users': cluster_users,  # 保存所有用户
            'display_users': cluster_users[:10],  # 用于显示的前10个用户
            'top_categories': top_categories.most_common(5)  # 前5个最常观看的类别
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
    ks = list(range(2, max_k + 1, max(1, max_k // 5)))
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


# 添加一个函数用于查看某个聚类的所有用户
def view_cluster_users(clusters, cluster_id):
    """
    查看指定聚类中的所有用户

    参数:
    - clusters: 聚类结果列表
    - cluster_id: 要查看的聚类ID

    返回:
    - 该聚类的所有用户列表
    """
    for cluster in clusters:
        if cluster['cluster_id'] == cluster_id:
            return cluster['users']
    return []


# 修改报告保存函数，包含所有用户信息
def save_user_clusters_report(clusters, filename="user_clusters.xlsx"):
    """
    保存用户聚类结果到Excel
    """
    wb = pd.ExcelWriter(filename)

    # 创建摘要表
    summary_data = []
    for c in clusters:
        top_cats = ", ".join([f"{cat} ({count})" for cat, count in c['top_categories'][:3]]) if c[
            'top_categories'] else "None"
        summary_data.append({
            'Cluster ID': c['cluster_id'],
            'Size': c['size'],
            'Top Categories': top_cats,
            'Top Users': ', '.join([u[0] for u in c['display_users'][:3]]) if c['display_users'] else "None"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(wb, sheet_name='Clusters Summary', index=False)

    # 为每个聚类创建详细表
    all_users = []
    for c in clusters:
        for u in c['users']:  # 使用所有用户而非前10个
            all_users.append({
                'Cluster ID': c['cluster_id'],
                'User ID': u[0],
                'Videos Watched': u[1],
                'Videos Liked': u[2]
            })

    details_df = pd.DataFrame(all_users)
    details_df.to_excel(wb, sheet_name='Cluster Details', index=False)

    # 兴趣分布表
    interests_data = []
    for c in clusters:
        for cat, count in c['top_categories']:
            interests_data.append({
                'Cluster ID': c['cluster_id'],
                'Category': cat,
                'Watch Count': count
            })

    interests_df = pd.DataFrame(interests_data)
    interests_df.to_excel(wb, sheet_name='Interest Distribution', index=False)

    wb.close()
    print(f"User clusters report saved to {filename}")
    print(f"完整聚类信息已保存到文件，可通过Excel查看每个聚类的所有用户")

def plot_user_clusters(clusters, filename="user_clusters.png"):
    """
    绘制用户聚类大小分布图
    """
    if not clusters:
        print("Warning: No clusters to plot")
        return

    sizes = [c['size'] for c in clusters]
    ids = [f"Cluster {c['cluster_id']}" for c in clusters]

    plt.figure(figsize=(12, 6))
    plt.bar(ids, sizes)
    plt.title('User Clusters by Size')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"User clusters chart saved to {filename}")
