import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm
import gc


def cluster_videos_by_viewers(videos, users, n_clusters=50, batch_size=5000):
    """
    基于观看用户相似性对视频进行聚类
    参数:
        videos: Video对象列表
        users: User对象列表
        n_clusters: 目标聚类数量
        batch_size: 批处理大小(内存优化)
    返回:
        聚类结果DataFrame
    """
    print("\n开始基于观看用户相似性的视频聚类...")

    # 构建用户-视频矩阵
    user_ids = [u.user_id for u in users]
    video_urls = [v.url for v in videos if v.url != "N/A"]

    # 创建稀疏矩阵 (用户x视频)
    print("构建用户-视频矩阵...")
    user_video_matrix = np.zeros((len(user_ids), len(video_urls)), dtype=np.float32)
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    video_index = {url: i for i, url in enumerate(video_urls)}

    # 填充矩阵 (批处理优化)
    for i, user in enumerate(tqdm(users, desc="处理用户观看记录")):
        for url, _ in user.watched_videos:
            if url in video_index:
                user_video_matrix[user_index[user.user_id], video_index[url]] = 1.0

    # 转置为视频x用户矩阵
    video_user_matrix = user_video_matrix.T
    del user_video_matrix
    gc.collect()

    # 使用MiniBatchKMeans进行聚类 (内存友好)
    print("执行聚类分析...")
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(video_user_matrix)

    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(video_urls)),
        batch_size=batch_size,
        random_state=42,
        n_init=3
    )
    clusters = kmeans.fit_predict(X_scaled)

    # 构建结果DataFrame
    results = []
    for i, url in enumerate(video_urls):
        results.append({
            "视频URL": url,
            "聚类ID": int(clusters[i]),
            "聚类大小": np.sum(clusters == clusters[i]),
            "代表性用户数": np.sum(X_scaled[i] > 0)
        })

    # 添加聚类中心信息
    centers = kmeans.cluster_centers_
    for cluster_id in range(kmeans.n_clusters):
        top_users_idx = np.argsort(centers[cluster_id])[-5:][::-1]
        top_users = [(user_ids[i], centers[cluster_id][i]) for i in top_users_idx]

        for result in results:
            if result["聚类ID"] == cluster_id:
                result["典型用户"] = "; ".join([f"{uid}({score:.2f})" for uid, score in top_users])

    print(f"视频聚类完成，共生成{kmeans.n_clusters}个聚类.")
    return pd.DataFrame(results)
