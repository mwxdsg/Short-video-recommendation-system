import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
import gc


def cluster_users_by_interests(users, videos, n_clusters=100, batch_size=5000):
    """
    基于观看兴趣对用户进行聚类
    参数:
        users: User对象列表
        videos: Video对象列表
        n_clusters: 目标聚类数量
        batch_size: 批处理大小(内存优化)
    返回:
        聚类结果DataFrame
    """
    print("\n开始基于观看兴趣的用户聚类...")

    # 构建视频类别特征
    valid_categories = set(v.category for v in videos if v.category)
    user_categories = defaultdict(list)

    # 收集用户观看的类别
    for user in tqdm(users, desc="收集用户观看类别"):
        watched_urls = {url for url, _ in user.watched_videos}
        for video in videos:
            if video.url in watched_urls and video.category in valid_categories:
                user_categories[user.user_id].append(video.category)

    # 创建TF-IDF特征矩阵
    print("构建TF-IDF特征矩阵...")
    user_ids = list(user_categories.keys())
    corpus = [" ".join(user_categories[uid]) for uid in user_ids]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.8
    )
    X = vectorizer.fit_transform(corpus)

    # 使用BIRCH进行聚类 (适合大数据)
    print("执行聚类分析...")
    birch = Birch(
        n_clusters=min(n_clusters, len(user_ids)),
        threshold=0.5,
        branching_factor=50
    )
    clusters = birch.fit_predict(X)

    # 构建结果DataFrame
    results = []
    for i, uid in enumerate(user_ids):
        results.append({
            "用户ID": uid,
            "聚类ID": int(clusters[i]),
            "聚类大小": np.sum(clusters == clusters[i]),
            "观看类别数": len(set(user_categories[uid])),
            "典型类别": _get_top_categories(user_categories[uid])
        })

    print(f"用户聚类完成，共生成{birch.n_clusters}个聚类.")
    return pd.DataFrame(results)


def _get_top_categories(categories, top_n=5):
    """获取最常见的类别"""
    from collections import Counter
    counter = Counter(categories)
    return "; ".join([f"{cat}({count})" for cat, count in counter.most_common(top_n)])
