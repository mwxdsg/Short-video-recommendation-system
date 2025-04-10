# -*- coding: utf-8 -*-
""" 优化版视频用户行为模拟系统 """
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from joblib import Memory
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
OUTPUT_FILE = "optimized_behavior.xlsx"
CACHE_DIR = "./cache"

# ==================== 数据缓存 ====================
memory = Memory(CACHE_DIR, verbose=0)

@memory.cache
def load_data(file_path):
    """ 优化版数据加载 """
    # 只读取必要列并指定数据类型
    dtype_spec = {
        '用户ID': 'category', '视频URL': 'string', 
        '所属类别': 'category', '观看次数': 'uint32', 
        '点赞次数': 'uint16'
    }
    
    with pd.ExcelFile(file_path) as xls:
        watch = pd.read_excel(
            xls, sheet_name="Watch History",
            usecols=["用户ID", "视频URL", "观看时间"],
            dtype=dtype_spec
        )
        videos = pd.read_excel(
            xls, sheet_name="Video Statistics",
            usecols=["视频URL", "所属类别", "观看次数", "点赞次数"],
            dtype=dtype_spec
        )
    return watch, videos

# ==================== 核心优化 ====================
class Recommender:
    def __init__(self, watch_df, videos_df):
        # 建立索引
        self.watch = watch_df.set_index("用户ID")
        self.videos = videos_df.set_index("视频URL")
        
        # 预计算全局数据
        self.category_stats = videos_df["所属类别"].value_counts(normalize=True)
        self.max_views = videos_df["观看次数"].max()
        self.max_likes = videos_df["点赞次数"].max()
        
        # 用户相似度矩阵
        self.user_sim_matrix = self._precompute_user_similarity()
    
    def _precompute_user_similarity(self):
        """ 预计算用户相似度 """
        user_cat_matrix = (
            self.watch.join(self.videos, on="视频URL")
            .groupby(["用户ID", "所属类别"])
            .size()
            .unstack(fill_value=0)
        )
        return cosine_similarity(user_cat_matrix)

    def get_user_recommendations(self, user_id, top_n=10, diversify=True):
        """ 综合推荐引擎 """
        # 获取用户历史
        try:
            user_watched = self.watch.loc[user_id]["视频URL"].tolist()
            user_categories = self.videos.loc[user_watched]["所属类别"]
            cat_counts = user_categories.value_counts()
        except KeyError:
            return self._get_fallback_recommendations(top_n)
        
        # 候选视频
        candidates = self.videos[~self.videos.index.isin(user_watched)].copy()
        
        # 向量化评分
        candidates["base_score"] = (
            0.4 * candidates["所属类别"].map(cat_counts).fillna(0) +
            0.4 * candidates["观看次数"] / self.max_views +
            0.2 * candidates["点赞次数"] / self.max_likes
        )
        
        # 多样性增强
        if diversify:
            candidates = self._apply_diversity(candidates, user_categories)
        
        return candidates.nlargest(top_n, "final_score")

    def _apply_diversity(self, candidates, user_categories):
        """ 多样性增强策略 """
        # 1. 类别覆盖率奖励
        candidates["cat_coverage"] = ~candidates["所属类别"].isin(user_categories)
        # 2. 长尾内容奖励
        candidates["longtail"] = candidates["观看次数"] < self.max_views * 0.1
        # 3. 综合得分
        candidates["final_score"] = (
            0.7 * candidates["base_score"] +
            0.2 * candidates["cat_coverage"] +
            0.1 * candidates["longtail"]
        )
        return candidates

    def _get_fallback_recommendations(self, top_n):
        """ 新用户冷启动策略 """
        return self.videos.nlargest(top_n, "观看次数")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    watch_df, videos_df = load_data("user_behavior_details_tfidf.xlsx")
    
    # 2. 初始化推荐引擎
    engine = Recommender(watch_df, videos_df)
    
    # 3. 为用户生成推荐
    user_id = "U0010"
    print(f"\n为用户 {user_id} 生成推荐:")
    recs = engine.get_user_recommendations(user_id, top_n=5)
    print(recs[["所属类别", "观看次数", "final_score"]])
    
