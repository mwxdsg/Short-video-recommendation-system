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
CACHE_DIR = "./cache"

# ==================== 数据缓存 ====================
memory = Memory(CACHE_DIR, verbose=0)

@memory.cache
def load_data(file_path):
    # 只读取必要列并指定数据类型
    dtype_spec = {
        '用户ID': 'category',
        '视频URL': 'string', 
        '类别': 'category',
        '观看次数': 'uint32', 
        '点赞数': 'uint16'
    }
    
    with pd.ExcelFile(file_path) as xls:
        # 1. 处理Watch History表（特殊格式）
        watch_raw = pd.read_excel(
            xls, 
            sheet_name="观看记录",
            dtype={'用户ID': 'category'}
        )
        
        # 转换7天分列格式为规范化的长格式
        watch_dfs = []
        date_columns = [col for col in watch_raw.columns if col not in ['用户ID']]
        
        for date_col in date_columns:
            temp_df = watch_raw[['用户ID', date_col]].copy()
            temp_df = temp_df.rename(columns={date_col: '视频URL'})
            temp_df['观看日期'] = date_col  # 保留原日期信息
            
            # 拆分逗号分隔的URL
            temp_df = (
                temp_df.dropna(subset=['视频URL'])
                .assign(视频URL=lambda x: x['视频URL'].str.split(',\s*'))
                .explode('视频URL')
            )
            watch_dfs.append(temp_df)
        
        watch_df = pd.concat(watch_dfs, ignore_index=True)
        
        # 2. 处理Video Statistics表（标准格式）
        videos_df = pd.read_excel(
            xls, 
            sheet_name="视频统计",
            usecols=["视频URL", "类别", "观看次数", "点赞数"],
            dtype=dtype_spec
        )
        
        
        return watch_df, videos_df

# ==================== 核心优化 ====================
class Recommender:
    def __init__(self, watch_df, videos_df):
        # 建立索引
        self.watch = watch_df.set_index("用户ID")
        self.videos = videos_df.set_index("视频URL")
        
        # 预计算全局数据
        self.category_stats = videos_df["类别"].value_counts(normalize=True)
        self.max_views = videos_df["观看次数"].max()
        self.max_likes = videos_df["点赞数"].max()
        
        # 用户相似度矩阵
        self.user_sim_matrix = self._precompute_user_similarity()
    
    def _precompute_user_similarity(self):
        """ 预计算用户相似度 """
        user_cat_matrix = (
            self.watch.join(self.videos, on="视频URL")
            .groupby(["用户ID", "类别"])
            .size()
            .unstack(fill_value=0)
        )
        return cosine_similarity(user_cat_matrix)

    def get_user_recommendations(self, user_id, top_n=10, diversify=True):
        """ 综合推荐引擎 """
        # 获取用户历史
        try:
            user_watched = self.watch.loc[user_id]["视频URL"].tolist()
            user_categories = self.videos.loc[user_watched]["类别"]
            cat_counts = user_categories.value_counts()
        except KeyError:
            return self._get_fallback_recommendations(top_n)
        
        # 候选视频
        candidates = self.videos[~self.videos.index.isin(user_watched)].copy()
        
        # 向量化评分
        candidates["base_score"] = (
            0.4 * candidates["类别"].map(cat_counts).fillna(0) +
            0.4 * candidates["观看次数"] / self.max_views +
            0.2 * candidates["点赞数"] / self.max_likes
        )
        
        # 多样性增强
        if diversify:
            candidates = self._apply_diversity(candidates, user_categories)
        
        return candidates.nlargest(top_n, "final_score")

    def _apply_diversity(self, candidates, user_categories):
        """ 多样性增强策略 """
        # 1. 类别覆盖率奖励
        candidates["cat_coverage"] = ~candidates["类别"].isin(user_categories)
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
    watch_df, videos_df = load_data("user_behavior_7days.xlsx")
    
    # 2. 初始化推荐引擎
    engine = Recommender(watch_df, videos_df)
    
    # 3. 为用户生成推荐
    user_id = "U0001"
    print(f"\n为用户 {user_id} 生成推荐:")
    recs = engine.get_user_recommendations(user_id, top_n=5)
    print(recs[["类别", "观看次数", "final_score"]])
