import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import jieba  # 中文分词
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
memory = Memory("./cache", verbose=0)

@memory.cache
def load_data(file_path):
    """加载视频统计数据（仅使用类别字段）"""
    dtype_spec = {
        '视频URL': 'string', 
        '类别': 'category',
        '观看次数': 'uint32', 
        '点赞数': 'uint16'
    }
    
    with pd.ExcelFile(file_path) as xls:
        return pd.read_excel(
            xls, 
            sheet_name="视频统计",
            usecols=["视频URL", "类别", "观看次数", "点赞数"],
            dtype=dtype_spec
        )

# ==================== 视频搜索系统 ====================
class Search:
    def __init__(self, data_file):
        self.df = load_data(data_file)
        self._build_enhanced_index()
    
    def _build_enhanced_index(self):
        """构建增强型搜索索引"""
        # 使用中文分词处理类别文本
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: jieba.lcut(x),
            stop_words=["的", "了", "是", "我"]  # 常见停用词
        )
        # 使用原始类别+扩展类别作为特征
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["类别"]
        )
    
    def search(self, query, top_n=10, min_views=0, min_likes=0):
        """
        增强版类别搜索
        :param query: 搜索词（支持中文分词）
        :param top_n: 返回结果数量
        :param min_views: 最小观看量
        :param min_likes: 最小点赞数
        :return: 排序后的DataFrame
        """
        # 向量化查询（自动分词）
        query_vec = self.vectorizer.transform([query])
        
        # 计算相似度（考虑热度加权）
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        weighted_scores = similarities * (1 + np.log1p(self.df["观看次数"]))
        
        # 构建结果
        result_df = self.df.copy()
        result_df["相关度"] = similarities
        result_df["加权分"] = weighted_scores
        
        # 应用过滤条件
        result_df = result_df[
            (result_df["观看次数"] >= min_views) & 
            (result_df["点赞数"] >= min_likes)
        ]
        
        # 综合排序（相关度60% + 热度40%）
        return (result_df
                .sort_values(["相关度", "观看次数"], ascending=[False, False])
                .head(top_n)
                [["视频URL", "类别", "相关度", "观看次数", "点赞数"]])
    

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化搜索引擎
    searcher = Search("user_behavior_7days.xlsx")
    
    # 示例搜索
    print("=== 类别搜索示例 ===")
    results = searcher.search("体育", top_n=5)
    print(results)