# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import warnings

warnings.filterwarnings("ignore")
import json

# ==================== 数据加载 ====================
memory = Memory("./cache", verbose=0)


@memory.cache
def load_data(file_path):
    # 只读取必要列并指定数据类型
    dtype_spec = {
        "用户ID": "category",
        "视频URL": "string",
        "类别": "category",
        "观看次数": "uint32",
        "点赞数": "uint16",
        "封面地址": "string",
    }

    with pd.ExcelFile(file_path) as xls:
        # 1. 处理Watch History表（特殊格式）
        watch_raw = pd.read_excel(
            xls, sheet_name="观看记录", dtype={"用户ID": "category"}
        )

        # 转换7天分列格式为规范化的长格式
        watch_dfs = []
        date_columns = [col for col in watch_raw.columns if col not in ["用户ID"]]

        for date_col in date_columns:
            temp_df = watch_raw[["用户ID", date_col]].copy()
            temp_df = temp_df.rename(columns={date_col: "视频URL"})
            temp_df["观看日期"] = date_col  # 保留原日期信息

            # 拆分逗号分隔的URL
            temp_df = (
                temp_df.dropna(subset=["视频URL"])
                .assign(视频URL=lambda x: x["视频URL"].str.split(",\s*"))
                .explode("视频URL")
            )
            watch_dfs.append(temp_df)

        watch_df = pd.concat(watch_dfs, ignore_index=True)

        # 2. 处理Video Statistics表（标准格式）
        videos_df = pd.read_excel(
            xls,
            sheet_name="视频统计",
            usecols=["视频URL", "类别", "观看次数", "点赞数", "封面地址"],
            dtype=dtype_spec,
        )

        # 3.处理merge
    merge_df = pd.read_csv(
        "merge.csv",
        usecols=["video_url", "title", "id"],
        dtype={"video_url": "string", "title": "string", "id": "int"},
        encoding="utf-8",
    )
    # 4.合并数据
    videos_df = videos_df.merge(
        merge_df, left_on="视频URL", right_on="video_url", how="left"
    )
    videos_df["title"] = videos_df["title"].fillna("未知标题")

    return watch_df, videos_df


def recommendations_to_json(recommendations):
    """将推荐结果转为格式化的JSON字符串"""
    if isinstance(recommendations, pd.DataFrame):
        # 重置索引并选择需要输出的列
        output_cols = ["title", "id", "封面地址"]
        available_cols = [col for col in output_cols if col in recommendations.columns]

        # 转换为字典列表
        recs_dict = recommendations.reset_index()[available_cols].to_dict(
            orient="records"
        )

        # 转换为JSON并美化输出
        return json.dumps(
            {"recommendations": recs_dict, "count": len(recommendations)},
            indent=2,
            ensure_ascii=False,
        )
    else:
        return json.dumps({"error": "推荐结果不是DataFrame格式"}, ensure_ascii=False)


# ==================== 推荐引擎 ====================
class Recommender:
    def __init__(self, watch_df, videos_df):
        self.watch = watch_df.set_index("用户ID")
        self.videos = videos_df.set_index("视频URL")
        self.max_views = videos_df["观看次数"].max()
        self.max_likes = videos_df["点赞数"].max()

    def get_user_recommendations(self, user_id, top_n=10, diversify=True):
        try:
            user_watched = self.watch.loc[user_id]["视频URL"].tolist()
            user_categories = self.videos.loc[user_watched]["类别"]
            cat_counts = user_categories.value_counts()
        except KeyError:
            return self._get_fallback_recommendations(top_n)

        # 候选筛选（新增质量门槛）
        candidates = self.videos[
            ~self.videos.index.isin(user_watched)
            & (self.videos["观看次数"] >= 10)
            & (self.videos["点赞数"] / self.videos["观看次数"].clip(lower=1) >= 0.02)
        ].copy()

        # 向量化评分
        candidates["base_score"] = (
            0.4 * candidates["类别"].map(cat_counts).fillna(0)
            + 0.4 * candidates["观看次数"] / self.max_views
            + 0.2 * candidates["点赞数"] / self.max_likes
        )

        # 多样性增强
        if diversify:
            candidates = self._apply_diversity(candidates, user_categories)

        return candidates.nlargest(top_n, "final_score")

    def _apply_diversity(self, candidates, user_categories):
        """多样性增强策略"""
        # 1. 类别覆盖率奖励
        candidates["cat_coverage"] = ~candidates["类别"].isin(user_categories)
        # 2. 长尾内容奖励
        candidates["longtail"] = candidates["观看次数"] < self.max_views * 0.1
        # 3. 综合得分
        candidates["final_score"] = (
            0.7 * candidates["base_score"]
            + 0.2 * candidates["cat_coverage"]
            + 0.1 * candidates["longtail"]
        )
        return candidates

    def _get_fallback_recommendations(self, top_n):
        """新用户冷启动策略"""
        return self.videos.nlargest(top_n, "观看次数")


# ==================== 使用示例 ====================
def main():

    try:
        watch_df, videos_df = load_data("user_behavior_7days.xlsx")
        engine = Recommender(watch_df, videos_df)

        user_id = "U0002"
        recs = engine.get_user_recommendations(user_id, top_n=100)
        recs_json = recommendations_to_json(recs)
        print(recs_json)
        return recs_json
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
