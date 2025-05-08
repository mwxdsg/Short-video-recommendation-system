import csv

from sympy import im

VIDEO_DB = []


def load_video_database(filepath = "merge.csv"):
    """
    Load video database from a CSV file.
    The CSV file should contain the following columns:
    - id: int
    - title: str
    - liked_count: int
    - viewd_count: int
    - video_url: str
    - video_cover_url: str
    """

    global VIDEO_DB
    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        VIDEO_DB = [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "keywords": row["source_keyword"],
                "liked_count": int(float(row["liked_count"])),
                "viewd_count": int(float(row["viewd_count"])),
                "video_url": row["video_play_url"],
                "video_cover_url": row["video_cover_url"],
                "video_urll": row["video_url"],
            }
            for row in reader
        ]


load_video_database("merge.csv")


import pandas as pd


class Video:
    __slots__ = [
        "id",
        "url",
        "keywords",
        "category",
        "title",
        "like_count",
        "cover_url",
        "play_url",
        "viewers",
        "liked_users",
    ]

    def __init__(
        self, video_id, url, keywords, title=None, cover_url=None, play_url=None
    ):
        self.id = str(video_id).strip() if video_id is not None else "N/A_ID"
        self.url = url.strip() if url else None
        keywords_str = str(keywords) if keywords is not None else ""
        self.keywords = tuple(keywords_str.split()[:15])
        self.category = None
        self.title = str(title).strip() if title else None
        self.like_count = 0
        self.cover_url = (
            cover_url.strip()[:200] if cover_url and pd.notna(cover_url) else None
        )
        self.play_url = (
            play_url.strip()[:200] if play_url and pd.notna(play_url) else None
        )
        self.viewers = set()
        self.liked_users = set()


def load_videos_from_csv(file_path: str):
    df = pd.read_csv(file_path, encoding="utf-8")

    videos = []
    for _, row in df.iterrows():
        video = Video(
            video_id=row.get("视频ID"),
            url=row.get("视频URL"),
            keywords=row.get("关键词"),
            title=row.get("标题"),
            cover_url=row.get("封面URL"),
            play_url=row.get("播放URL"),
        )
        # 扩展可选字段
        video.category = row.get("类别")
        video.like_count = (
            int(row.get("点赞数", 0)) if pd.notna(row.get("点赞数")) else 0
        )

        # 将观看者和点赞用户以逗号分割存入 set
        watchers = row.get("观看者数")
        video.viewers = set(str(watchers).split(",")) if pd.notna(watchers) else set()

        liked_users = row.get("点赞用户数")
        video.liked_users = (
            set(str(liked_users).split(",")) if pd.notna(liked_users) else set()
        )

        videos.append(video)
    return videos

videos = load_videos_from_csv("videos_output.csv")


import pandas as pd
from datetime import datetime
from typing import Dict
class User:
    def __init__(self, user_id, all_categories=None):
        self.user_id = user_id  # U0001
        self.interests = []
        self.watched_videos = []  # [(url, timestamp_str)]
        self.liked_videos = []  # [(url, timestamp_str)]


def load_all_user_data(
    interest_csv: str, behavior_xlsx: str, watch_sheet="观看记录", like_sheet="点赞记录"
) -> Dict[str, User]:

    # Step 1: 加载兴趣数据
    interest_df = pd.read_csv(interest_csv, encoding="utf-8")
    user_dict: Dict[str, User] = {}

    for _, row in interest_df.iterrows():
        numeric_id = int(row["用户ID"])
        uid = f"U{numeric_id:04d}"
        user = User(uid)
        interest_str = str(row["兴趣类别"])
        user.interests = [cat.strip() for cat in interest_str.split(",")]
        user_dict[uid] = user

    # Step 2: 加载观看记录
    watch_df = pd.read_excel(behavior_xlsx, sheet_name=watch_sheet)
    for _, row in watch_df.iterrows():
        uid = row["用户ID"]
        if uid not in user_dict:
            user_dict[uid] = User(uid)

        for col in watch_df.columns[1:]:  # 跳过“用户ID”
            try:
                date_obj = datetime.strptime(col, "%m-%d").replace(year=2025)
            except:
                continue

            urls = str(row[col]).split(",") if pd.notna(row[col]) else []
            for url in urls:
                url = url.strip()
                if url:
                    timestamp = date_obj.strftime("%Y-%m-%d 00:00:00")
                    user_dict[uid].watched_videos.append((url, timestamp))

    # Step 3: 加载点赞记录
    like_df = pd.read_excel(behavior_xlsx, sheet_name=like_sheet)
    for _, row in like_df.iterrows():
        uid = row["用户ID"]
        url = str(row["视频URL"]).strip()
        time_str = str(row["点赞时间"]).strip()

        if not url or not time_str:
            continue

        if uid not in user_dict:
            user_dict[uid] = User(uid)

        user_dict[uid].liked_videos.append((url, time_str))

    return user_dict

users = load_all_user_data(
    interest_csv="user.csv",
    behavior_xlsx="user_behavior_7days.xlsx"
)
users_list = list(users.values())


from user_clusters_read import load_data_from_excel_sheet_with_dtypes
# 读取用户聚类数据
USER_CLUSTER_DF = load_data_from_excel_sheet_with_dtypes(
        excel_filepath="user_clusters.xlsx",
        sheet_name="用户聚类",
        expected_dtypes={
            "用户ID": "object",
            "聚类ID": "int64",
            "聚类大小": "int64",
            "观看类别数": "int64",
            "典型类别": "object",
        },
    )
# 读取视频聚类数据

from video_clusters_read import load_clusters_from_excel_with_dtypes

VIDEO_CLUSTER_DF = load_clusters_from_excel_with_dtypes(
    excel_filepath="video_clusters_optimized_balanced.xlsx",
    sheet_name="视频聚类详情",
    expected_dtypes={
        "视频URL": "object",
        "聚类ID": "int64",
        "观看该视频的用户数": "int64",
        "聚类内视频数": "int64",
    },
)
