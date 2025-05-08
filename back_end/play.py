from flask import Blueprint, jsonify
from data_load import VIDEO_DB, VIDEO_CLUSTER_DF, videos
import random


video_bp = Blueprint("video", __name__)


@video_bp.route("/video/<int:video_id>", methods=["GET"])
def get_video_by_id(video_id):
    # 二分查找
    left, right = 0, len(VIDEO_DB) - 1
    while left <= right:
        mid = (left + right) // 2
        if VIDEO_DB[mid]["id"] == video_id:
            return jsonify(VIDEO_DB[mid])  # 找到视频，返回数据
        elif VIDEO_DB[mid]["id"] < video_id:
            left = mid + 1
        else:
            right = mid - 1

    # 如果未找到，返回错误信息
    return jsonify({"error": "Video not found"}), 404


@video_bp.route("/video/<path:url>", methods=["GET"])
def find_id(url):
    for video in VIDEO_DB:
        if video["video_urll"] == url:
            return jsonify({"id": video["id"]})  # 返回 JSON 格式的响应
    return jsonify({"id": -1})  # 返回 JSON 格式的响应


@video_bp.route("/video/<int:video_id>/recommend", methods=["GET"])
def recommend_for_video(video_id):
    # 为兼容 get_similar_videos_by_id，确保每个视频 dict 包含 'url' 和 'cover_url' 字段
    for video in VIDEO_DB:
        video["url"] = video.get("video_url", "N/A")
        video["cover_url"] = video.get("video_cover_url", "N/A")

    # 调用聚类推荐函数（视频直接传 dict）
    json_result = get_similar_videos_by_id(
        target_video_id=video_id,
        videos = videos,
        video_clusters_df=VIDEO_CLUSTER_DF,
        num_similar=20,
    )

    return jsonify(json.loads(json_result))

import json
import pandas as pd
from collections import defaultdict
import traceback  # 用于打印详细错误追踪
def get_similar_videos_by_id(
    target_video_id, videos, video_clusters_df, num_similar=10
):
    """
    根据视频聚类结果，查找与指定视频 ID 相似的其他视频。
    相似性基于观看这些视频的用户群体 (即，查找同一聚类中的其他热门视频)。

    Args:
        target_video_id (str): 目标视频的唯一 ID。
        videos (list): 包含 Video 对象的列表 (需要有 'id' 和 'url' 属性)。
        video_clusters_df (pd.DataFrame): 视频聚类结果，必须包含 '视频URL' 和 '聚类ID' 列。
                                          推荐也包含 '观看该视频的用户数' 以便排序。
        num_similar (int): 希望返回的相似视频数量。

    Returns:
        str: 包含相似视频列表 (id, title, cover_url) 的 JSON 字符串，
             或错误信息的 JSON 字符串。
    """
    # --- Input Validation ---
    if not target_video_id:
        error_info = {"error": "必须提供 target_video_id。"}
        return json.dumps(error_info, ensure_ascii=False, indent=4)

    if not isinstance(videos, list) or not videos:
        error_info = {"error": "视频列表 'videos' 无效或为空。"}
        return json.dumps(error_info, ensure_ascii=False, indent=4)

    if not isinstance(video_clusters_df, pd.DataFrame) or video_clusters_df.empty:
        error_info = {
            "error": "视频聚类结果 DataFrame 'video_clusters_df' 无效或为空。"
        }
        return json.dumps(error_info, ensure_ascii=False, indent=4)

    required_cols = ["视频URL", "聚类ID"]
    if not all(col in video_clusters_df.columns for col in required_cols):
        missing_cols = [
            col for col in required_cols if col not in video_clusters_df.columns
        ]
        error_info = {"error": f"视频聚类结果 DataFrame 缺少必需列: {missing_cols}。"}
        return json.dumps(error_info, ensure_ascii=False, indent=4)

    if not isinstance(num_similar, int) or num_similar <= 0:
        error_info = {"error": f"要查找的相似视频数量 ({num_similar}) 必须是正整数。"}
        return json.dumps(error_info, ensure_ascii=False, indent=4)

    try:
        # --- Find Target Video Object by ID ---
        # 创建 ID -> Video 和 URL -> Video 的查找字典
        video_id_lookup = {}
        video_url_lookup = {}
        for video in videos:
            if hasattr(video, "id"):
                video_id_lookup[str(getattr(video, "id", None))] = video
            if (
                hasattr(video, "url")
                and getattr(video, "url", None)
                and getattr(video, "url") != "N/A"
            ):
                video_url_lookup[str(getattr(video, "url"))] = video

        target_video = video_id_lookup.get(str(target_video_id))

        if not target_video:
            error_info = {
                "error": f"在视频列表中找不到 ID 为 '{target_video_id}' 的视频。"
            }
            return json.dumps(error_info, ensure_ascii=False, indent=4)

        # --- Get Target Video URL and Cluster ---
        target_url = getattr(target_video, "url", None)
        if not target_url or target_url == "N/A":
            error_info = {
                "error": f"ID 为 '{target_video_id}' 的视频 URL 无效或为 'N/A'。"
            }
            return json.dumps(error_info, ensure_ascii=False, indent=4)

        target_cluster_info = video_clusters_df[
            video_clusters_df["视频URL"] == target_url
        ]

        if target_cluster_info.empty:
            error_info = {
                "error": f"在聚类结果中找不到目标视频 URL: '{target_url}' (来自 ID '{target_video_id}')。请检查聚类 DataFrame 是否包含此 URL。"
            }
            return json.dumps(error_info, ensure_ascii=False, indent=4)

        target_cluster_id = target_cluster_info["聚类ID"].iloc[0]

        # --- Find Other Videos in the Same Cluster ---
        similar_video_rows = video_clusters_df[
            (video_clusters_df["聚类ID"] == target_cluster_id)
            & (video_clusters_df["视频URL"] != target_url)  # 排除目标视频本身
        ].copy()  # 使用 .copy() 避免警告

        if similar_video_rows.empty:
            result = {
                "message": f"聚类 {target_cluster_id} 中没有找到其他与视频 ID '{target_video_id}' (标题: '{getattr(target_video, 'title', 'N/A')}') 相似的视频。",
                "similar_videos": [],
            }
            return json.dumps(result, ensure_ascii=False, indent=4)

        # --- Retrieve Details and Rank ---
        # 确定排序依据
        sort_by_cluster_viewers = "观看该视频的用户数" in similar_video_rows.columns

        # 添加排序键列
        if sort_by_cluster_viewers:
            similar_video_rows["sort_key"] = pd.to_numeric(
                similar_video_rows["观看该视频的用户数"], errors="coerce"
            ).fillna(0)
        else:
            # Fallback: 使用 video 对象中的 viewers 集合大小
            viewer_counts = []
            for url in similar_video_rows["视频URL"]:
                video_obj = video_url_lookup.get(url)
                count = 0
                if (
                    video_obj
                    and hasattr(video_obj, "viewers")
                    and isinstance(getattr(video_obj, "viewers", None), set)
                ):
                    count = len(getattr(video_obj, "viewers"))
                viewer_counts.append(count)
            similar_video_rows["sort_key"] = viewer_counts

        # 排序
        similar_video_rows = similar_video_rows.sort_values(
            by="sort_key", ascending=False
        )

        # 提取 Top N 视频的信息
        top_similar_videos = []
        for _, row in similar_video_rows.head(num_similar).iterrows():
            url = row["视频URL"]
            video = video_url_lookup.get(url)
            if video:
                # 确保视频对象有必要的属性
                video_id_out = getattr(video, "id", "N/A_ID")  # ID for output JSON
                title_out = getattr(video, "title", "N/A") or "N/A"
                cover_url_out = getattr(video, "cover_url", "N/A") or "N/A"

                top_similar_videos.append(
                    {
                        "id": video_id_out,  # 使用视频对象的 ID
                        "title": title_out,
                        "cover": cover_url_out,
                    }
                )

        result = {"similar_videos": top_similar_videos}
        return json.dumps(result, ensure_ascii=False, indent=4)

    except KeyError as ke:
        error_info = {
            "error": f"处理聚类结果时发生键错误 (可能是列名 '{ke}' 不匹配或不存在)。"
        }
        print(
            f"详细错误追踪 (get_similar_videos_by_id, ID: {target_video_id} - KeyError):"
        )
        traceback.print_exc()
        return json.dumps(error_info, ensure_ascii=False, indent=4)
    except Exception as e:
        error_info = {
            "error": f"查找 ID '{target_video_id}' 的相似视频时发生意外错误: {str(e)}"
        }
        print(
            f"详细错误追踪 (get_similar_videos_by_id, ID: {target_video_id} - Exception):"
        )
        traceback.print_exc()
        return json.dumps(error_info, ensure_ascii=False, indent=4)
