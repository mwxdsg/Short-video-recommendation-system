import random
import re
from flask import Blueprint, jsonify, request
from data_load import VIDEO_DB
from searcher import Search, recommendations_to_json, format_for_frontend

search_bp = Blueprint("search", __name__)

searcher = Search("user_behavior_7days.xlsx")


@search_bp.route("/search", methods=["GET"])
def video_search():
    # 获取关键词参数
    keyword = request.args.get("keyword", "")

    # 安全检查：空关键词
    if not keyword.strip():
        return jsonify({"error": "缺少搜索关键词"}), 400

    # 搜索调用
    results_df = searcher.search(query=keyword, top_n=20)
    results = format_for_frontend(results_df)
    
    # 转为 JSON 格式返回
    return results
