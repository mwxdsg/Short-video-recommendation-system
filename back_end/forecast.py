from anyio import key
from flask import Blueprint, request, jsonify
from forecaster import (
    load_daily_views,
    hybrid_forecast
)  
from data_load import VIDEO_DB  
import re

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/video-analysis", methods=["GET"])
def analyze_video():
    try:
        video_id = request.args.get("link", type=int)
        if video_id is None:
            return jsonify({"error": "缺少视频 ID 参数"}), 400

        # 根据 ID 查找对应的视频 URL
        target_video = next((v for v in VIDEO_DB if v["id"] == video_id), None)
        if not target_video:
            return jsonify({"error": f"未找到ID为 {video_id} 的视频"}), 404

        target_url = target_video["video_urll"]
        print(f"目标视频 URL: {target_url}")

        raw_keywords = target_video.get("keywords", "")
        keyword_list = re.split(r"[，,\s]+", raw_keywords.strip())
        keyword_list = [kw for kw in keyword_list if kw]
        # 目标视频的 URL

        # Excel 数据路径
        input_file = "user_behavior_7days.xlsx"

        # 加载目标视频的历史观看量和类别
        daily_views, category = load_daily_views(input_file, target_url)
        if daily_views is None:
            return jsonify({"error": "数据加载失败"}), 500

        # 执行预测
        forecast = hybrid_forecast(daily_views)

        # 构造响应
        result = {
            "title": target_video["title"],
            "heatCurve": [
                {"time": date.strftime("%Y-%m-%d"), "views": int(views)}
                for date, views in forecast.items()
            ],
            "keywords": keyword_list,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
