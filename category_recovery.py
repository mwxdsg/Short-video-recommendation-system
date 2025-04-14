# -*- coding: utf-8 -*-
import os
from openpyxl import load_workbook
from collections import defaultdict


def extract_categories_from_report(report_file="user_behavior_7days.xlsx"):
    """从之前生成的报告文件中提取视频分类信息"""
    if not os.path.exists(report_file):
        print(f"报告文件 {report_file} 不存在")
        return None

    try:
        # 加载Excel工作簿
        wb = load_workbook(report_file, read_only=True)

        # 尝试从分类统计表中获取信息
        if "Category Stats" in wb.sheetnames:
            categories = set()
            sheet = wb["Category Stats"]
            # 跳过标题行
            for row in list(sheet.rows)[1:]:
                if row[0].value:  # 确保分类名不为空
                    categories.add(row[0].value)
            print(f"从分类统计表中提取了 {len(categories)} 个分类")
        else:
            categories = set()

        # 从视频统计表中提取视频URL和分类的映射
        url_to_category = {}
        if "Video Statistics" in wb.sheetnames:
            sheet = wb["Video Statistics"]
            rows = list(sheet.rows)
            # 确认列结构
            header = [cell.value for cell in rows[0]]
            try:
                url_idx = header.index("Video URL")
                cat_idx = header.index("Category")

                # 提取映射关系
                for row in rows[1:]:
                    if len(row) > max(url_idx, cat_idx) and row[url_idx].value and row[cat_idx].value:
                        url_to_category[row[url_idx].value] = row[cat_idx].value

                print(f"从报告中提取了 {len(url_to_category)} 个视频的分类信息")
            except ValueError:
                print("在视频统计表中未找到必要的列")

        return {
            "categories": categories,
            "url_to_category": url_to_category
        }

    except Exception as e:
        print(f"从报告文件提取分类信息时出错: {str(e)}")
        return None
    finally:
        wb.close()


def apply_categories_to_videos(videos, url_to_category):
    """将提取的分类信息应用到视频对象"""
    if not url_to_category:
        return 0

    categories_applied = 0
    for video in videos:
        if video.url in url_to_category:
            video.category = url_to_category[video.url]
            categories_applied += 1

    return categories_applied


def recover_categories(videos, report_files=None):
    """从多个可能的报告文件中恢复分类信息"""
    if report_files is None:
        # 默认尝试的文件列表，按优先级排序
        report_files = [
            "user_behavior_7days.xlsx",
            "video_clusters.xlsx",
            "category_report.xlsx"
        ]

    for file in report_files:
        if not os.path.exists(file):
            continue

        print(f"尝试从 {file} 恢复分类信息...")
        result = extract_categories_from_report(file)

        if result and result["url_to_category"]:
            applied = apply_categories_to_videos(videos, result["url_to_category"])
            print(f"成功应用 {applied} 个分类 (共 {len(videos)} 个视频)")

            if applied > 0:
                return result["categories"], applied

    print("未能从任何报告文件中恢复分类信息")
    return set(), 0
