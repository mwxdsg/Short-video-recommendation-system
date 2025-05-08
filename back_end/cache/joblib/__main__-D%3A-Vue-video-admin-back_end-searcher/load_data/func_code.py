# first line: 16
@memory.cache
def load_data(file_path):
    """加载视频统计数据（仅使用类别字段）"""
    dtype_spec = {
        "用户ID": "category",
        "视频URL": "string",
        "类别": "category",
        "观看次数": "uint32",
        "点赞数": "uint16",
        "封面地址": "string",
    }

    with pd.ExcelFile(file_path) as xls:
        videos = pd.read_excel(
            xls,
            sheet_name="视频统计",
            usecols=["视频URL", "类别", "观看次数", "点赞数", "封面地址"],
            dtype=dtype_spec,
        )

    merge_df = pd.read_csv(
        "merge.csv",
        usecols=["video_url", "title", "id"],
        dtype={"video_url": "string", "title": "string", "id": "int"},
        encoding="utf-8",
    )
    # 4.合并数据
    videos = videos.merge(merge_df, left_on="视频URL", right_on="video_url", how="left")
    videos["title"] = videos["title"].fillna("未知标题")
    return videos
