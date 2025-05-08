# first line: 15
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
