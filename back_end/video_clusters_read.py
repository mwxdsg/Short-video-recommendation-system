import pandas as pd
import os # 用于检查文件是否存在

def load_clusters_from_excel_with_dtypes(excel_filepath, sheet_name, expected_dtypes=None):

    if not os.path.exists(excel_filepath):
        print(f"错误：Excel文件 '{excel_filepath}' 未找到。")
        return pd.DataFrame()

    try:
        print(f"正在从 Excel 文件 '{excel_filepath}' 的工作表 '{sheet_name}' 加载数据...")
        if expected_dtypes:
            print(f"  尝试使用预定义的数据类型: {expected_dtypes}")
            # 使用 dtype 参数来指定列的数据类型
            # 对于 .xlsx 文件，默认引擎通常是 'openpyxl'，确保已安装 (pip install openpyxl)
            reloaded_df = pd.read_excel(excel_filepath, sheet_name=sheet_name, dtype=expected_dtypes)
        else:
            print("  将由 pandas 自动推断数据类型。")
            reloaded_df = pd.read_excel(excel_filepath, sheet_name=sheet_name)

        print(f"成功从Excel加载 {len(reloaded_df)} 条记录。")
        print("加载后的DataFrame信息 (包含数据类型):")
        reloaded_df.info() # .info() 会显示每列的数据类型
        return reloaded_df
    except FileNotFoundError:
        print(f"错误：Excel文件 '{excel_filepath}' 未找到（在尝试读取时）。")
        return pd.DataFrame()
    except ValueError as ve: # 通常是 sheet_name 找不到
        print(f"从Excel文件 '{excel_filepath}' 加载数据时发生值错误: {ve}")
        print(f"  请确保工作表名称 '{sheet_name}' 正确无误。")
        return pd.DataFrame()
    except Exception as e:
        print(f"从Excel文件 '{excel_filepath}' 的工作表 '{sheet_name}' 加载数据时发生其他错误: {e}")
        return pd.DataFrame()

# --- 示例用法 ---
if __name__ == "__main__":
    # 这是您的主脚本保存的Excel文件的路径
    cluster_excel_file_path = "video_clusters_optimized_balanced.xlsx"
    # 这是您要读取的工作表的名称
    target_sheet_name = "视频聚类详情" # 主数据通常在这个sheet

    expected_column_dtypes = {
        "视频URL": "object",        # 'object' 通常用于字符串
        "聚类ID": "int64",          # 或者 'int32' 如果数值范围允许
        "观看该视频的用户数": "int64", # 或者 'int32'
        "聚类内视频数": "int64"       # 或者 'int32'
    }
    # 确保这里的列名与您的Excel文件 "视频聚类详情" 工作表中的列名完全匹配。

    print(f"\n--- 场景 1: 从Excel '{target_sheet_name}' 工作表使用期望的数据类型加载 ---")

    # 为了使这个示例能独立运行，如果Excel文件不存在，我们创建一个虚拟的
    if not os.path.exists(cluster_excel_file_path):
        print(f"警告: '{cluster_excel_file_path}' 不存在。将创建一个虚拟Excel文件用于测试。")
        # 确保虚拟数据的列与 expected_column_dtypes 中的键匹配
        dummy_data_details = {
            "视频URL": ["http://example.com/vid1_excel", "http://example.com/vid2_excel", "http://example.com/vid3_excel"],
            "聚类ID": [10, 11, 10],
            "观看该视频的用户数": [1000, 500, 1200],
            "聚类内视频数": [20, 10, 20]
        }
        dummy_df_details = pd.DataFrame(dummy_data_details)

        # 您的Excel文件有两个sheet，我们也创建两个
        dummy_data_stats = { # 假设的统计数据表结构 (与主脚本中的一致)
            "聚类ID": [10, 11],
            "视频数量": [2, 1],
            "平均观看用户数_每个视频": [1100.0, 500.0], # 注意这个可能是 float
            "实际簇大小": [20,10]
        }
        dummy_df_stats = pd.DataFrame(dummy_data_stats)

        try:
            # 需要安装 openpyxl: pip install openpyxl
            with pd.ExcelWriter(cluster_excel_file_path, engine='openpyxl') as writer:
                dummy_df_details.to_excel(writer, sheet_name="视频聚类详情", index=False)
                dummy_df_stats.to_excel(writer, sheet_name="聚类统计", index=False)
            print(f"虚拟Excel文件 '{cluster_excel_file_path}' 已创建。")
        except Exception as ex_create_err:
            print(f"创建虚拟Excel文件失败: {ex_create_err}")
            print("  请确保已安装 'openpyxl' 库 (pip install openpyxl)。")


    reloaded_clusters_from_excel_with_types = load_clusters_from_excel_with_dtypes(
        excel_filepath=cluster_excel_file_path,
        sheet_name=target_sheet_name,
        expected_dtypes=expected_column_dtypes
    )

    if not reloaded_clusters_from_excel_with_types.empty:
        print(f"\n从Excel '{target_sheet_name}' 重新加载的视频聚类数据 (指定了Dtypes):")
        print(reloaded_clusters_from_excel_with_types.head())
        print("\n数据类型:")
        print(reloaded_clusters_from_excel_with_types.dtypes) # 再次确认数据类型