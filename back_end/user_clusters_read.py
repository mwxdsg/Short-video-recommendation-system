import pandas as pd
import os # 用于检查文件是否存在

def load_data_from_excel_sheet_with_dtypes(excel_filepath, sheet_name, expected_dtypes=None):

    if not os.path.exists(excel_filepath):
        print(f"错误：Excel文件 '{excel_filepath}' 未找到。")
        return pd.DataFrame()

    try:
        print(f"正在从 Excel 文件 '{excel_filepath}' 的工作表 '{sheet_name}' 加载数据...")
        if expected_dtypes:
            print(f"  尝试使用预定义的数据类型: {expected_dtypes}")
            reloaded_df = pd.read_excel(excel_filepath, sheet_name=sheet_name, dtype=expected_dtypes)
        else:
            print("  将由 pandas 自动推断数据类型。")
            reloaded_df = pd.read_excel(excel_filepath, sheet_name=sheet_name)

        print(f"成功从Excel加载 {len(reloaded_df)} 条记录。")
        print("加载后的DataFrame信息 (包含数据类型):")
        reloaded_df.info()
        return reloaded_df
    except FileNotFoundError:
        print(f"错误：Excel文件 '{excel_filepath}' 未找到（在尝试读取时）。")
        return pd.DataFrame()
    except ValueError as ve:
        print(f"从Excel文件 '{excel_filepath}' 加载数据时发生值错误: {ve}")
        print(f"  请确保工作表名称 '{sheet_name}' 正确无误。")
        return pd.DataFrame()
    except Exception as e:
        print(f"从Excel文件 '{excel_filepath}' 的工作表 '{sheet_name}' 加载数据时发生其他错误: {e}")
        return pd.DataFrame()

# --- 示例用法 ---
if __name__ == "__main__":
    # 这是您的主脚本保存的用户聚类Excel文件的路径
    user_cluster_excel_file_path = "user_clusters.xlsx"

    # --- 1. 加载 "用户聚类" 工作表 ---
    main_sheet_name = "用户聚类"

    # !!! 重要 !!!
    # 这个 'expected_user_cluster_dtypes' 字典应该基于您主脚本中打印出的
    # `user_clusters.dtypes.to_dict()` 的输出。
    # 您需要从主脚本的输出中复制这个字典结构，并修改值的格式。
    # 例如，如果主脚本输出 `{'聚类ID': dtype('int64')}`，这里应该是 `{'聚类ID': 'int64'}`。
    #
    # 根据您提供的 `user_clustering.py` 中 `cluster_users_by_interests` 函数的输出结构：
    expected_user_cluster_dtypes = {
        "用户ID": "object",        # 字符串
        "聚类ID": "int64",          # 整数
        "聚类大小": "int64",        # 整数
        "观看类别数": "int64",    # 整数
        "典型类别": "object"       # 字符串
    }
    # 确保这里的列名与您的Excel文件 "用户聚类" 工作表中的列名完全匹配。

    print(f"\n--- 加载 '{main_sheet_name}' 工作表 (用户聚类详情) ---")

    # 为了使这个示例能独立运行，如果Excel文件不存在，我们创建一个虚拟的
    if not os.path.exists(user_cluster_excel_file_path):
        print(f"警告: '{user_cluster_excel_file_path}' 不存在。将创建一个虚拟Excel文件用于测试。")
        dummy_data_main = {
            "用户ID": ["U0001", "U0002", "U0003"],
            "聚类ID": [1, 0, 1],
            "聚类大小": [20, 15, 20],
            "观看类别数": [5, 3, 6],
            "典型类别": ["CAT_001(3); CAT_002(2)", "CAT_005(2); CAT_008(1)", "CAT_001(4); CAT_003(2)"]
        }
        dummy_df_main = pd.DataFrame(dummy_data_main)

        dummy_data_stats = {
            "聚类ID": [0, 1],
            "用户数量": [15, 20],
            "平均类别数": [3.0, 5.5] # 注意这个是 float
        }
        dummy_df_stats = pd.DataFrame(dummy_data_stats)

        try:
            with pd.ExcelWriter(user_cluster_excel_file_path, engine='openpyxl') as writer:
                dummy_df_main.to_excel(writer, sheet_name=main_sheet_name, index=False)
                dummy_df_stats.to_excel(writer, sheet_name="聚类统计", index=False)
            print(f"虚拟Excel文件 '{user_cluster_excel_file_path}' 已创建。")
        except Exception as ex_create_err:
            print(f"创建虚拟Excel文件失败: {ex_create_err}")
            print("  请确保已安装 'openpyxl' 库 (pip install openpyxl)。")

    # 使用这个变量来存储从 "用户聚类" sheet 加载的数据
    reloaded_user_clusters_main = load_data_from_excel_sheet_with_dtypes(
        excel_filepath=user_cluster_excel_file_path,
        sheet_name=main_sheet_name,
        expected_dtypes=expected_user_cluster_dtypes
    )

    if not reloaded_user_clusters_main.empty:
        print(f"\n从Excel '{main_sheet_name}' 重新加载的用户聚类数据:")
        print(reloaded_user_clusters_main.head())
        print("\n数据类型:")
        print(reloaded_user_clusters_main.dtypes) # 再次确认数据类型