import pandas as pd
from joblib import Memory
import warnings
warnings.filterwarnings('ignore')
import json

# ==================== 数据加载 ====================
memory = Memory("./cache", verbose=0)

@memory.cache
def load_data(file_path):
    # 只读取必要列并指定数据类型
    dtype_spec = {
        '用户ID': 'category',
        '视频URL': 'string', 
        '观看次数': 'uint32', 
        '点赞数': 'uint16',
        '封面地址':'string',
            }
    
    with pd.ExcelFile(file_path) as xls:
        watch_raw = pd.read_excel(
            xls, 
            sheet_name="热门排行",
            usecols=["视频URL", "封面地址"],
            dtype=dtype_spec
        )
        #3.处理merge
    merge_df = pd.read_csv(
        'merge.csv', 
        usecols=["video_url", "title","id"],
        dtype={'video_url': 'string', 'title': 'string','id':'int'},
        encoding='utf-8'
        )
    #4.合并数据
    watch_raw = watch_raw.merge(
            merge_df,
            left_on='视频URL',
            right_on='video_url',
            how='left'
        )
    watch_raw['title'] = watch_raw['title'].fillna('未知标题')
        
    return watch_raw

def to_json(raw):
    """将推荐结果转为格式化的JSON字符串"""
    if isinstance(raw, pd.DataFrame):
    # 重置索引并选择需要输出的列
        output_cols = ["title","id", "封面地址", "观看次数"]
        available_cols = [col for col in output_cols if col in raw.columns]
                
                # 转换为字典列表
        recs_dict = raw.reset_index()[available_cols].to_dict(orient='records')
                
                # 转换为JSON并美化输出
        return json.dumps({
                "raw": recs_dict,
                "count": len(raw)
            }, indent=2, ensure_ascii=False)
    else:
        return json.dumps({"error": "推荐结果不是DataFrame格式"}, ensure_ascii=False)
    
def main():
    watch_raw = load_data("user_behavior_7days.xlsx")
    watch_raw_json = to_json(watch_raw)
    print(watch_raw_json)
    return watch_raw_json

if __name__ == "__main__":
    main()