# -*- coding: utf-8 -*-
""" 视频观看量预测系统（适配7天分列格式） """
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_preprocess(file_path, sheet_name):
    """
    加载并预处理7天分列格式的观看数据
    返回包含[用户ID, 视频URL, 观看时间]的DataFrame
    """
    try:
        # 读取原始Excel数据
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 验证必要列是否存在
        if '用户ID' not in raw_df.columns:
            raise ValueError("Excel文件中缺少'银狐ID'列")
        
        # 转换7天分列格式为长格式
        date_columns = [col for col in raw_df.columns if col != '用户ID']
        if not date_columns:
            raise ValueError("未找到日期列")
        
        processed_data = []
        for date_col in date_columns:
            # 提取日期部分（兼容不同列名格式）
            date_match = re.search(r'(\d{1,2}-\d{1,2})', date_col)
            if not date_match:
                logging.warning(f"跳过无法解析的列: {date_col}")
                continue
                
            date_str = date_match.group(1)
            try:
                base_date = datetime.strptime(f"2023-{date_str}", "%Y-%m-%d").date()
            except ValueError:
                logging.warning(f"日期解析失败: {date_str}")
                continue
            
            # 处理每个日期列的数据
            temp_df = raw_df[['用户ID', date_col]].copy()
            temp_df.columns = ['用户ID', '视频URL']
            
            # 拆分逗号分隔的URL并生成观看时间
            temp_df = (
                temp_df.dropna(subset=['视频URL'])
                .assign(视频URL=lambda x: x['视频URL'].str.split(r',\s*'))
                .explode('视频URL')
                .assign(视频URL=lambda x: x['视频URL'].str.strip())
                .query("视频URL != ''")
            )
            
            # 为每条记录生成随机小时（8-23点）
            np.random.seed(42)  # 保证可重复性
            random_hours = np.random.randint(8, 24, size=len(temp_df))
            temp_df['观看时间'] = [
                datetime.combine(base_date, datetime.min.time()) + pd.Timedelta(hours=int(h))
                for h in random_hours
            ]
            
            processed_data.append(temp_df)
        
        if not processed_data:
            raise ValueError("未找到有效数据")
            
        # 合并所有日期的数据
        df = pd.concat(processed_data, ignore_index=True)
        
        logging.info(f"数据加载成功，共处理{len(df)}条观看记录")
        return df[['用户ID', '视频URL', '观看时间']]
    
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        return None

def get_daily_views(df, video_url):
    """
    获取指定视频的日观看量
    返回按日期索引的Series（确保连续日期）
    """
    try:
        # 过滤目标视频并统计每日观看量
        views = (
            df[df["视频URL"] == video_url.strip()]
            .set_index("观看时间")
            .resample('D')
            .size()
            .rename('views')
        )
        
        # 确保完整的日期范围（填充缺失日期为0）
        if not views.empty:
            full_range = pd.date_range(
                start=views.index.min().floor('D'),
                end=views.index.max().ceil('D'),
                freq='D'
            )
            views = views.reindex(full_range, fill_value=0)
        
        return views
    except Exception as e:
        logging.error(f"日观看量统计失败: {str(e)}")
        return pd.Series(dtype=int)

def hybrid_forecast(historical_data, forecast_days=7):
    """
    混合预测方法（结合ARIMA、线性回归和加权移动平均）
    返回包含预测日期索引的Series
    """
    # 确保输入数据有效
    if historical_data.empty:
        logging.warning("无历史数据，使用默认预测")
        return _generate_default_forecast(forecast_days)
    
    # 数据预处理
    clean_data = historical_data.copy()
    clean_data = clean_data[clean_data > 0]  # 移除零值对模型的影响
    
    # 情况1：数据量充足时使用ARIMA
    if len(clean_data) >= 14:  # 至少2周数据
        try:
            model = ARIMA(clean_data, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_days)
            return _format_forecast(forecast, historical_data)
        except Exception as e:
            logging.warning(f"ARIMA建模失败: {str(e)}")
    
    # 情况2：小样本混合预测
    # 加权移动平均（最近3天权重最高）
    wma_weights = np.array([0.5, 0.3, 0.2])
    last_values = clean_data.iloc[-3:].values
    wma = np.sum(last_values * wma_weights[:len(last_values)]) / np.sum(wma_weights[:len(last_values)])
    
    # 线性趋势预测
    trend = 0
    if len(clean_data) >= 5:
        X = np.arange(len(clean_data)).reshape(-1, 1)
        y = clean_data.values
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict([[len(clean_data)]])[0]
    
    # 混合预测值
    blended = (wma + max(trend, 0)) / 2
    blended = max(blended, 1)  # 确保至少1次观看
    
    # 添加合理噪声
    noise = np.random.normal(0, blended*0.1, forecast_days)
    forecast_values = np.round(np.clip(blended + noise, 1, None))
    
    return _format_forecast(forecast_values, historical_data)

def _generate_default_forecast(days):
    """ 生成默认预测（新视频冷启动） """
    return pd.Series(
        [1] * days,
        index=pd.date_range(
            start=datetime.now().date(),
            periods=days,
            freq='D'
        ),
        name='forecast'
    )

def _format_forecast(values, historical_data):
    """ 格式化预测结果 """
    last_date = historical_data.index.max()
    return pd.Series(
        values,
        index=pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(values),
            freq='D'
        ),
        name='forecast'
    )

def main():
    """ 主执行流程 """
    # 配置参数
    input_file = "user_behavior_7days.xlsx"
    sheet_name = "Watch History"
    target_video = "https://www.kuaishou.com/short-video/3x3pxpxeaz8jipk"  # 替换为实际视频URL
    
    # 1. 数据加载
    logging.info("开始加载数据...")
    df = load_and_preprocess(input_file, sheet_name)
    if df is None:
        return
    
    # 2. 获取目标视频数据
    logging.info(f"分析视频: {target_video}")
    daily_views = get_daily_views(df, target_video)
    
    if daily_views.empty:
        logging.warning("该视频暂无观看数据，使用冷启动预测")
        forecast = _generate_default_forecast(7)
    else:
        # 3. 执行预测
        logging.info(f"有效历史数据天数: {len(daily_views)}")
        forecast = hybrid_forecast(daily_views)
    
    # 4. 结果处理与输出
    forecast = forecast.astype(int)  # 确保整数结果
    print("\n=== 预测结果 ===")
    print(forecast.to_string())
    

if __name__ == "__main__":
    main()