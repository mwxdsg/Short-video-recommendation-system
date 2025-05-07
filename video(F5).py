# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_daily_views(file_path, target_url=None):
    """
    直接从"每日观看"工作表加载数据
    返回目标视频的历史观看量（日期索引的Series）
    """
    try:
        # 读取数据
        df = pd.read_excel(file_path, sheet_name="每日观看")
        
        # 验证数据结构
        if '视频URL' not in df.columns:
            raise ValueError("Excel中缺少'视频URL'列")
        
        # 自动识别日期列（兼容年月日和月日格式）
        date_cols = []
        for col in df.columns:
            if col == '视频URL':
                continue
            try:
                # 尝试解析为年月日格式
                pd.to_datetime(col, format='%Y-%m-%d')
                date_cols.append(col)
            except:
                try:
                    # 尝试解析为月日格式（自动补充当前年）
                    pd.to_datetime(f"2023-{col}", format='%Y-%m-%d')
                    date_cols.append(col)
                except:
                    continue
        
        if not date_cols:
            raise ValueError("未找到有效的日期列")
        
        merge_df = pd.read_csv(
        'merge.csv', 
        usecols=["video_url", "title","id"],
        dtype={'video_url': 'string', 'title': 'string','id':'int'},
        encoding='utf-8'
        )
    #4.合并数据
        df = df.merge(
                merge_df,
                left_on='视频URL',
                right_on='video_url',
                how='left'
            )
        df['title'] = df['title'].fillna('未知标题')

        # 如果指定了目标URL
        if target_url:
            if target_url not in df['视频URL'].values:
                raise ValueError(f"未找到视频URL: {target_url}")
            target_df = df[df['视频URL'] == target_url]
            if len(target_df) == 0:
                raise ValueError("视频数据为空")
            
            # 提取观看数据
            views_data = target_df.iloc[0][date_cols].astype(int)
            dates = pd.to_datetime(date_cols, errors='coerce')
            return pd.Series(
                views_data.values,
                index=dates,
                name='views'
            )
        
        # 未指定URL时返回全部数据
        return {
            row['视频URL']: pd.Series(
                row[date_cols].astype(int),
                index=pd.to_datetime(date_cols),
                name='views'
            )
            for _, row in df.iterrows()
        }
        
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        return None

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
    # 配置参数
    input_file = "user_behavior_7days.xlsx"
    target_url = "https://www.kuaishou.com/short-video/3x8pp3ssxjjatnw"
    
    # 1. 直接加载日观看数据
    daily_views = load_daily_views(input_file, target_url)
    if daily_views is None:
        return
    
    # 2. 执行预测
    forecast = hybrid_forecast(daily_views)
    
    #3.转换
    result = {
        "heatCurve": [
            {"time": date.strftime("%Y-%m-%d"), "views": int(views)}
            for date, views in forecast.items()
        ]
    }
    
    # 4. 输出JSON结果
    print(json.dumps(result, ensure_ascii=False, indent=2))
    

if __name__ == "__main__":
    main()
