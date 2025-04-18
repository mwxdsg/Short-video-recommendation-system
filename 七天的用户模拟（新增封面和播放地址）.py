视频用户行为模拟系统 - 完整版（含每日观看统计和热门排行）

功能特点：
1. 从Excel加载视频数据
2. 自动分类视频（TF-IDF+层次聚类）
3. 模拟用户7天行为
4. 生成包含6个工作表的报告：
   - 分类统计
   - 观看记录（按天分列）
   - 点赞记录
   - 视频统计数据
   - 每日观看统计（新增）
   - 热门视频排行（新增）
"""

# ==================== 导入依赖库 ====================
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from openpyxl import load_workbook, Workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from tqdm import tqdm

# ==================== 全局配置 ====================
NUM_USERS = 1000
MIN_INTERESTS = 4
MAX_INTERESTS = 6
OUTPUT_FILENAME = "user_behavior_7days.xlsx"
DAYS_TO_SIMULATE = 7

# ==================== 核心类定义 ====================
class Video:
    """视频对象，存储元数据和行为统计"""
    def __init__(self, url, keywords, cover_url=None, play_url=None):
        self.url = url.strip()
        self.keywords = [kw for kw in (keywords.split() if keywords else []) if kw]
        self.cover_url = cover_url.strip() if cover_url else None  # 新增封面地址
        self.play_url = play_url.strip() if play_url else None      # 新增播放地址
        self.category = None
        self.like_count = 0
        self.viewers = set()  # 存储观看过的用户ID（自动去重）
        self.liked_users = set()  # 存储点赞过的用户ID

class User:
    """用户对象，包含兴趣偏好和行为记录"""
    def __init__(self, user_id, all_categories):
        self.user_id = f"U{user_id:04d}"
        self.interests = self._generate_interests(all_categories)
        self.watched_videos = []  # 格式: [(url, timestamp_str), ...]
        self.liked_videos = []    # 格式: [(url, timestamp_str), ...]
    
    def _generate_interests(self, categories):
        """随机生成用户兴趣分布"""
        if not categories:
            return {}
        # 确保兴趣数量不超过实际类别数
        num_interests = random.randint(
            min(MIN_INTERESTS, len(categories)),
            min(MAX_INTERESTS, len(categories)))
        chosen_cats = random.sample(categories, num_interests)
        return {cat: random.uniform(0.7, 1.0) for cat in chosen_cats}

# ==================== 数据处理函数 ====================
def load_excel_data(file_path):
    """从Excel加载视频数据（自动检测URL和关键词列）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件 {file_path} 不存在")

    try:
        wb = load_workbook(file_path)
        ws = wb.active
        videos = []
        col_mapping = {}

        # 自动检测列位置（兼容中英文列名）
        for col in range(1, ws.max_column + 1):
            header = ws.cell(row=1, column=col).value
            if header and 'url' in header.lower():
                col_mapping['url'] = col
            elif header and ('keyword' in header.lower() or '标签' in header.lower()):
                col_mapping['keywords'] = col
            elif header and ('cover' in header.lower() or '封面' in header.lower()):  # 新增封面地址检测
                col_mapping['cover_url'] = col
            elif header and ('play' in header.lower() or '播放' in header.lower()):    # 新增播放地址检测
                col_mapping['play_url'] = col

        if not col_mapping.get('url'):
            raise ValueError("未找到必要的URL列")

        # 读取视频数据
        for row in range(2, ws.max_row + 1):
            url = ws.cell(row=row, column=col_mapping['url']).value
            keywords = ws.cell(row=row, column=col_mapping['keywords']).value if 'keywords' in col_mapping else ""
            cover_url = ws.cell(row=row, column=col_mapping['cover_url']).value if 'cover_url' in col_mapping else None
            play_url = ws.cell(row=row, column=col_mapping['play_url']).value if 'play_url' in col_mapping else None
            
            if url:
                videos.append(Video(url, keywords, cover_url, play_url))

        return videos
    except Exception as e:
        raise ValueError(f"Excel读取错误: {str(e)}")

def auto_categorize(videos):
    """使用层次聚类对视频分类（最多400类）"""
    if not videos:
        print("警告: 没有视频数据可供分类")
        return {}

    # 准备文本特征
    video_docs = [' '.join(v.keywords) for v in videos if v.keywords]
    if not video_docs:
        print("警告: 无有效关键词")
        return {}

    # TF-IDF特征提取
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=5000,
        token_pattern=r'\b\w+\b'
    )
    try:
        tfidf_matrix = tfidf.fit_transform(video_docs)
    except ValueError:
        print("警告: 特征不足，跳过分类")
        return {}

    # 层次聚类
    MAX_CATEGORIES = 400
    actual_clusters = min(MAX_CATEGORIES, len(videos))
    
    clustering = AgglomerativeClustering(
        n_clusters=actual_clusters,
        metric='cosine',
        linkage='average'
    )
    clusters = clustering.fit_predict(tfidf_matrix.toarray())

    # 生成类别标签（基于高频关键词）
    category_map = {}
    cluster_keywords = defaultdict(list)
    feature_names = tfidf.get_feature_names_out()

    for idx, cluster_id in enumerate(clusters):
        vec = tfidf_matrix[idx]
        top_features = sorted(zip(vec.indices, vec.data), key=lambda x: -x[1])[:3]
        selected_keywords = [feature_names[i] for i, _ in top_features]
        cluster_keywords[cluster_id].extend(selected_keywords)

    for cluster_id, keywords in cluster_keywords.items():
        top_keywords = [kw for kw, _ in Counter(keywords).most_common(3)]
        label = "_".join(top_keywords[:2]) if top_keywords else "其他"
        category_map[cluster_id] = f"CAT_{label}"

    # 分配类别
    for idx, video in enumerate(videos):
        if video.keywords:
            video.category = category_map.get(clusters[idx], "其他")
        else:
            video.category = "无关键词"

    # 打印分类统计
    print("\n=== 分类统计 ===")
    category_counts = Counter(v.category for v in videos)
    for cat, count in category_counts.most_common(15):
        print(f"{cat:<25} {count:<10}")
    print(f"总分类数: {len(category_counts)}")

    return category_map

# ==================== 行为模拟函数 ====================
def simulate_behavior(users, videos):
    """模拟7天的用户观看和点赞行为"""
    if not users or not videos:
        print("警告: 缺少用户或视频数据")
        return

    # 按类别索引视频
    cat_index = defaultdict(list)
    for v in videos:
        if v.category:
            cat_index[v.category].append(v)

    # 7天日期范围（从6天前到今天）
    base_date = datetime.now().date()
    date_range = [base_date - timedelta(days=i) for i in range(DAYS_TO_SIMULATE-1, -1, -1)]

    for user in tqdm(users, desc="模拟用户行为"):
        for day in date_range:
            # 每天1-3个会话
            for _ in range(random.randint(1, 3)):
                # 生成会话时间（当天8-23点）
                session_time = datetime.combine(day, datetime.min.time()).replace(
                    hour=random.randint(8, 23),
                    minute=random.randint(0, 59))

                # 每个会话观看10-30个视频
                for _ in range(random.randint(10, 30)):
                    # 90%概率按兴趣选择视频
                    if user.interests and random.random() < 0.9:
                        chosen_cat = random.choices(
                            list(user.interests.keys()),
                            weights=list(user.interests.values()),
                            k=1
                        )[0]
                        candidates = cat_index.get(chosen_cat, [])
                    else:
                        candidates = videos

                    if candidates:
                        video = random.choice(candidates)
                        _record_watch(user, video, session_time)

                        # 动态计算点赞概率（基础20% + 兴趣权重）
                        like_prob = 0.2
                        if video.category in user.interests:
                            like_prob += user.interests[video.category] * 0.8

                        if random.random() < like_prob:
                            _record_like(user, video, session_time)

def _record_watch(user, video, time):
    """记录观看行为（自动去重）"""
    entry = (video.url, time.strftime('%Y-%m-%d %H:%M:%S'))
    user.watched_videos.append(entry)
    video.viewers.add(user.user_id)

def _record_like(user, video, time):
    """记录点赞行为（确保单用户不重复点赞）"""
    if user.user_id not in video.liked_users:
        entry = (video.url, time.strftime('%Y-%m-%d %H:%M:%S'))
        user.liked_videos.append(entry)
        video.like_count += 1
        video.liked_users.add(user.user_id)

# ==================== 新增：每日观看统计 ====================
def count_daily_views(videos, users, days_to_simulate=7):
    """统计每个视频每天的观看次数"""
    base_date = datetime.now().date()
    date_range = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') 
                 for i in range(days_to_simulate-1, -1, -1)]
    
    daily_stats = {v.url: {date: 0 for date in date_range} for v in videos}
    
    for user in users:
        for url, time_str in user.watched_videos:
            watch_date = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
            if watch_date in date_range:
                daily_stats[url][watch_date] += 1
    
    return daily_stats

# ==================== 新增：热门视频排行 ====================
def merge_sort_videos(videos, key_func):
    """归并排序实现（稳定排序）"""
    if len(videos) <= 1:
        return videos
    
    mid = len(videos) // 2
    left = merge_sort_videos(videos[:mid], key_func)
    right = merge_sort_videos(videos[mid:], key_func)
    return _merge(left, right, key_func)

def _merge(left, right, key_func):
    """合并两个已排序的子列表"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        # 降序排列（热度高的在前）
        if key_func(left[i]) > key_func(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def generate_hot_ranking(videos, top_n=100):
    """生成热门视频排行榜"""
    def sort_key(video):
        # 热度公式：观看次数*0.2 + 点赞数*0.8
        return len(video.viewers) * 0.2 + video.like_count * 0.8
    
    # 使用归并排序
    sorted_videos = merge_sort_videos(videos, key_func=sort_key)
    
    # 构建排行数据
    ranking = []
    for rank, video in enumerate(sorted_videos[:top_n], 1):
        ranking.append({
            "排名": rank,
            "视频URL": video.url,
            "类别": video.category,
            "观看数": len(video.viewers),
            "点赞数": video.like_count,
            "热度": round(sort_key(video), 2)
        })
    
    return ranking

# ==================== 报告生成函数 ====================
def save_detailed_report(users, videos, filename):
    """生成包含6个工作表的Excel报告"""
    try:
        wb = Workbook()
        
        # Sheet 1: 分类统计
        _create_category_sheet(wb, videos)
        
        # Sheet 2: 观看记录（按天分列）
        _create_watch_history_sheet(wb, users)
        
        # Sheet 3: 点赞记录
        _create_like_history_sheet(wb, users)
        
        # Sheet 4: 视频统计
        _create_video_stats_sheet(wb, videos)
        
        # Sheet 5: 每日观看统计（新增）
        _create_daily_views_sheet(wb, videos, users)
        
        # Sheet 6: 热门视频排行（新增）
        _create_hot_ranking_sheet(wb, videos)
        
        wb.save(filename)
    except Exception as e:
        raise IOError(f"Excel保存失败: {str(e)}")

# 各工作表的详细创建函数
def _create_category_sheet(wb, videos):
    ws = wb.active
    ws.title = "分类统计"
    ws.append(["类别名称", "视频数量"])
    for cat, count in Counter(v.category for v in videos).most_common():
        ws.append([cat, count])

def _create_watch_history_sheet(wb, users):
    ws = wb.create_sheet("观看记录")
    date_headers = [(datetime.now().date() - timedelta(days=i)).strftime("%m-%d") 
                   for i in range(DAYS_TO_SIMULATE-1, -1, -1)]
    ws.append(["用户ID"] + date_headers)
    
    for user in users:
        daily_watches = {day: [] for day in date_headers}
        for url, time_str in user.watched_videos:
            day_key = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').strftime("%m-%d")
            if day_key in daily_watches:
                daily_watches[day_key].append(url)
        
        ws.append([user.user_id] + [", ".join(daily_watches[day]) for day in date_headers])

def _create_like_history_sheet(wb, users):
    ws = wb.create_sheet("点赞记录")
    ws.append(["用户ID", "视频URL", "点赞时间"])
    for user in users:
        for url, time in user.liked_videos:
            ws.append([user.user_id, url, time])

def _create_video_stats_sheet(wb, videos):
    ws = wb.create_sheet("视频统计")
    # 添加新字段到表头
    ws.append(["视频URL", "播放地址", "封面地址", "类别", "观看次数", "点赞数"])
    for video in videos:
        # 添加新字段到数据行
        ws.append([
            video.url, 
            video.play_url or "", 
            video.cover_url or "", 
            video.category, 
            len(video.viewers), 
            video.like_count
        ])

def _create_daily_views_sheet(wb, videos, users):
    ws = wb.create_sheet("每日观看")
    daily_views = count_daily_views(videos, users)
    
    dates = [(datetime.now().date() - timedelta(days=i)).strftime("%Y-%m-%d") 
            for i in range(DAYS_TO_SIMULATE-1, -1, -1)]
    ws.append(["视频URL"] + dates)
    
    for video in videos:
        ws.append([video.url] + [daily_views[video.url][date] for date in dates])


def _create_hot_ranking_sheet(wb, videos):
    ws = wb.create_sheet("热门排行")
    ranking = generate_hot_ranking(videos)
    
    # 添加新字段到表头
    headers = ["排名", "视频URL", "播放地址", "封面地址", "类别", "观看数", "点赞数", "热度"]
    ws.append(headers)
    
    for item in ranking:
        video = next((v for v in videos if v.url == item["视频URL"]), None)
        # 添加新字段到数据行
        ws.append([
            item["排名"],
            item["视频URL"],
            video.play_url if video else "",
            video.cover_url if video else "",
            item["类别"],
            item["观看数"],
            item["点赞数"],
            item["热度"]
        ])

# ==================== 主程序 ====================
if __name__ == "__main__":
    try:
        print("正在加载视频数据...")
        videos = load_excel_data("videos.xlsx")
        print(f"加载成功，共 {len(videos)} 个视频")

        print("正在进行视频分类...")
        categories = auto_categorize(videos)

        print(f"创建 {NUM_USERS} 个模拟用户...")
        valid_categories = [c for c in categories.values() if c not in ["其他", "无关键词"]]
        users = [User(i, valid_categories) for i in range(NUM_USERS)]

        print(f"模拟 {DAYS_TO_SIMULATE} 天用户行为...")
        simulate_behavior(users, videos)

        print(f"生成报告 {OUTPUT_FILENAME}...")
        save_detailed_report(users, videos, OUTPUT_FILENAME)

        # 打印汇总统计
        total_views = sum(len(u.watched_videos) for u in users)
        total_likes = sum(len(u.liked_videos) for u in users)
        print(f"\n=== 模拟结果 ===")
        print(f"总观看次数: {total_views}")
        print(f"总点赞次数: {total_likes}")
        print(f"点赞率: {total_likes/total_views:.1%}")
        print(f"已保存到 {OUTPUT_FILENAME}")

    except Exception as e:
        print(f"错误发生: {str(e)}")