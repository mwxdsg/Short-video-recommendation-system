# Short-video-recommendation-system
Course project
数据结构好难
word2vec数据库下载：https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg（怎么用，自己搜下ai）
先把初步成果传一下
# -*- coding: utf-8 -*-
import random
import os
from datetime import datetime
from collections import defaultdict, Counter
from openpyxl import load_workbook, Workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 全局配置
NUM_USERS = 1000        # 模拟用户数量
MIN_INTERESTS = 2       # 每个用户最少兴趣类别数  
MAX_INTERESTS = 4       # 每个用户最多兴趣类别数
OUTPUT_FILENAME = "user_behavior_details_tfidf.xlsx"  # 输出文件名
SIMILARITY_THRESHOLD = 0.5  # 余弦相似度阈值

class Video:
    def __init__(self, url, keywords):
        self.url = url.strip()
        self.keywords = [kw for kw in (keywords.split() if keywords else []) if kw]  # 空格分割关键词并过滤空值
        self.category = None
        self.like_count = 0  # 视频累计点赞数
        self.viewers = set() # 记录观看过该视频的用户ID

class User:
    def __init__(self, user_id, all_categories):
        self.user_id = f"U{user_id:04d}"
        self.interests = self._generate_interests(all_categories)
        self.watched_videos = []  # 格式: [(video_url, timestamp)]
        self.liked_videos = []    # 格式: [(video_url, timestamp)]
    
    def _generate_interests(self, categories):
        if not categories:
            return {}
            
        num_interests = random.randint(
            min(MIN_INTERESTS, len(categories)),
            min(MAX_INTERESTS, len(categories))
        )
        chosen_cats = random.sample(categories, num_interests)
        return {cat: random.uniform(0.5, 1.0) for cat in chosen_cats}

def load_excel_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found")
    
    try:
        wb = load_workbook(file_path)
        ws = wb.active  
        videos = []
        
        # 自动检测列位置
        col_mapping = {}
        for col in range(1, ws.max_column + 1):
            header = ws.cell(row=1, column=col).value
            if header and 'url' in header.lower():
                col_mapping['url'] = col
            elif header and ('keyword' in header.lower() or 'tag' in header.lower()):
                col_mapping['keywords'] = col
        
        if not col_mapping:
            raise ValueError("Required columns (url/keywords) not found in Excel file")
        
        for row in range(2, ws.max_row + 1):
            url = ws.cell(row=row, column=col_mapping['url']).value
            keywords = ws.cell(row=row, column=col_mapping['keywords']).value or ""
            if url:
                videos.append(Video(url, keywords))
        
        return videos
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")

def auto_categorize(videos):
    #使用TF-IDF和余弦相似度进行关键词分类
    if not videos:
        print("Warning: No videos provided for categorization")
        return {}

    # 获取所有唯一关键词（过滤空字符串）
    all_keywords = list(set(kw for v in videos for kw in v.keywords if kw))
    
    if not all_keywords:
        print("Warning: No valid keywords found in videos")
        return {}
    
    print("Calculating TF-IDF vectors...")
    try:
        # 将每个关键词视为一个"文档"来计算TF-IDF
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        # 每个关键词需要表示为列表形式(单元素列表)
        keyword_docs = [[kw] for kw in all_keywords]
        tfidf_matrix = tfidf.fit_transform(keyword_docs)
    except Exception as e:
        print(f"Error in TF-IDF calculation: {str(e)}")
        return {}

    print("Computing cosine similarities...")
    try:
        # 计算所有关键词之间的相似度矩阵
        similarity_matrix = cosine_similarity(tfidf_matrix)
    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        return {}

    # 基于相似度阈值进行分组
    groups = []
    used_keywords = set()
    
    for i, keyword in enumerate(tqdm(all_keywords, desc="Grouping keywords")):
        if keyword in used_keywords:
            continue
        
        # 找到相似的关键词
        similar_indices = [
            j for j, sim in enumerate(similarity_matrix[i]) 
            if sim >= SIMILARITY_THRESHOLD and i != j
        ]
        similar_keywords = [all_keywords[j] for j in similar_indices]
        
        # 合并当前关键词和相似关键词
        group = [keyword] + similar_keywords
        groups.append(group)
        used_keywords.update(group)
    
    # 统计每个组的关键词在所有视频中的出现频率
    print("Generating category labels...")
    category_labels = {}
    for i, group in enumerate(groups):
        word_counts = Counter()
        
        # 统计这些关键词在所有视频中的出现频率
        for v in videos:
            for kw in v.keywords:
                if kw in group:
                    word_counts[kw] += 1
        
        if word_counts:
            most_common = word_counts.most_common(1)[0][0]
            category_name = f"CAT-{i}_{most_common}"
            category_labels[category_name] = group
    
    # 为视频分配类别
    print("Assigning categories to videos...")
    uncategorized = 0
    for v in tqdm(videos, desc="Categorizing videos"):
        if not v.keywords:
            uncategorized += 1
            continue
            
        for cat_name, keywords in category_labels.items():
            if any(kw in keywords for kw in v.keywords):
                v.category = cat_name
                break
        else:
            uncategorized += 1
            v.category = "CAT-OTHER"  # 添加默认类别
    
    if uncategorized > 0:
        print(f"Note: {uncategorized} videos assigned to default category")
    
    return category_labels

def list_all_categories(videos):
    category_stats = defaultdict(int)
    for video in videos:
        if video.category:
            category_stats[video.category] += 1
    
    print("\n=== Category Statistics ===")
    print("{:<30} {:<10}".format("Category Name", "Video Count"))
    print("-" * 45)
    for category, count in sorted(category_stats.items(), 
                                key=lambda x: x[1], reverse=True):
        print("{:<30} {:<10}".format(category, count))
    
    return category_stats

def simulate_behavior(users, videos):
    if not users or not videos:
        print("Warning: No users or videos provided for simulation")
        return
        
    cat_index = defaultdict(list)
    for v in videos:
        if v.category:
            cat_index[v.category].append(v)
    
    for user in tqdm(users, desc="Simulating behaviors"):
        for _ in range(random.randint(1, 3)):
            session_time = datetime.now().replace(
                hour=random.randint(8, 23),
                minute=random.randint(0, 59))
            
            for _ in range(random.randint(2, 8)):
                if user.interests and random.random() < 0.7:
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
                    
                    like_prob = 0.2
                    if video.category in user.interests:
                        like_prob += 0.6 * user.interests[video.category]
                    
                    if random.random() < like_prob:
                        _record_like(user, video, session_time)

def _record_watch(user, video, time):
    entry = (video.url, time.strftime('%Y-%m-%d %H:%M:%S'))
    user.watched_videos.append(entry)
    video.viewers.add(user.user_id)

def _record_like(user, video, time):
    entry = (video.url, time.strftime('%Y-%m-%d %H:%M:%S'))
    user.liked_videos.append(entry)
    video.like_count += 1

def save_detailed_report(users, videos, filename):
    try:
        wb = Workbook()
        
        # Sheet 1: Category Statistics
        ws_cats = wb.active
        ws_cats.title = "Category Stats"
        ws_cats.append(["Category Name", "Video Count"])
        cat_stats = defaultdict(int)
        for v in videos:
            if v.category:
                cat_stats[v.category] += 1
        for cat, count in sorted(cat_stats.items(), key=lambda x: x[1], reverse=True):
            ws_cats.append([cat, count])
        
        # Sheet 2: Watch History
        ws_watches = wb.create_sheet("Watch History")
        ws_watches.append(["User ID", "Video URL", "Watch Time"])
        for user in users:
            for url, time in user.watched_videos:
                ws_watches.append([user.user_id, url, time])
        
        # Sheet 3: Like History  
        ws_likes = wb.create_sheet("Like History")
        ws_likes.append(["User ID", "Video URL", "Like Time"])
        for user in users:
            for url, time in user.liked_videos:
                ws_likes.append([user.user_id, url, time])
        
        # Sheet 4: Video Stats
        ws_videos = wb.create_sheet("Video Statistics")
        ws_videos.append(["Video URL", "Category", "View Count", "Like Count"])
        for video in videos:
            view_count = sum(1 for u in users if any(url == video.url for url, _ in u.watched_videos))
            ws_videos.append([video.url, video.category, view_count, video.like_count])
        
        wb.save(filename)
    except Exception as e:
        raise IOError(f"Error saving Excel file: {str(e)}")

if __name__ == "__main__":
    try:
        print("Loading video data...")
        videos = load_excel_data("videos.xlsx")
        print(f"Loaded {len(videos)} videos")
        
        print("Categorizing videos using TF-IDF and cosine similarity...")
        cat_names = auto_categorize(videos)
        
        print("Displaying categories:")
        list_all_categories(videos)
        
        print(f"Creating {NUM_USERS} users...")
        users = [User(i, list(cat_names.keys())) for i in range(NUM_USERS)]
        
        print("Simulating user behaviors...")
        simulate_behavior(users, videos)
        
        print(f"Saving report to {OUTPUT_FILENAME}...")
        save_detailed_report(users, videos, OUTPUT_FILENAME)
        
        total_views = sum(len(u.watched_videos) for u in users)
        total_likes = sum(len(u.liked_videos) for u in users)
        
        print("\n=== Simulation Summary ===")
        print(f"Total views: {total_views}")
        print(f"Total likes: {total_likes}") 
        print(f"Like rate: {total_likes/total_views:.1%}")
        print(f"Report saved to {OUTPUT_FILENAME}")
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        print(f"Data Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")