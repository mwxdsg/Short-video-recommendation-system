import time
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD # <--- Added for dimensionality reduction
from scipy.sparse import lil_matrix, csr_matrix # <--- Added for sparse matrices
from collections import defaultdict, Counter
from tqdm import tqdm
import gc
import traceback

def cluster_videos_by_viewers_balanced_optimized(
    videos,
    users,
    n_clusters=100,
    batch_size=5000,
    max_size_factor=1.8,
    split_k=2,
    svd_components=150, # <-- New parameter: Number of components for SVD (0 to disable)
    mbk_n_init=5       # <-- New parameter: n_init for MiniBatchKMeans
):
    """
    Optimized video clustering based on viewer similarity using sparse matrices,
    optional SVD, and cluster balancing. Handles larger datasets.

    Args:
        videos (list): List of Video objects.
        users (list): List of User objects.
        n_clusters (int): Initial target number of clusters (forced to 100 internally).
        batch_size (int): Batch size for MiniBatchKMeans.
        max_size_factor (float): Threshold factor to define "large" clusters for splitting.
        split_k (int): Number of sub-clusters to split large clusters into.
        svd_components (int): Number of dimensions for TruncatedSVD reduction.
                              Set to 0 or None to disable SVD.
        mbk_n_init (int): The 'n_init' parameter for MiniBatchKMeans.

    Returns:
        pd.DataFrame: DataFrame with clustering results, or empty DataFrame on error.
    """
    print(f"\n开始优化版视频聚类 (目标: {n_clusters} 个聚类，尝试平衡大小)...")
    print(f"  使用稀疏矩阵，SVD={svd_components if svd_components else '禁用'}, 平衡因子={max_size_factor}")
    start_time = time.time()

    # --- Parameter Enforcement & Validation ---
    target_n_clusters = 100
    if n_clusters != target_n_clusters:
        print(f"  注意：函数被调用时 n_clusters={n_clusters}，但内部强制设置为 {target_n_clusters}。")

    if not videos or not users:
        print("错误：视频列表或用户列表为空。")
        return pd.DataFrame()

    # --- Data Preparation ---
    try:
        user_ids = [u.user_id for u in users]
        n_users = len(user_ids)
        valid_videos = [v for v in videos if v.url and v.url != "N/A"]
        if not valid_videos:
            print("错误：没有找到有效的视频URL。")
            return pd.DataFrame()

        video_urls = [v.url for v in valid_videos]
        n_valid_videos = len(video_urls)
        print(f"  准备处理 {n_valid_videos} 个有效视频 和 {n_users} 个用户。")

        initial_actual_clusters = min(target_n_clusters, n_valid_videos)
        if initial_actual_clusters < target_n_clusters:
            print(f"  警告：有效视频数 ({n_valid_videos}) 少于目标聚类数 ({target_n_clusters})。初始聚类数减为 {initial_actual_clusters}。")
        if initial_actual_clusters <= 1:
            print(f"  警告：有效视频数 ({n_valid_videos}) 过少，无法进行有意义的聚类。")
            # Allow processing to continue if 1 cluster is possible, maybe return single cluster result
            # If initial_actual_clusters is 0, need to handle below.

        # --- Build Sparse User-Video Matrix ---
        print(f"构建稀疏用户-视频观看矩阵 ({n_users} 用户 x {n_valid_videos} 视频)...")
        # Use lil_matrix for efficient incremental construction
        user_video_matrix_sparse = lil_matrix((n_users, n_valid_videos), dtype=np.float32)
        user_index = {uid: i for i, uid in enumerate(user_ids)}
        video_index = {url: i for i, url in enumerate(video_urls)}
        watched_video_count = 0

        for user in tqdm(users, desc="填充稀疏矩阵"):
            if user.user_id in user_index:
                u_idx = user_index[user.user_id]
                for url, _ in user.watched_videos:
                    if url in video_index:
                        v_idx = video_index[url]
                        user_video_matrix_sparse[u_idx, v_idx] = 1.0
                        watched_video_count += 1

        if watched_video_count == 0:
            print("错误：没有用户观看任何有效视频的记录。")
            return pd.DataFrame()

        # Convert to CSR format for efficient calculations and transpose
        print("  转换为 CSR 格式并转置...")
        user_video_matrix_sparse = user_video_matrix_sparse.tocsr()
        video_user_matrix_sparse = user_video_matrix_sparse.T.tocsr()
        del user_video_matrix_sparse; gc.collect()
        print(f"  稀疏矩阵构建完成。密度: {video_user_matrix_sparse.nnz / (n_valid_videos * n_users):.4%}")

        # --- Calculate Actual Viewer Counts (Efficiently from Sparse Matrix) ---
        print("计算每个视频的实际观看用户数...")
        # video_user_matrix_sparse shape is (n_videos, n_users)
        # .nnz on a row gives the number of non-zero elements (users who watched)
        actual_viewer_counts = [video_user_matrix_sparse.getrow(i).nnz
                                for i in tqdm(range(n_valid_videos), desc="计算观看数")]

        # --- Dimensionality Reduction (Optional) ---
        if svd_components and svd_components > 0 and svd_components < n_users and svd_components < n_valid_videos:
            print(f"执行 TruncatedSVD 降维到 {svd_components} 个组件...")
            svd = TruncatedSVD(n_components=svd_components, random_state=42)
            try:
                X_reduced = svd.fit_transform(video_user_matrix_sparse)
                print(f"  降维后数据形态: {X_reduced.shape}")
                # Explained variance (optional check)
                # print(f"  SVD 解释方差比: {svd.explained_variance_ratio_.sum():.2%}")
                X_input_for_scaling = X_reduced
                del video_user_matrix_sparse; gc.collect()
            except Exception as svd_err:
                 print(f"错误：SVD降维失败 - {svd_err}。将尝试不使用SVD。")
                 X_input_for_scaling = video_user_matrix_sparse # Fallback to original sparse
                 svd_components = 0 # Mark SVD as disabled
        else:
            print("跳过 SVD 降维。")
            X_input_for_scaling = video_user_matrix_sparse # Use original sparse matrix

        # --- Feature Scaling ---
        # StandardScaler works with sparse matrices if with_mean=False
        print("缩放特征...")
        scaler = StandardScaler(with_mean=False)
        try:
            # Fit and transform the (potentially reduced) data
            X_scaled = scaler.fit_transform(X_input_for_scaling)
            print(f"  缩放后数据形态: {X_scaled.shape}, 类型: {'稀疏' if isinstance(X_scaled, csr_matrix) else '稠密'}")
            del X_input_for_scaling; gc.collect()
        except ValueError as e:
            print(f"错误: 特征缩放失败 - {e}")
            return pd.DataFrame()
        except MemoryError:
             print("错误: 特征缩放时内存不足！")
             return pd.DataFrame()


        # --- Initial MiniBatchKMeans Clustering ---
        if initial_actual_clusters <= 0:
            print("错误：无法计算有效的初始聚类数。")
            return pd.DataFrame()

        print(f"执行初始 MiniBatchKMeans 聚类 (k={initial_actual_clusters}, n_init={mbk_n_init})...")
        effective_batch_size = min(batch_size, X_scaled.shape[0]) # Ensure batch size <= num samples
        if effective_batch_size <= 0:
            print(f"错误：计算出的有效batch_size ({effective_batch_size}) 无效。")
            return pd.DataFrame()

        kmeans_initial = MiniBatchKMeans(
            n_clusters=initial_actual_clusters,
            batch_size=effective_batch_size,
            random_state=42,
            n_init=mbk_n_init, # Use parameter
            max_iter=300,
            init='k-means++' # k-means++ can be slow on very large sparse data, consider 'random' if needed
        )
        try:
             initial_clusters = kmeans_initial.fit_predict(X_scaled)
             print("初始聚类完成。")
        except MemoryError:
             print("错误: 初始聚类时内存不足！")
             return pd.DataFrame()
        except Exception as km_err:
             print(f"错误: 初始聚类失败 - {km_err}")
             traceback.print_exc()
             return pd.DataFrame()

        # --- Post-processing: Split Large Clusters ---
        print("后处理：检查并拆分大簇...")
        final_clusters = initial_clusters.copy()
        current_cluster_id_max = initial_actual_clusters - 1
        iteration = 0
        max_iterations = 5 # Prevent infinite loops

        while iteration < max_iterations:
            iteration += 1
            print(f"  拆分迭代 {iteration}...")
            cluster_sizes = Counter(final_clusters)
            # Use the actual number of current clusters for average calculation
            num_current_clusters = len(cluster_sizes)
            if num_current_clusters == 0:
                print("  错误：没有找到任何簇，无法进行拆分。")
                break
            avg_size = n_valid_videos / num_current_clusters
            max_allowed_size = int(avg_size * max_size_factor)
            print(f"    当前簇数: {num_current_clusters}, 平均簇大小: {avg_size:.2f}, 最大允许大小: {max_allowed_size}")

            clusters_to_split = sorted([cid for cid, size in cluster_sizes.items() if size > max_allowed_size])

            if not clusters_to_split:
                print("  没有需要拆分的大簇了。")
                break

            print(f"    找到 {len(clusters_to_split)} 个大簇需要拆分 (ID示例: {clusters_to_split[:5]}...)")

            # Store new assignments temporarily to avoid modifying `final_clusters` during iteration
            new_assignments = {} # {index: new_cluster_id}

            for cluster_id_to_split in clusters_to_split:
                indices_to_split = np.where(final_clusters == cluster_id_to_split)[0]

                if len(indices_to_split) <= max_allowed_size or len(indices_to_split) < split_k:
                    continue # Skip if already shrunk or too small to split

                print(f"    拆分簇 {cluster_id_to_split} (大小: {len(indices_to_split)})...")
                # Select the *scaled* data for the sub-cluster
                data_to_split = X_scaled[indices_to_split]

                # Ensure sub-cluster batch size is valid
                sub_batch_size = min(batch_size, data_to_split.shape[0])
                if sub_batch_size <= 0:
                     print(f"     警告：跳过簇 {cluster_id_to_split}，无法计算有效的子聚类batch_size ({sub_batch_size})")
                     continue

                sub_kmeans = MiniBatchKMeans(
                    n_clusters=split_k,
                    batch_size=sub_batch_size,
                    random_state=42 + cluster_id_to_split,
                    n_init=max(1, mbk_n_init // 2), # Fewer inits for sub-clustering
                    max_iter=150
                )
                try:
                    sub_labels = sub_kmeans.fit_predict(data_to_split)
                except ValueError:
                    print(f"     警告：无法拆分簇 {cluster_id_to_split} (子聚类失败)")
                    continue
                except MemoryError:
                    print(f"     警告：拆分簇 {cluster_id_to_split} 时内存不足！")
                    continue # Skip this cluster


                # Assign new cluster IDs
                new_cluster_id_start = current_cluster_id_max + 1
                split_success = False
                for sub_label_idx in range(split_k):
                    # Get the original indices corresponding to this sub-label
                    sub_indices = indices_to_split[sub_labels == sub_label_idx]
                    if len(sub_indices) == 0: continue # Skip empty sub-clusters

                    if sub_label_idx == 0:
                        # Keep original ID for the first part
                        # No change needed in final_clusters for these indices yet
                        pass
                    else:
                        # Assign new ID to the other parts
                        assigned_new_id = new_cluster_id_start
                        # Update the temporary assignment dict
                        for idx in sub_indices:
                            new_assignments[idx] = assigned_new_id
                        current_cluster_id_max += 1
                        new_cluster_id_start += 1
                        split_success = True

                # If split happened, print new max ID
                # if split_success:
                    # print(f"      簇 {cluster_id_to_split} 部分拆分，新最大簇ID: {current_cluster_id_max}")


            # Apply the new assignments after iterating through all clusters_to_split
            if new_assignments:
                print(f"    应用 {len(new_assignments)} 个新簇分配...")
                for idx, new_cid in new_assignments.items():
                    final_clusters[idx] = new_cid
            else:
                print("    本次迭代没有成功拆分并重新分配簇。")


            print(f"  迭代 {iteration} 结束，当前总簇数: {len(np.unique(final_clusters))}")
            # Optional: Check if cluster counts changed significantly, maybe break early


        if iteration == max_iterations:
            print("  达到了最大拆分迭代次数。")

        # --- Build Final Results DataFrame ---
        print("整理最终聚类结果...")
        final_cluster_map = {cid: i for i, cid in enumerate(np.unique(final_clusters))}
        remapped_final_clusters = np.array([final_cluster_map[cid] for cid in final_clusters])
        final_cluster_sizes = Counter(remapped_final_clusters) # Use remapped IDs

        results = []
        for i, url in enumerate(tqdm(video_urls, desc="构建结果")):
            cluster_id = int(remapped_final_clusters[i])
            results.append({
                "视频URL": url,
                "聚类ID": cluster_id,
                # "聚类内视频数": final_cluster_sizes.get(cluster_id, 0), # Recalculate below if needed
                "观看该视频的用户数": int(actual_viewer_counts[i]) # Use pre-calculated actual counts
            })

        results_df = pd.DataFrame(results)

        # Add accurate cluster size after creating the DataFrame
        print("添加最终簇大小列...")
        cluster_counts = results_df['聚类ID'].map(results_df['聚类ID'].value_counts())
        results_df['聚类内视频数'] = cluster_counts

        end_time = time.time()
        final_num_clusters = len(final_cluster_sizes)
        print(f"\n优化版平衡视频聚类完成。")
        print(f"  共生成 {final_num_clusters} 个最终聚类。")
        print(f"  耗时: {end_time - start_time:.2f} 秒。")

        if final_num_clusters > 0:
            sizes = list(results_df['聚类内视频数'].unique()) # Get actual sizes from the column
            if sizes:
                print(f"  最终簇大小统计: 最小={min(sizes)}, 最大={max(sizes)}, 平均={np.mean(sizes):.2f}, 标准差={np.std(sizes):.2f}")
            else:
                 print("  无法计算最终簇大小统计 (无有效大小)。")
        else:
             print("  未生成任何聚类。")

        return results_df

    except MemoryError:
        print("\n错误：优化版平衡聚类过程中发生内存不足错误！")
        gc.collect()
        return pd.DataFrame()
    except Exception as e:
        print(f"\n错误：执行优化版平衡视频聚类时发生意外错误: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

# --- How to use in the main script ---
# Replace the call in Step 7 (F6) with:
# video_clusters = video_clustering.cluster_videos_by_viewers_balanced_optimized(
#     videos,
#     users,
#     n_clusters=100,       # Still forced to 100 internally for initial step
#     max_size_factor=1.8,  # Adjust balance strictness (lower is stricter)
#     split_k=2,            # How many parts to split large clusters into
#     svd_components=150,   # <--- Tune SVD dimensions (e.g., 100-300), or 0 to disable
#     mbk_n_init=5          # <--- Adjust MiniBatchKMeans n_init
# )
# Make sure this function definition is in your 'video_clustering.py' file.
# Remember to use a distinct output filename, e.g., "video_clusters_optimized.xlsx"
