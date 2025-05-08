<template>
  <div class="player-page">
    <!-- 顶部搜索栏 -->
    <div class="search-bar">
      <el-input
        v-model="searchKeyword"
        placeholder="搜索其他视频"
        class="search-input"
        @keyup.enter="handleSearch"
        clearable
      />
      <el-button type="primary" @click="handleSearch">搜索</el-button>

        <el-button
    :icon="House"
    circle
    @click="goHome"
    class="return-home-btn"
    title="返回主页"
  />
    </div>

    <div class="content">
      <!-- 主播放区 -->
          <div class="main-frame">
      <iframe
        v-if="videoPageUrl"
        :src="videoPageUrl"
        class="iframe-player"
        frameborder="0"
        allowfullscreen
      ></iframe>
      <div v-else class="no-video">页面加载失败</div>
      <div class="video-title">{{ videoTitle }}</div>
    </div>

      <!-- 推荐侧栏 -->
      <div class="recommend-sidebar">
        <h3 class="recommend-title">相关推荐</h3>
        <div class="recommend-list">
          <div
            v-for="(video, idx) in paginatedRecommend"
            :key="video.id"
            class="recommend-item"
            @click="playAnother(video)"
          >
            <img :src="video.cover" alt="封面" class="recommend-cover" />
            <div class="recommend-info">
              <div class="recommend-title-text">{{ video.title }}</div>
            </div>

            <!-- 分割线（除了最后一个） -->
            <div
              v-if="idx !== paginatedRecommend.length - 1"
              class="divider-line"
            ></div>
          </div>
        </div>

        <el-pagination
          background
          layout="prev, pager, next"
          :total="recommendList.length"
          :page-size="pageSize"
          v-model:current-page="currentPage"
          class="pagination"
        />
      </div>
    </div>
  </div>
</template>


<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '@/utils/api'
import { House } from '@element-plus/icons-vue'

function goHome() {
  router.push('/home')
}


const route = useRoute()
const router = useRouter()

const id = ref(route.params.id)
const videoPageUrl = ref('')
const recommendList = ref([])

const searchKeyword = ref('')

const currentPage = ref(1)
const pageSize = 6

const videoTitle = ref('')

const paginatedRecommend = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return recommendList.value.slice(start, start + pageSize)
})

async function loadVideo() {
  const res = await api.get(`/video/${id.value}`)
  videoPageUrl.value = res.data.video_url || ''  // 🔥 指定网页地址
  videoTitle.value = res.data.title || '视频加载失败'
}

async function loadRecommend() {
  try {
    const res = await api.get(`/video/${id.value}/recommend`)
    console.log("res.data:", res.data)
    recommendList.value = res.data.similar_videos || []
    currentPage.value = 1
  } catch (error) {
    console.error('加载推荐失败', error)
  }
}

function playAnother(video) {
  router.push(`/player/${video.id}`)
}

function handleSearch() {
  if (!searchKeyword.value.trim()) return
  router.push(`/videos?keyword=${encodeURIComponent(searchKeyword.value)}`)
}

onMounted(() => {
  loadVideo()
  loadRecommend()
})

watch(() => route.params.id, (newId) => {
  id.value = newId
  loadVideo()
  loadRecommend()
})
</script>


<style scoped>
.player-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #eef3f8;
}

.search-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 24px;
  background-color: #fff;
  border-bottom: 1px solid #e5e5e5;
}

.search-input {
  width: 300px; /* ✅ 设置固定宽度，与首页统一 */
}


.content {
  display: flex;
  padding: 24px;
  gap: 24px;
  box-sizing: border-box;
}

/* 主播放器区域 */
.main-frame {
  flex: none; /* 防止拉伸 */
  width: 75%; /* 固定宽度为 75% 或者直接写成例如 900px */
  max-width: 1000px;
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}
.iframe-player {
  width: 100%;
  height: 500px;
  border-radius: 8px;
  border: none;
  background: #000;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.video-title {
  margin-top: 16px;
  font-size: 18px;
  font-weight: 600;
  color: #333;
  text-align: center;
}

/* 推荐视频栏 */
.recommend-sidebar {
  flex: none;
  width: 25%; /* 剩下的25% */
  min-width: 260px; /* 设置最小宽度防止太窄 */
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
}

.recommend-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 16px;
  color: #333;
}

.recommend-list {
  flex: 1;
  overflow-y: auto;
}

.recommend-item {
  display: flex;
  align-items: center;
  margin-bottom: 14px;
  cursor: pointer;
  padding: 4px;
  border-radius: 6px;
  transition: background 0.2s;
}

.recommend-item:hover {
  background: #f9f9f9;
}

.recommend-cover {
  width: 80px;
  height: 50px;
  border-radius: 6px;
  object-fit: cover;
  margin-right: 10px;
}

.recommend-title-text {
  font-size: 14px;
  color: #333;
  font-weight: 500;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.divider-line {
  height: 1px;
  background: #eee;
  margin: 8px 0;
}

.pagination {
  margin-top: 12px;
  text-align: center;
}

.return-home-btn {
  box-shadow: 0 2px 6px rgba(64, 158, 255, 0.2);
}

</style>
