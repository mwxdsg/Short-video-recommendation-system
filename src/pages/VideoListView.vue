<template>

  <div class="search-page">
    <!-- 顶部搜索栏 -->
    <div class="search-bar">
      <el-input
        v-model="keyword"
        placeholder="输入关键词搜索视频"
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

    <!-- 搜索提示 -->
    <div class="search-info" v-if="keyword">
      <span class="info-label">当前搜索：</span><span class="info-highlight">{{ keyword }}</span>
    </div>

    <!-- 视频结果列表 -->
    <el-row :gutter="24" class="video-list">
      <el-col :span="8" v-for="video in paginatedResult" :key="video.id">
        <el-card shadow="always" class="video-card" @click="viewVideo(video.id)">
          <img :src="video.cover" alt="封面" class="video-cover" />
          <div class="video-title">{{ video.title }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 翻页器 -->
    <el-pagination
      background
      layout="prev, pager, next"
      :total="resultList.length"
      :page-size="pageSize"
      v-model:current-page="currentPage"
      class="pagination"
    />
  </div>
</template>

<script setup>
import {
  ref, computed, onMounted, watch
} from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '@/utils/api'
import { House } from '@element-plus/icons-vue'

function goHome() {
  router.push('/home')
}

const router = useRouter()
const route = useRoute()

const keyword = ref('')
const resultList = ref([])

const pageSize = 9
const currentPage = ref(1)

const paginatedResult = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return resultList.value.slice(start, start + pageSize)
})

async function searchVideos() {
  if (!keyword.value.trim()) return

  try {
    const res = await api.get('/search', {
      params: { keyword: keyword.value }
    })
    resultList.value = res.data || []
    currentPage.value = 1
  } catch (error) {
    console.error('搜索失败', error)
  }
}

function handleSearch() {
  router.push({ path: '/videos', query: { keyword: keyword.value } }) // 自动更新URL
}


function viewVideo(id) {
  router.push(`/player/${id}`)
}

onMounted(() => {
  if (route.query.keyword) {
    keyword.value = route.query.keyword
    searchVideos()
  }
})

// ✅ 添加这个 watch，监听路由参数变化
watch(() => route.query.keyword, (newKeyword) => {
  if (newKeyword) {
    keyword.value = newKeyword
    searchVideos()
  }
})

</script>
<style scoped>
.search-page {
  padding: 40px 60px;
  background: #f5f7fa;
  min-height: 100vh;
}

.search-bar {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin-bottom: 30px;
}

.search-input {
  width: 400px;
}

.search-info {
  text-align: center;
  margin-bottom: 30px;
  font-size: 18px;
  color: #666;
}

.info-label {
  font-weight: normal;
}

.info-highlight {
  font-weight: bold;
  color: #409eff;
}

.video-list {
  margin-top: 10px;
}

.video-card {
  cursor: pointer;
  padding: 12px;
  border-radius: 12px;
  overflow: hidden;
  transition: all 0.3s;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.video-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
  background: #f0f2f5;
}

.video-cover {
  width: 100%;
  height: 180px;
  object-fit: cover;
  border-radius: 10px;
  margin-bottom: 10px;
}

.video-title {
  font-size: 16px;
  font-weight: 600;
  text-align: center;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pagination {
  margin: 40px auto 0 auto; /* 上40px间距，水平居中 */
  text-align: center;
  display: flex;
  justify-content: center;
}

.return-home-btn {
  box-shadow: 0 2px 6px rgba(64, 158, 255, 0.2);
}

</style>


