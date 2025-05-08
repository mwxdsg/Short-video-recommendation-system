

<template>
  <div class="user-profile-page">
    <div class="content-left">
      <div class="user-header">
  <img :src="userAvatar" alt="头像" class="header-avatar" />
  <div class="header-info">
    <h2 class="page-title">用户中心</h2>
    <p class="user-id">ID: {{ userId }}</p>
  </div>
<el-button
  @click="goHome"
  :icon="House"
  circle
  class="return-home-btn"
/>

</div>

      <!-- 浏览记录 -->
      <el-card class="section-card">
        <h3 class="section-title">浏览记录</h3>
        <div v-for="video in paginatedHistory" :key="video.id" class="video-item" @click="goPlay(video.id)">
          <img :src="video.cover" alt="封面" class="video-cover" />
          <div class="video-info">
            <div class="video-title">{{ video.title }}</div>
          </div>
        </div>

        <el-pagination
          background
          layout="prev, pager, next"
          :total="historyList.length"
          :page-size="pageSize"
          v-model:current-page="currentHistoryPage"
          class="pagination"
        />
      </el-card>

      <!-- 点赞记录 -->
      <el-card class="section-card" style="margin-top: 24px;">
        <h3 class="section-title">点赞记录</h3>
        <div v-for="video in paginatedLike" :key="video.id" class="video-item" @click="goPlay(video.id)">
          <img :src="video.cover" alt="封面" class="video-cover" />
          <div class="video-info">
            <div class="video-title">{{ video.title }}</div>
          </div>
        </div>

        <el-pagination
          background
          layout="prev, pager, next"
          :total="likeList.length"
          :page-size="pageSize"
          v-model:current-page="currentLikePage"
          class="pagination"
        />
      </el-card>
    </div>

    <!-- 相似用户栏 -->
    <div class="content-right">
  <el-card class="similar-users-card">
    <h3 class="section-title">相似用户</h3>
    <div
      v-for="user in paginatedSimilarUsers"
      :key="user.id"
      class="user-item"
      @click="goUserProfile(user.id)"
    >
      <div class="user-avatar-wrapper">
        <img :src="user.avatar" alt="头像" class="user-avatar" />
      </div>
      <div class="user-name">{{ user.name }}</div>
    </div>

    <!-- 分页器 -->
    <el-pagination
      background
      layout="prev, pager, next"
      :total="similarUsers.length"
      :page-size="userPageSize"
      v-model:current-page="currentUserPage"
      class="pagination"
    />
  </el-card>
</div>

  </div>
</template>

<script setup>
import { House } from '@element-plus/icons-vue'
import { ref, computed, onMounted, watch} from 'vue'
import { useRouter, useRoute } from 'vue-router'
import api from '@/utils/api'

const router = useRouter()
const route = useRoute()
const userId = computed(() => route.params.id)



const userAvatar = ref(`https://i.pravatar.cc/100?img=${Math.floor(Math.random() * 70)}`)

const historyList = ref([])
const likeList = ref([])
const similarUsers = ref([])

const pageSize = 5
const userPageSize = 4
const currentHistoryPage = ref(1)
const currentLikePage = ref(1)
const currentUserPage = ref(1)

const paginatedHistory = computed(() => {
  const start = (currentHistoryPage.value - 1) * pageSize
  return historyList.value.slice(start, start + pageSize)
})

const paginatedLike = computed(() => {
  const start = (currentLikePage.value - 1) * pageSize
  return likeList.value.slice(start, start + pageSize)
})

const paginatedSimilarUsers = computed(() => {
  const start = (currentUserPage.value - 1) * userPageSize
  return similarUsers.value.slice(start, start + userPageSize)
})

async function loadHistory() {
  try {
    const res = await api.get(`/user/${userId.value}/history`)
    historyList.value = res.data || []
  } catch (error) {
    console.error('加载浏览记录失败', error)
  }
}

async function loadLikes() {
  try {
    const res = await api.get(`/user/${userId.value}/likes`)
    likeList.value = res.data || []
  } catch (error) {
    console.error('加载点赞记录失败', error)
  }
}

async function loadSimilarUsers() {
  try {
    const res = await api.get(`/user/${userId.value}/similar`)
    console.log("res.data:", res.data)

    similarUsers.value = res.data.map(u => ({
      ...u,
      name: u.id,
      avatar: `https://i.pravatar.cc/100?img=${Math.floor(Math.random() * 70)}`
    })) || []
  } catch (error) {
    console.error('加载相似用户失败', error)
  }
}

async function goPlay(url) {
  try {
    // 发送 GET 请求获取视频信息
    const res = await api.get(`/video/${url}`);

    // 后端返回的数据格式应该是 { id: 123 }
    const video_id = res.data.id;

    if (!video_id) {
      console.error('未获取到视频 ID');
      return;
    }

    // 跳转到播放页面
    router.push(`/player/${video_id}`);
  } catch (error) {
    console.error('获取视频信息失败:', error);
  }
}

function goUserProfile(id) {
  // 提取数字部分，例如从 "U0028" 得到 "28"
  const numericId = id.replace(/^\D+/, ''); // 去掉前缀的非数字字符
  router.push(`/user/${numericId}`);
}

function goHome() {
  router.push('/home')
}

onMounted(() => {
  loadHistory()
  loadLikes()
  loadSimilarUsers()
})

watch(() => route.params.id, (newId, oldId) => {
  if (newId !== oldId) {
    loadHistory()
    loadLikes()
    loadSimilarUsers()

  }
})
</script>

<style scoped>
.user-profile-page {
  display: flex;
  gap: 24px;
  padding: 24px 48px;
  background: #f5f7fa;
  min-height: 100vh;
  box-sizing: border-box;
  width: 100%;
  overflow: hidden;
}

.content-left {
  flex: none;
  width: 72%; /* 主体区域占比 */
  min-width: 800px;
}

.content-right {
  flex: none;
  width: 28%; /* 右栏占比 */
  min-width: 280px;
}

.user-header {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 16px;
  margin-bottom: 24px;
}

.header-avatar {
  width: 60px;
  height: 60px;
  border-radius: 50%;
}

.header-info {
  display: flex;
  flex-direction: column;
  margin-right: auto; /* 推开右侧按钮 */
}

.page-title {
  font-size: 24px;
  font-weight: bold;
  color: #333;
}

.user-id {
  font-size: 14px;
  color: #999;
  margin-top: 4px;
}

.section-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.similar-users-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  position: sticky;
  top: 24px;
}

.section-title {
  font-size: 22px;
  font-weight: bold;
  margin-bottom: 16px;
  color: #333;
}

.video-item {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding: 8px;
  border-radius: 8px;
  transition: background 0.2s;
  cursor: pointer;
}

.video-item:hover {
  background: #f0f2f5;
}

.video-cover {
  width: 100px;
  height: 60px;
  object-fit: cover;
  border-radius: 8px;
  margin-right: 12px;
}

.video-info {
  flex: 1;
}

.video-title {
  font-size: 16px;
  font-weight: 500;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-item {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding: 8px;
  border-radius: 8px;
  transition: background 0.2s;
  cursor: pointer;
}

.user-item:hover {
  background: #f0f2f5;
}

.user-avatar {
  width: 48px;
  height: 48px;
  object-fit: cover;
  border-radius: 50%;
  margin-right: 12px;
}

.user-name {
  font-size: 16px;
  color: #333;
  font-weight: 500;
}

.pagination {
  margin-top: 20px;
  text-align: center;
}

/* 相似用户区域美化 */
.similar-users-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  position: sticky;
  top: 24px;
}

.user-item {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
  padding: 8px;
  border-radius: 10px;
  transition: all 0.3s;
  cursor: pointer;
}

.user-item:hover {
  background: #f0f2f5;
  transform: translateX(5px);
}

.user-avatar-wrapper {
  width: 52px;
  height: 52px;
  overflow: hidden;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  margin-right: 12px;
  transition: all 0.3s;
}

.user-avatar {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s;
}

.user-item:hover .user-avatar {
  transform: scale(1.1);
}

.user-name {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  transition: color 0.2s;
}

.user-item:hover .user-name {
  color: #409eff;
}

.pagination {
  margin-top: 20px;
  text-align: center;
}

.return-home-btn {
  min-width: auto;
  padding: 4px 10px;
  font-size: 13px;
}
</style>
