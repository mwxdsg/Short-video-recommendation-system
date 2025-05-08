<template>

  <div class="admin-dashboard">
      <!-- 新增标题行 -->
  <h2 class="video-title">{{ videoTitle }}</h2>
    <!-- 顶部输入 -->
    <div class="input-bar">
      <el-input
        v-model="videoLink"
        placeholder="请输入视频序号"
        class="search-input"
        @keyup.enter="searchVideo"
      />
      <el-button type="primary" @click="searchVideo">查询</el-button>
    </div>

    <!-- 视频信息展示 -->
    <div v-if="chartData.length" class="data-panel">
      <h2 class="panel-title">热度变化曲线</h2>
      <div id="heat-chart" class="chart-container"></div>

      <h2 class="panel-title" style="margin-top: 32px;">相关关键词</h2>
      <el-tag
        v-for="(word, index) in keywords"
        :key="index"
        type="info"
        class="keyword-tag"
      >
        {{ word }}
      </el-tag>
    </div>
<div style="display: flex; justify-content: space-between; align-items: center; margin: 24px 0 16px;">
  <el-button type="danger" size="small" @click="logout" style="margin-top: 8px;">退出登录</el-button>
</div>


  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import * as echarts from 'echarts'
import api from '@/utils/api'

const videoLink = ref('')
const chartData = ref([])  // 时间-播放量数组
const keywords = ref([])   // 关键词数组
const videoTitle = ref('')

import { useRouter } from 'vue-router'
const router = useRouter()

function logout() {
  localStorage.removeItem('token') // 或 sessionStorage.clear()
  router.push('/login') // 跳转到登录页路径
}


async function searchVideo() {
  if (!videoLink.value.trim()) return
  if (!/^\d+$/.test(videoLink.value.trim())) {
    alert('账号只能为数字')
    return
  }

  try {
    const res = await api.get('/admin/video-analysis', {
      params: { link: videoLink.value }
    })
    console.log("后端返回的数据是：", res.data)
    videoTitle.value = res.data.title || ''
    chartData.value = res.data.heatCurve || []
    keywords.value = res.data.keywords || []
    nextTick(() => {
      renderChart()
    })
  } catch (error) {
    console.error('查询失败', error)
  }
}

function renderChart() {
  const chartDom = document.getElementById('heat-chart')
  if (!chartDom) return

  const myChart = echarts.init(chartDom)
  myChart.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: chartData.value.map(item => item.time),
      boundaryGap: false,
    },
    yAxis: {
      type: 'value',
    },
    series: [
      {
        name: '播放量',
        type: 'line',
        smooth: true,
        data: chartData.value.map(item => item.views),
        areaStyle: {},
        lineStyle: { width: 2 },
      }
    ]
  })
}
</script>

<style scoped>
.admin-dashboard {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.input-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}

.search-input {
  width: 400px;
}

.data-panel {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.panel-title {
  font-size: 20px;
  font-weight: bold;
  color: #333;
  margin-bottom: 16px;
}

.chart-container {
  width: 100%;
  height: 400px;
}

.keyword-tag {
  margin: 6px;
}

.video-title {
  font-size: 22px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 12px;
  line-height: 1.3;
}
</style>
