<template>
  <div>
    <h2>视频详情页 ID: {{ id }}</h2>
    <el-button @click="playVideo">播放视频</el-button>
  </div>
</template>

<script setup>
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const id = route.params.id

function playVideo() {
  router.push(`/player/${id}`)
}
</script>
