import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', component: () => import('@/pages/LoginView.vue') },
  { path: '/home', component: () => import('@/pages/HomeView.vue') }, // ⭐ 新增
  { path: '/admin', component: () => import('@/pages/AdminDashboard.vue') },
  { path: '/videos', component: () => import('@/pages/VideoListView.vue') },
  { path: '/player/:id', component: () => import('@/pages/VideoPlayerView.vue') },
  { path: '/user/:id', component: () => import('@/pages/UserProfileView.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
