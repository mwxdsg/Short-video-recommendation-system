<template>
  <div class="login-page">
    <div class="login-card">
      <h2 class="login-title">
  欢迎登录 <span class="cursive-text">Mann</span>
</h2>

      <el-form @submit.prevent="handleLogin" class="login-form">
        <el-form-item>
          <el-input
            v-model="form.id"
            placeholder="请输入用户ID"
            size="large"
          />
        </el-form-item>

        <el-form-item>
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            size="large"
          />
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            size="large"
            class="login-button"
            @click="handleLogin"
            style="width: 100%;"
          >
            登录
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const form = reactive({
  id: '',
  password: '',
})

function handleLogin() {
  if (!form.id) {
    alert('请输入账号')
    return
  }
  if (!((/^\d+$/.test(form.id))|| form.id === 'admin')) {
    alert('账号只能为数字')
    return
  }
  if (!form.password) {
    alert('请输入密码')
    return
  }

  // 登录成功后记录 userId
  localStorage.setItem('userId', form.id)

  if (form.id === 'admin') {
    router.push('/admin')
  } else {
    router.push('/home')  // 普通用户跳到首页
  }
}

</script>

<style scoped>
.login-page {
  width: 100%;
  height: 100vh;
  background: linear-gradient(135deg, #74ebd5, #acb6e5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.login-card {
  width: 400px;
  background: white;
  border-radius: 16px;
  padding: 40px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.login-title {
  font-size: 28px;
  font-weight: bold;
  color: #333;
  margin-bottom: 30px;
}

.login-form {
  width: 100%;
}

.login-button {
  margin-top: 10px;
}

.cursive-text {
  font-style: italic;
}
</style>

