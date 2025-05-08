import axios from 'axios'

const instance = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

instance.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  },
)

export default instance
