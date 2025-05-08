<template>


  <div class="home-container">
    <!-- 顶部操作栏 -->
    <div class="top-bar">
      <el-input
        v-model="searchKeyword"
        placeholder="搜索视频"
        class="search-input"
      />
      <el-button type="primary" @click="search">搜索</el-button>
      <el-button @click="refresh">刷新推荐</el-button>
      <el-button @click="goProfile">个人中心</el-button>
      <!-- 靠右 -->
      <div class="user-info">

        <span v-if="username">Hi，{{ username }}</span>
        <el-button type="danger" size="small" @click="logout">退出登录</el-button>
      </div>
    </div>

    <!-- 主内容区：视频推荐 + 排行榜 -->
    <el-row :gutter="24" class="main-content">
      <!-- 左侧：推荐视频 -->
      <el-col :span="16">
        <!-- 推荐视频区 -->
    <el-card class="section-card">
      <h2 class="section-title">精选推荐</h2>

      <el-row :gutter="20" class="recommend-list">
        <el-col :span="12" v-for="video in recommendVideos" :key="video.id">
          <el-card shadow="hover" class="video-card" @click="viewVideo(video.id)">
            <img :src="video.cover" alt="封面" class="video-cover" />
            <div class="video-info">
              <div class="video-title">{{ video.title }}</div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </el-card>
      </el-col>

      <!-- 右侧：喜爱榜和观看榜 -->
      <el-col :span="8">
        <!-- 喜爱榜 -->
<el-card class="section-card">
  <h2 class="section-title">喜爱榜</h2>
  <div
    v-for="(video, index) in paginatedLikeRank"
    :key="video.id"
    class="rank-item"
    @click="viewVideo(video.id)"
  >
    <div class="rank-badge">{{ getTopBadge(index) }}</div>
    <img :src="video.cover" alt="封面" class="rank-cover" />
    <div class="rank-info">
      <div class="rank-title">{{ video.title }}</div>
      <div class="rank-sub">👍 {{ video.likes }} 次点赞</div>
    </div>
  </div>
  <el-pagination
    background
    layout="prev, pager, next"
    :total="likeRank.length"
    :page-size="pageSize"
    v-model:current-page="currentLikePage"
    class="pagination"
  />
</el-card>

<!-- 观看榜 -->
<el-card class="section-card" style="margin-top: 24px;">
  <h2 class="section-title">观看榜</h2>
  <div
    v-for="(video, index) in paginatedViewRank"
    :key="video.id"
    class="rank-item"
    @click="viewVideo(video.id)"
  >
    <div class="rank-badge">{{ getTopBadge(index) }}</div>
    <img :src="video.cover" alt="封面" class="rank-cover" />
    <div class="rank-info">
      <div class="rank-title">{{ video.title }}</div>
      <div class="rank-sub">👀 {{ video.views }} 次观看</div>
    </div>
  </div>
  <el-pagination
    background
    layout="prev, pager, next"
    :total="viewRank.length"
    :page-size="pageSize"
    v-model:current-page="currentViewPage"
    class="pagination"
  />
</el-card>

      </el-col>
    </el-row>
  </div>
</template>


<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/utils/api'

const router = useRouter()

const username = ref('') // Add this line to define the username variable
const searchKeyword = ref('')
const allVideos = ref([]) // 所有视频数据
const recommendVideos = ref([])

// 原始数据
const likeRank = ref([
  { id: 53689, title: '“我也是有极限的啊。拖着这样半龙半人的身体，为哥哥鞍前马后地跑，哥哥还不领情，总以为我给他的那些好处是自来的似的。”“哥哥，你淋雨，我就不打伞。” #配音 #路鸣泽', likes: 44, cover: 'https://p2.a.yximgs.com/upic/2023/08/31/22/BMjAyMzA4MzEyMjExMDZfMTY4MTc5ODM1OV8xMTE4NDIzNTk4NjdfMV8z_ccc_Becc6a4562d6a003adace7f3417892bd3.jpg?tag=1-1744982237-xpcwebsearch-0-3donuea17z-7c197e428e5bd6' },
  { id: 32969, title: '#旅行中的风景 #我的旅行故事 #旅途中的我 #走喽', likes: 43, cover: 'https://p2.a.yximgs.com/upic/2025/04/11/09/BMjAyNTA0MTEwOTMwNTNfMzg3MDg2OTUzXzE2MTQzMjAwNDUwOF8yXzM=_B32439b88fd8bc7859fdfc491dadfeae6.jpg?tag=1-1744708209-xpcwebsearch-0-x0y9reo0ac-d5e6493307f5c1fa&c' },
  { id: 57675, title: '单腿深蹲的训练有助于让你膝盖越练越强 @快手热点(O3xddgkd5fav5if9) @快手粉条(O3xhcy6vhfzcu3qe) @快手健身(O3xq6pxy9umkct3w)', likes: 42, cover: 'https://p4.a.yximgs.com/upic/2021/06/01/18/BMjAyMTA2MDExODA2NTBfMjI5NDI5ODQ1MV81MDYzNjQwNzQ4OV8xXzM=_Be42ad6b7b7319fcadc29d0479b973b8c.jpg?tag=1-1744983441-xpcwebsearch-0-1113styqkc-4cc29d81868675b1&c' },
  { id: 24972, title: '#蓝心羽版淋一场雨 #就当作淋一场雨湿了眼睛 讨厌异地恋一万次，但我喜欢你一万零一次 #治愈系动画 #情侣日常', likes: 42, cover: 'https://p1.a.yximgs.com/upic/2025/01/25/21/BMjAyNTAxMjUyMTE0MzhfMjgxODU4NjM5MF8xNTQ5NzUzNzM1MjZfMV8z_ccc_B4818adbd910cf56a1d9828dfced990bf.jpg?tag=1-1744705594-xpcwebsearch-0-7w6ohihuq7-50b4e8f57432a8' },
  { id: 61704, title: '给大家爆料一下二手手机怎么选择 #手机使用小技巧 #手机小知识 #手机宝宝', likes: 40, cover: 'https://p1.a.yximgs.com/upic/2023/10/05/16/BMjAyMzEwMDUxNjI5NDVfMTIxMDI4NTkzMl8xMTQzODQ1MzU5MjdfMl8z_B72aff057172751b0c71ad7b8934b3cfb.jpg?tag=1-1744984831-xpcwebsearch-0-nwjxxelale-05bba5b6a2789305&c' },
  { id: 57637, title: '深蹲你真的做对了吗？ #健身打卡燥一夏', likes: 39, cover: 'https://p3.a.yximgs.com/upic/2021/05/31/21/BMjAyMTA1MzEyMTM0MTBfMTg0NDE1OTIzMl81MDU5MDM4NjEwNF8xXzM=_Bc57d614ecbaf7a9945b13e97436a86ed.jpg?tag=1-1744983435-xpcwebsearch-0-9jxgxkjigj-0b0dd0fe132b2e09&c' },
  { id: 32998, title: '#旅途中的我 #我的旅行故事', likes: 38, cover: 'https://p1.a.yximgs.com/upic/2025/04/10/06/BMjAyNTA0MTAwNjUxMjhfOTIyNTI3Mjc5XzE2MTM1NTQxMDE1Nl8yXzM=_Bf1c5c40082a9a09fb3a11017a5bb46d3.jpg?tag=1-1744708212-xpcwebsearch-0-xkkufui0ci-8acf2783b20af0b8&c' },
  { id: 48474, title: '日常英语口语 英语日常口语 语言学习 零基础学英语', likes: 38, cover: 'https://picsum.photos/300/80?random' },
  { id: 8389, title: '芒果台这次真的捡到宝了，《浪姐6》的唱跳黑马竟是搞笑女张小婉，张小婉凭实力打破了喜剧人的标签，直接飞升女神舞娘，真的惊艳了一眼又一眼。踢后腿，空中臀桥，舞蹈功底直接拉满。', likes: 38, cover: 'https://p2.a.yximgs.com/upic/2025/04/13/09/BMjAyNTA0MTMwOTUxMjZfMjQ5NzcwNzIzNV8xNjE2MjUwMDQzMThfMV8z_B4a015a77c53dcfc28f8c2a0a57d11630.jpg?tag=1-1744604763-xpcwebsearch-0-kupknwk5iz-4e1a8366327769cd&c' },
  { id: 8402, title: '林莜作品，改编版练习。魔三斤的这种学唱，可谓是魔音绕梁啊。也能看出有些曲艺的功底', likes: 38, cover: 'https://p1.a.yximgs.com/upic/2021/01/14/22/BMjAyMTAxMTQyMjE3MzhfMjE2NDU1MDUzMF80MjM4MTEzMjgwNV8yXzM=_B965a4d38bee6606ccedd109839f8416e.jpg?tag=1-1744604766-xpcwebsearch-0-lqmj6ykplg-3388e9019b77e759&c' },
  { id: 57639, title: '把深蹲这个简单的动作练对，已经能够快速提升弹跳，找对方法永远大于盲目努力。', likes: 38, cover: 'https://p1.a.yximgs.com/upic/2024/12/06/21/BMjAyNDEyMDYyMTE0MThfMzA5NzQ1NTQ3NV8xNTA0ODIwNDMwNTdfMF8z_Bdbad33ae22fed81ffef04a94b73f5f29.jpg?tag=1-1744983438-xpcwebsearch-0-5avfkqzdq3-15105d37ee419fb0&c' },
  { id: 8376, title: '恭喜王珞丹姐姐在三公正式成为演员版唱跳歌手', likes: 37, cover: 'https://p1.a.yximgs.com/upic/2025/04/12/22/BMjAyNTA0MTIyMjE0MzdfMTE2MDAwMzMyMV8xNjE1OTU0ODI5MjJfMl8z_B7e08b2eda4f5b81940ea8b58580bf776.jpg?tag=1-1744604763-xpcwebsearch-0-ccihfopxp6-669d1537230aa019&c' },
  { id: 48459, title: '英语口语 英语听力 英语日常口语 语言学习 零基础学英语', likes: 37, cover: 'https://picsum.photos/300/80?random' },
  { id: 53572, title: '世界上最美妙的感觉就是 当你拥抱一个你爱的人时，他竟然把你抱得更紧. #治愈系动画 #情侣日常 #动画', likes: 36, cover: 'https://p1.a.yximgs.com/upic/2025/02/16/15/BMjAyNTAyMTYxNTU2NDRfMjgxODU4NjM5MF8xNTcxMDc4NjA2OTJfMV8z_Bbebf141ce3153823a8e4743ba7d39895.jpg?tag=1-1744982200-xpcwebsearch-0-01nabh1i2g-2ed948b39d7766ee&c' },
  { id: 53548, title: '#请你别走太快 我可以不听，但你不能不管 #甜甜的恋爱 #情侣日常 #治愈系动画', likes: 36, cover: 'https://p1.a.yximgs.com/upic/2025/03/19/18/BMjAyNTAzMTkxODUxMjZfMzM4NDUzMTk5OF8xNTk2MTc0MTI1MTVfMl8z_B802e5cae4864f0a61892ed94fde0f5ac.jpg?tag=1-1744982197-xpcwebsearch-0-ry3ygytzrf-4f09a9a2482a8b51&c' },
  { id: 53699, title: '#配音 你已经长大了', likes: 36, cover: 'https://p5.a.yximgs.com/upic/2024/07/28/15/BMjAyNDA3MjgxNTA0MzFfMjY2MDM3NzAyXzEzOTA1OTM0Nzg1OV8xXzM=_B952f5ca9777e3da8459d6cf9ccb2f739.jpg?tag=1-1744982237-xpcwebsearch-0-qv1xhrgzvm-ba4223d68690065c&c' },
  { id: 57674, title: '做深蹲膝盖痛，你就这么练！一个月后整个人都🐮了 #深蹲', likes: 36, cover: 'https://p1.a.yximgs.com/upic/2022/12/18/19/BMjAyMjEyMTgxOTU5MjJfMTE2MjM4NDA5NV85MTQyNzg0NDkwMV8yXzM=_Bbdae8adfbb471a9559002401236467a6.jpg?tag=1-1744983441-xpcwebsearch-0-hcheqyy0gb-67ff908bf8be447e&c' },
  { id: 57666, title: '这个视频有点长 请看到最后 能否换来一个小红心♥️ #客厅健身房 ##客厅挑战赛', likes: 36, cover: 'https://p2.a.yximgs.com/upic/2020/01/22/14/BMjAyMDAxMjIxNDI2MzlfMjAyMjYzOTlfMjIyMjk2NzU5OTBfMV8z_B49050a4aa77cb4dc33b4abd753267ca3.jpg?tag=1-1744983441-xpcwebsearch-0-0y51whbchc-4f30abf5e746cb14&clien' },
  { id: 53683, title: '哪一个是你最喜欢的角色声音', likes: 36, cover: 'https://p2.a.yximgs.com/upic/2024/05/19/17/BMjAyNDA1MTkxNzA0MzFfOTc0NTQ5MTA1XzEzMjc5MjcxMzkzOF8xXzM=_ccc_B76d5371bbb74586a3f7ec6dc4eda4285.jpg?tag=1-1744982234-xpcwebsearch-0-w66gh5hplz-126333f3bc71a1' },
  { id: 48452, title: '日常英语口语 英语口语练习打卡 英语日常口语 语言学习 零基础学英语', likes: 36, cover: 'https://picsum.photos/300/80?random' },
  { id: 57663, title: '如何正确的做好深蹲 #深蹲', likes: 35, cover: 'https://p2.a.yximgs.com/upic/2021/12/17/22/BMjAyMTEyMTcyMjE1MDlfMjMwMjc3NDk3MV82Mjk3MDA0NjIyNl8xXzM=_Bd59d44cda64b0f8e67b73dc4cde98a7b.jpg?tag=1-1744983441-xpcwebsearch-0-ehjjectulu-07088e9be383421e&c' },
  { id: 400, title: '白鲸吃鱼！', likes: 35, cover: 'https://picsum.photos/300/80?random' },
  { id: 53530, title: '拧巴的人需要一个赶不走的爱人 #治愈系动画 #情侣日常 #动画', likes: 35, cover: 'https://p2.a.yximgs.com/upic/2025/01/19/18/BMjAyNTAxMTkxODExMDhfMjgxODU4NjM5MF8xNTQzNzg5NTc2MjRfMV8z_ccc_Bd885d5463e9794b71f7956133a6a40af.jpg?tag=1-1744982193-xpcwebsearch-0-pjq5iwywpj-4ea71ef1126889' },
  { id: 57645, title: '你练翘臀的深蹲姿势对了吗？正确的深蹲才能有效地臀桥，快点赞收藏练起来吧！#减肥', likes: 35, cover: 'https://p3.a.yximgs.com/upic/2019/11/26/18/BMjAxOTExMjYxODM1NDlfMTM2OTY3Mjc4XzE5OTA5MjE1NzI0XzFfMw==_B1860ab7b499080df94f662bd3e89df95.jpg?tag=1-1744983438-xpcwebsearch-0-dojfh8slzj-41239490a6408ad2&c' },
  { id: 53578, title: '感情出现裂痕后还可以复原吗 #治愈系动画 #情侣日常 #内容启发分享计划', likes: 35, cover: 'https://p1.a.yximgs.com/upic/2025/03/18/18/BMjAyNTAzMTgxODAxNTRfMjgxODU4NjM5MF8xNTk1NDcwNTYzMjdfMV8z_ccc_B45b9e81b915e85a74d4cb4d8104ca9f3.jpg?tag=1-1744982200-xpcwebsearch-0-lnfgp8wvyv-287ec598d4f66c' },
  { id: 53580, title: '@睡前动漫故事(O3xqvbh6ufd5e9ic) 的精彩视频', likes: 34, cover: 'https://p5.a.yximgs.com/upic/2020/05/17/11/BMjAyMDA1MTcxMTM3MzZfODUxNzM3Njk5XzI4NzEzODc2NTY1XzJfMw==_Bb29196065aa03b73f98767bdebd24147.jpg?tag=1-1744982200-xpcwebsearch-0-bod9ajblgd-95798f1758fcda1f&c' },
  { id: 8398, title: '一些有趣的瞬间📷', likes: 34, cover: 'https://p1.a.yximgs.com/upic/2025/04/13/18/BMjAyNTA0MTMxODU3MTlfMTc3MzY1ODRfMTYxNjc1MTU1ODIzXzFfMw==_Bb074eae4b9d461529e5e8f9050622bb7.jpg?tag=1-1744604766-xpcwebsearch-0-swbpvw3aov-f3c8ef00584899fc&c' },
  { id: 57668, title: '瘦大腿 宽距深蹲够厉害👍 也可以抱个壶铃！！！负重更难！也更酸爽！！！', likes: 34, cover: 'https://p1.a.yximgs.com/upic/2024/10/29/21/BMjAyNDEwMjkyMTAzNDFfMTA4MzYzNzY2XzE0NzI4NjA4MDQ4MF8xXzM=_Bf77bb190fd6302a850e1a96397a3e726.jpg?tag=1-1744983441-xpcwebsearch-0-hjjttdvaki-2c494222c6fbda61&c' },
  { id: 57670, title: '深蹲教学', likes: 34, cover: 'https://p2.a.yximgs.com/upic/2019/07/08/17/BMjAxOTA3MDgxNzI0MDNfMTUwNzU4OTQzXzE0OTM3Mjg3NTcwXzFfMw==_B865824e1ad50a84c5b631ca2b58c117d.jpg?tag=1-1744983441-xpcwebsearch-0-m8xreuhuyo-dcb315994fd9a156&c' },
  { id: 8404, title: '不唱改跳了', likes: 34, cover: 'https://p2.a.yximgs.com/upic/2025/04/12/20/BMjAyNTA0MTIyMDQxNDZfODY5MzczMjQ0XzE2MTU4NTM2NzczMl8yXzM=_B813f62aa20e5a00f05ede1bf5d5c1402.jpg?tag=1-1744604766-xpcwebsearch-0-hogambphfz-16e1ec7c3e04413b&c' },
  { id: 8362, title: '最近被这土嗨土嗨的神曲洗脑了😂来！左边跟我一起……', likes: 33, cover: 'https://p4.a.yximgs.com/upic/2019/09/13/19/BMjAxOTA5MTMxOTQxNDBfMjA0MjcyNjQzXzE3NDM5Nzk5ODMyXzFfMw==_Bc08cc162291e8fbebb3215f4fdb92ecd.jpg?tag=1-1744604760-xpcwebsearch-0-cveil1iqts-72227b85dedceb4c&c' },
  { id: 8369, title: '唱跳 唱跳', likes: 33, cover: 'https://p2.a.yximgs.com/upic/2025/04/12/13/BMjAyNTA0MTIxMzQxNTBfOTA1MzQ5NTI5XzE2MTUzOTU4MDc3OF8xXzM=_Bb0ce3015915376856ea46c8ac15e8625.jpg?tag=1-1744604760-xpcwebsearch-0-xkcqqsh12s-3622865a82ae6630&c' },
  { id: 8397, title: '四舍五入我也是个唱跳博主了', likes: 33, cover: 'https://p2.a.yximgs.com/upic/2020/03/15/18/BMjAyMDAzMTUxODA0MjFfMTYwNzg2NV8yNTAwNzY4NjQyN18xXzM=_B0e9d4a8a696dfc654eb90d94347b3005.jpg?tag=1-1744604766-xpcwebsearch-0-z9xnnmo2qw-87a48d595b05dcc9&clien' },
  { id: 48593, title: '@红寺堡婉居软装(O3xhhjefkwte7sus) 的精彩视频', likes: 33, cover: 'https://picsum.photos/300/80?random' },
  { id: 57664, title: '如何练习深蹲 这几个动作要领你记住了吗？ #健身', likes: 33, cover: 'https://p1.a.yximgs.com/upic/2019/12/12/11/BMjAxOTEyMTIxMTQ3MDlfMTYxNjM2NjU1NV8yMDQyNDY3NjM3NV8yXzM=_B16607a17558210829eef5119c27c93cc.jpg?tag=1-1744983441-xpcwebsearch-0-rceypsdjzy-7d07e82fc5de2012&c' },
  { id: 360, title: '水下拍摄 海底世界 海洋生物 鲨鱼', likes: 32, cover: 'https://picsum.photos/300/80?random' },
  { id: 2340, title: '同城好店推荐 四平电器 家电补贴 职场变迁 超实惠', likes: 32, cover: 'https://picsum.photos/300/80?random' },
  { id: 53171, title: '熬过了异地恋 就是一辈子#以爱之名你还愿意吗#情侣日常#治愈系动画', likes: 32, cover: 'https://p2.a.yximgs.com/upic/2024/10/18/13/BMjAyNDEwMTgxMzA1NTJfNDEyOTEyNjU4NF8xNDYzNDUwMzAzNjJfMF8z_Bfff59d4ca0c1a555623c2f35a024b452.jpg?tag=1-1744982095-xpcwebsearch-0-kxzzokn3ab-63336315dc5c002e&c' },
  { id: 48619, title: '婉居软装 我们家的婚庆款高端私人定制绝', likes: 32, cover: 'https://picsum.photos/300/80?random' },
  { id: 53187, title: '情侣之间频繁亲嘴的“后果”#治愈系动画 #情侣日常 #动画', likes: 32, cover: 'https://p2.a.yximgs.com/upic/2025/02/04/18/BMjAyNTAyMDQxODE5MDNfMjgxODU4NjM5MF8xNTU5NTQ0MDUzODNfMV8z_ccc_Bcdd724d7ec0a66abe3e173e9f87a70fd.jpg?tag=1-1744982098-xpcwebsearch-0-pgwmxr1iw6-c08e6d910d5106' },
  { id: 57661, title: '深蹲，深蹲跳 标准做法和训练方法', likes: 32, cover: 'https://p4.a.yximgs.com/upic/2021/10/11/18/BMjAyMTEwMTExODU2MTJfNTQxODgwNzQ1XzU4ODgwNDkwNzk4XzFfMw==_B23e51e9dcd7803ab59491c85c687cec4.jpg?tag=1-1744983441-xpcwebsearch-0-sodtvllmey-48d206bfed5e7426&c' },
  { id: 53527, title: '宫崎骏笔下的动漫，很治愈。', likes: 32, cover: 'https://p1.a.yximgs.com/upic/2025/02/23/23/BMjAyNTAyMjMyMzMzMzlfMTQyOTQ1MjI4NF8xNTc3MTcwMTkxNzhfMV8z_B044edcf313ebcf3226dcb40c4551676d.jpg?tag=1-1744982193-xpcwebsearch-0-f9egaztnpu-42098fc3b4a89ed3&c' },
  { id: 53539, title: '觉得像谁就艾特谁吧 #情侣日常 #治愈系动画', likes: 32, cover: 'https://p3.a.yximgs.com/upic/2024/01/18/12/BMjAyNDAxMTgxMjA2MTRfMzg3MDM3MjI2NV8xMjIzOTAwNTA3MzlfMl8z_Bb136448ec76333a69a71e0014ae2281f.jpg?tag=1-1744982193-xpcwebsearch-0-mfxeg06hsf-69533492af12727f&c' },
  { id: 32945, title: '#旅行中的风景 #旅途中的我 #我的旅行故事 #我们公司旅行可以不', likes: 32, cover: 'https://p1.a.yximgs.com/upic/2025/04/13/09/BMjAyNTA0MTMwOTE2NDlfMTI5Njk4MTgwOF8xNjE2MjIzOTMzNjJfMl8z_B1a6aefc6ec2b97dad1aefb6a8a89a5f0.jpg?tag=1-1744708205-xpcwebsearch-0-fiugh8ikj0-bcdd491f410a2de4&c' },
  { id: 48599, title: '为新家添置的每一件单品都是我精心挑选的，装修新家耗尽了我所有的心思，在自己能力范围之内想把所有喜欢的东西都买回来放在我的新家，或许这是对理想生活的一种期待。', likes: 32, cover: 'https://picsum.photos/300/80?random' },
  { id: 53680, title: '宇智波鼬的声优石川英郎本色出演，他真的 我哭死 @ㅤqi(O3xpb8442swnyade)', likes: 31, cover: 'https://p2.a.yximgs.com/upic/2023/11/04/09/BMjAyMzExMDQwOTQxNTRfMjgxMzQyMTM4NV8xMTY0MDk0OTM5MDFfMl8z_B64511693fa142fe0eb2957df15ec3eb0.jpg?tag=1-1744982234-xpcwebsearch-0-qnqdzk5e8z-6f211955c9477583&c' },
  { id: 8372, title: '@贞青春有你(O3x6e9bsyg92zf3e) 的精彩视频', likes: 31, cover: 'https://p2.a.yximgs.com/upic/2025/04/12/15/BMjAyNTA0MTIxNTExMDhfNDQ1MjA2NTcwOF8xNjE1NDkxOTY0MjBfMF8z_Bad181bf8a0ef8dda1a74c5d81b576b03.jpg?tag=1-1744604760-xpcwebsearch-0-mm5cv5lcjj-d1480551c0bebf56&c' },
  { id: 24987, title: '熬过异地恋，就是一辈子。#治愈系动画 #情侣日常 #异地恋 #熬过异地恋就是一辈子', likes: 31, cover: 'https://p5.a.yximgs.com/upic/2025/02/23/22/BMjAyNTAyMjMyMjM0MTZfNDMxNzQ0ODE1XzE1NzcxMzgxNjc0Ml8xXzM=_B233ec1733ab0a39e7cc60446ed2f9d93.jpg?tag=1-1744705597-xpcwebsearch-0-xtcv3fdgng-cc37ec705bc38172&c' },
  { id: 8392, title: '后面直接唱错了😓', likes: 31, cover: 'https://p3.a.yximgs.com/upic/2025/04/13/22/BMjAyNTA0MTMyMjUzMDdfMTc3Nzk5ODE1OV8xNjE2OTc2Nzc1NDZfMl8z_ccc_Bd9090c71b72927b112a280afe1b8d5cb.jpg?tag=1-1744604763-xpcwebsearch-0-pfosb4jrqy-6572e2e3961291' },
  { id: 8391, title: '自编自跳 小哥真行', likes: 31, cover: 'https://p1.a.yximgs.com/upic/2025/04/12/20/BMjAyNTA0MTIyMDQ5NDJfNDEwNDYxODkzXzE2MTU4NjMzOTcwM18yXzM=_Ba2b60c5c0dce0e67249cfb8cf95de989.jpg?tag=1-1744604763-xpcwebsearch-0-lfzwsa6axc-360a146b4b8cf301&c' },
  { id: 46368, title: '一年四季常绿，而且很好养，浇浇水就可以', likes: 30, cover: 'https://picsum.photos/300/80?random' },
  { id: 57650, title: '深蹲，健身必练的一个动作，当之无愧的动作之王，你做对了吗？', likes: 30, cover: 'https://p2.a.yximgs.com/upic/2019/05/15/11/BMjAxOTA1MTUxMTUwNTNfMTMzMjg5ODEyN18xMzA4MDI0NjkzMV8xXzM=_B89b1c33cc464beacc3748814778393c8.jpg?tag=1-1744983438-xpcwebsearch-0-maxbqkcpcy-60e01fb13dd75b39&c' },
  { id: 8401, title: '那边的朋友 让我听到你们的声音！', likes: 30, cover: 'https://p2.a.yximgs.com/upic/2019/09/21/18/BMjAxOTA5MjExODEyMjhfMzY1NzgyOV8xNzcwNzk4MjY5NF8xXzM=_Ba6d8a59e8421278458d37a5a8ac68fc9.jpg?tag=1-1744604766-xpcwebsearch-0-kwsifufk5m-1082cfcdcf92e5f6&clien' },
  { id: 53557, title: '“我想你了”=你在身边就好了 #治愈系动画 #情侣日常 #动画', likes: 29, cover: 'https://p1.a.yximgs.com/upic/2025/03/03/19/BMjAyNTAzMDMxOTA3MDdfMjgxODU4NjM5MF8xNTgzNDc2MTI0MzVfMV8z_B8601816474fa159bea756b0159ff7e17.jpg?tag=1-1744982197-xpcwebsearch-0-m0am5jlhri-47b1c62382ce50a2&c' },
  { id: 8359, title: '学了半个月的舞，拿出来不丢人吧', likes: 29, cover: 'https://p1.a.yximgs.com/upic/2019/10/24/19/BMjAxOTEwMjQxOTU2NTNfOTgxMzAwMjdfMTg4MzYzNTMwNzhfMV8z_Bfa2703b990ced81ea285ba4b30fb66d6.jpg?tag=1-1744604760-xpcwebsearch-0-564uzazite-3b1d0dce17f3b0ba&clien' },
  { id: 53547, title: '#侵袭式回忆 多抱抱你的男友吧 #情侣日常 #治愈系动画 #内容过于真实', likes: 29, cover: 'https://p5.a.yximgs.com/upic/2025/03/21/23/BMjAyNTAzMjEyMzU5NTdfMjg1NTkzODUyNF8xNTk3ODY0NDE5MzJfMV8z_B99e49597c7a6670bdab5c0c02c5abc37.jpg?tag=1-1744982197-xpcwebsearch-0-popvckwzic-7a60d80dbb239326&c' },
  { id: 8371, title: '@海鸟短剧(O3xqwmsgy7tyevjc) 的精彩视频', likes: 29, cover: 'https://p1.a.yximgs.com/upic/2025/04/14/10/BMjAyNTA0MTQxMDIyMzJfNDA1NzkwMTQ0NF8xNjE3MjA2ODgxMTJfMF8z_Baad444de458de37dc5171b55fa0415e8.jpg?tag=1-1744604760-xpcwebsearch-0-qgo6dosygb-a4ca22de5789b0a3&c' },
  { id: 32978, title: '#旅途中的我 #旅行中的风景 #我的旅行故事 #出来放风', likes: 29, cover: 'https://p2.a.yximgs.com/upic/2025/04/10/17/BMjAyNTA0MTAxNzQ3MTRfNDE0MDM2MzI2XzE2MTM4OTUyMzQxM18yXzM=_B5c22fc4723d46beba1e800d611dc4c7b.jpg?tag=1-1744708209-xpcwebsearch-0-urmsg5wffd-c81dd78aa28ea823&c' },
  { id: 48446, title: '日常英语口语 英语口语练习打卡 英语日常口语 语言学习 零基础学英语', likes: 28, cover: 'https://picsum.photos/300/80?random' },
  { id: 62775, title: '选择题万能通用口诀！让考试超简单！轻松拿高分～', likes: 28, cover: 'https://picsum.photos/300/80?random' },
  { id: 48455, title: '英语口语练习 英语听力 英语日常口语 语言学习 零基础学英语', likes: 28, cover: 'https://picsum.photos/300/80?random' },
  { id: 48604, title: '@红寺堡婉居软装(O3xhhjefkwte7sus) 的精彩视频', likes: 27, cover: 'https://picsum.photos/300/80?random' },
  { id: 48592, title: '香薰', likes: 27, cover: 'https://picsum.photos/300/80?random' },
  { id: 48448, title: '日常英语口语 英语日常口语 语言学习 零基础学英语', likes: 26, cover: 'https://picsum.photos/300/80?random' },
  { id: 8385, title: '宿舍唱跳版恋爱告急来啦，哈哈 #鞠婧祎恋爱告急', likes: 26, cover: 'https://p4.a.yximgs.com/upic/2023/03/12/14/BMjAyMzAzMTIxNDAxMzRfMTEyMTUyMTM0OF85ODA1NTIwNTQ5Ml8xXzM=_B2d200bceb20d030d731dd8bb64cb99c1.jpg?tag=1-1744604763-xpcwebsearch-0-1nbk7aaxng-4ff8609bfd304c63&c' },
  { id: 19229, title: '初高中学习资料分享 学霸秘籍 高分秘籍', likes: 26, cover: 'https://picsum.photos/300/80?random' },
  { id: 53681, title: '#配音 #鬼灭之刃 #动漫 @快手粉条(O3xhcy6vhfzcu3qe) @快手热点(O3xddgkd5fav5if9) @快手用户1733684182531(O3x8tieq2gk9hk2y) 鬼灭合集来了！', likes: 25, cover: 'https://p1.a.yximgs.com/upic/2024/07/10/10/BMjAyNDA3MTAxMDUzMDNfNTQ3NzE0NjgxXzEzNzI3MTAxNzk0N18xXzM=_B3922f0978d80fc2aef1af37de964da96.jpg?tag=1-1744982234-xpcwebsearch-0-8hfqcunymm-c159be6b6b907087&c' },
  { id: 32968, title: '#旅途中的我 #我的旅行故事 #旅行中的风景', likes: 25, cover: 'https://p1.a.yximgs.com/upic/2025/02/02/09/BMjAyNTAyMDIwOTQxNTdfMTQwNDcyNjM1OV8xNTU3MTgwMzU4MDFfMl8z_B8ae7d90041e2ef13677dd7bd5ae43e07.jpg?tag=1-1744708209-xpcwebsearch-0-pms6l6gisn-9b62b972f6e55934&c' },
  { id: 32965, title: '#旅途中的我 #我的旅行故事 #旅行中的风景', likes: 24, cover: 'https://p5.a.yximgs.com/upic/2025/04/13/07/BMjAyNTA0MTMwNzQ3NDhfODAyMTYxNzMxXzE2MTYxNzEwNzEyMV8yXzM=_Bf906746386a9c5df34c6133a7bf3d988.jpg?tag=1-1744708209-xpcwebsearch-0-nnif1eviaw-094c06fb953e2b3d&c' },
  { id: 62388, title: '我爱发明之《逆天科技top5》！ #离谱 #搞笑 #我爱发明 #一口气看完', likes: 21, cover: 'https://p2.a.yximgs.com/upic/2025/01/20/16/BMjAyNTAxMjAxNjIxMDFfMjg3MTI4OTkyNV8xNTQ0NjQ5NzkwNTBfMF8z_Bde04f99864de2928042be9b75a231d00.jpg?tag=1-1744985070-xpcwebsearch-0-pfdzde7mgt-afe96a48a5f21788&c' },
  { id: 1845, title: '🌱春耕启航·播种希望 🌞夏耘深耕·高效工作 🍂秋收硕果·升职加薪 ❄️冬藏蓄力·工作顺心生活美满 四季耕耘，成长加速！ 加入我们，让学习有「季」可循！', likes: 21, cover: 'https://picsum.photos/300/80?random' },
  { id: 35783, title: '“只有狗狗不在乎你在外面混的好或坏，当它见到你的时候，开心的像个孩子” #可爱狗 #修勾 #久别重逢', likes: 20, cover: 'https://p4.a.yximgs.com/upic/2025/01/07/08/BMjAyNTAxMDcwODA4MDZfMTE4MjIzMjFfMTUzMjcxMzM4OTM0XzJfMw==_B5845f223b09f30056946b0adad142cf5.jpg?tag=1-1744709113-xpcwebsearch-0-katndc5w9i-89f15b79315ac29c&c' },
  { id: 16033, title: '新手也能零失败，法式硬欧轻松搞定 #太原烘焙培训 #教学员技术配方毫无保留 #烘焙培训', likes: 20, cover: 'https://p5.a.yximgs.com/upic/2025/03/19/22/BMjAyNTAzMTkyMjMyNDlfOTc4MDgyMTAxXzE1OTYzNTMwOTcwOV8yXzM=_B562718972581f9ddee42bae17e6354c3.jpg?tag=1-1744602063-xpcwebsearch-0-4dwlyajq9y-3dbd06247bade7ae&c' },
  { id: 13356, title: '是龙你就盘着，是虎你就卧着，做事低调，活的心安。#快手小剧场 #天天拍好剧#快手短剧冷启大赏', likes: 20, cover: 'https://p1.a.yximgs.com/upic/2023/06/21/13/BMjAyMzA2MjExMzQ0MTFfMzUwMTQ0OTcwN18xMDYwNTQyMDUyMjNfMF8z_Bba2b7bd5e5560d7f801aacd5574c5942.jpg?tag=1-1744603526-xpcwebsearch-0-xbk8hnngzh-5de61527f179020c&c' },
  { id: 9408, title: '#真实还原系列 琪燕妈妈又出来嘚瑟了', likes: 20, cover: 'https://p5.a.yximgs.com/upic/2024/04/01/19/BMjAyNDA0MDExOTIxMjFfMTc0NjA0NjA0M18xMjg3NTAyNjM3MDBfMV8z_Bd0e87855a84a0c27566331b5a2b96261.jpg?tag=1-1744603952-xpcwebsearch-0-cnysekha8g-d8575e7929b966ef&c' },
  { id: 58514, title: '告别死记硬背！历史高效四步法', likes: 20, cover: 'https://picsum.photos/300/80?random' },
  { id: 17319, title: '第一次如何面试瑞幸的兼职', likes: 20, cover: 'https://picsum.photos/300/80?random' },
  { id: 34414, title: '#搞笑配音 #快成长计划', likes: 20, cover: 'https://p1.a.yximgs.com/upic/2025/04/14/06/BMjAyNTA0MTQwNjQzNDNfMTg0NjU3NjgyMl8xNjE3MTExNzQ3MzRfMl8z_B6c06e44b39044d4afcb0ff0bd6a0b3ac.jpg?tag=1-1744703105-xpcwebsearch-0-dmktpfzkrk-9889558ff12d9aa1&c' },
  { id: 26663, title: '“我突然发现，我好像没什么朋友，也没有很爱我的人，突然笑自己，怎么会这么孤独，明明我以前是，那么爱玩的人啊，现在怎么就我自己了呢” #情感 #手机摄影 #热文案', likes: 20, cover: 'https://p1.a.yximgs.com/upic/2022/03/30/21/BMjAyMjAzMzAyMTMxNTlfMjM5MjM0NTUxMl83MDgzNDQyNDkxMl8yXzM=_B07213038ec5ffa84e0de8ab2d8ecc5fa.jpg?tag=1-1744706382-xpcwebsearch-0-mcrpm2l2qo-5f4e7a45e956200c&c' },
  { id: 44732, title: '第73集 - 你不知道的大疆无人机“禁飞区”！', likes: 20, cover: 'https://picsum.photos/300/80?random' },
  { id: 46097, title: '猫咪怎么养才听话黏人 喵星人', likes: 20, cover: 'https://picsum.photos/300/80?random' },
  { id: 12712, title: '盘点沙雕农村搞笑视频 #农村搞笑视频逗乐每一天 #纯属搞笑让大家乐一乐 #老爷们儿是不是这样的呢 幽默搞笑农村人分享快乐生活每一天', likes: 19, cover: 'https://p1.a.yximgs.com/upic/2024/08/30/07/BMjAyNDA4MzAwNzQ5MzVfNDYzMzc1OTA1XzE0MjI2MTE5OTEwOF8yXzM=_Bb9881efb4f2b49514f166db4220601e3.jpg?tag=1-1744603431-xpcwebsearch-0-i2klbpnogu-269b574112b49fba&c' },
  { id: 33247, title: '#婚纱照 #变装 #氛围感', likes: 19, cover: 'https://p1.a.yximgs.com/upic/2025/01/10/22/BMjAyNTAxMTAyMjAyMzlfMTEzMTU0NzcxM18xNTM1Njk2MTY1OThfMV8z_B415371d90fa438a61e28150ff4dcf144.jpg?tag=1-1744708265-xpcwebsearch-0-dng395jwce-6f3a5a8a5a3c1a0d&c' },
  { id: 24679, title: '厨房防潮没做好，到处发霉湿哒哒！这几个防潮小技巧 ，一个都不能少！【厨房防水层】地面全防水墙面要做30cm高水槽区域要做到水龙头往上10cm的高度❗【台盆】选台下盆，再搭配抽卡水龙头日常清扫hin方便【注意❗】很多宝宝会担心台下盆会掉，但是现在技术很发达啦，金牌刚拿到了『台盆下扣技术』的专利，台盆上站人都没关系~【挡水条】说过hin多遍~一定一定，要做挡水条前挡水5mm，后挡水4cm高度【水槽柜】底板单单用铝箔纸包覆是不够的🏻底板的材料可以用雪弗板（雪弗板是运用在轮船底部的材料）水槽柜四周可以用不锈钢板，更加防水防潮！【踢脚线】一般做10-15cm，更好地保护柜体呀！', likes: 19, cover: 'https://p2.a.yximgs.com/upic/2021/06/15/15/BMjAyMTA2MTUxNTE2MDhfMTE4MDQ0MDE0NF81MTQ2MjYzODU3NF8wXzM=_Be838695e79ffd91a7b0882503ed83da7.jpg?tag=1-1744705497-xpcwebsearch-0-gky7ganh5g-ebb9d5d829341bb3&c' },
  { id: 33207, title: '终于用上这个BGM喽👰 #婚纱照拍摄花絮', likes: 19, cover: 'https://p1.a.yximgs.com/upic/2024/01/02/17/BMjAyNDAxMDIxNzQ4NDNfMTk0MjM2MDhfMTIxMjEwNDI4MDY0XzFfMw==_B28a5b5f46f4d1d03617150c9a810ebaf.jpg?tag=1-1744708259-xpcwebsearch-0-vehlzgvzrb-ec275863748df1ae&c' },
  { id: 17890, title: '@有妈真好🌈(O3x2p3e2fm2rmrva) 的精彩视频', likes: 19, cover: 'https://picsum.photos/300/80?random' },
  { id: 46482, title: '春蚕到死丝方尽，蜡炬成灰泪始干', likes: 19, cover: 'https://picsum.photos/300/80?random' },
  { id: 41178, title: '美国中央司令部公布战斗机起飞视频证实美军对也门发动袭击', likes: 19, cover: 'https://picsum.photos/300/80?random' },
  { id: 50042, title: '欢迎收看我的早间日常 #我的vlog日常 #日常vlog原创作品 #小洛很乖 #感谢快手平台记录每天的自己 关注小号 @xiao洛的碎碎念✨(O3xahjyprkp5ytzs)', likes: 19, cover: 'https://p5.a.yximgs.com/upic/2025/03/27/18/BMjAyNTAzMjcxODI1MzVfNTQ2NDg2Mzk4XzE2MDIzNzIwOTM0NV8xXzM=_B3fbbe5a2187e65f8d749943fff4121da.jpg?tag=1-1744977210-xpcwebsearch-0-vj8vqu09g9-c68132255bc54bac&c' },
  { id: 30752, title: '宝子们👋，夏天的脚步越来越近，是不是又到了满衣橱找短裤的时候啦？今天必须给大家分享我压箱底的宝藏 —— 洋气百搭短裤，穿上它，瞬间变身时尚弄潮儿，轻松解锁夏日 N 种时尚造型💃 #洋气百搭短裤#夏日时尚#时尚穿搭必备', likes: 19, cover: 'https://p2.a.yximgs.com/upic/2025/04/11/15/BMjAyNTA0MTExNTU1MTFfNTQ3MDgzMTAyXzE2MTQ1NDYxNDgxNF8wXzM=_B0152c670e0d09b56cf2baf22f54e9324.jpg?tag=1-1744707590-xpcwebsearch-0-bmjrjmiine-16841f442e2bee4b&c' },
  { id: 21291, title: '#集结吧光合创作者男人其实只是想忙碌了一天回到家能有一个贤妻良母般的女人能真心的疼爱自己，她的贤淑，勤劳，知书达理，温柔体贴能融化男人的所有压力和委屈，给男人母爱般的柔情和抚慰，希望余生大家都能遇到这样的女人 #做家务', likes: 19, cover: 'https://p2.a.yximgs.com/upic/2022/01/23/10/BMjAyMjAxMjMxMDE2MDZfMTY3Mjg3MzI0OV82NTUyNTMyNTY5Ml8yXzM=_B5d7a0fedb8bf25018a18f7e030a47a11.jpg?tag=1-1744702722-xpcwebsearch-0-jbn1h3fl0r-387da099a3cbe89d&c' },
  { id: 60312, title: '书桌脏乱差，看看我收纳 #生活小技巧 #生活小妙招 #收纳 #书桌', likes: 19, cover: 'https://p2.a.yximgs.com/upic/2024/11/08/20/BMjAyNDExMDgyMDMxNTZfMzY5OTk1NDI2XzE0ODA5MjYxNjYzNV8yXzM=_B23ff315e1961d585c905d51150e567f5.jpg?tag=1-1744977536-xpcwebsearch-0-y22wyzqhhe-a9e218090abdfc0c&c' },
  { id: 12545, title: '#颜值气质美女完美身材 #大长腿 #黑丝穿搭 #快成长计划', likes: 19, cover: 'https://p1.a.yximgs.com/upic/2024/12/03/12/BMjAyNDEyMDMxMjMyNTNfMzYyODM1MDA1NF8xNTAyMDk2NTk2MDFfMV8z_Beeaab15ebd439d2dde2624d83e237b96.jpg?tag=1-1744602405-xpcwebsearch-0-qxzteaa0ek-42b7e81923d55409&c' },
  { id: 43481, title: '世界上最恐怖的幼儿园。 @快手热点(O3xddgkd5fav5if9) @快手粉条(O3xhcy6vhfzcu3qe) @我要上热门(O3x8er38dpbhvbaa)', likes: 19, cover: 'https://picsum.photos/300/80?random' },
  { id: 19599, title: 'fyp 内容过于真实 搞笑幽默 课代表日常', likes: 19, cover: 'https://picsum.photos/300/80?random' },
  { id: 20262, title: '大学生找实习的3个平台', likes: 19, cover: 'https://picsum.photos/300/80?random' },
  { id: 49955, title: '#家庭教育很重要 #父母课堂 #兴趣才艺', likes: 19, cover: 'https://p2.a.yximgs.com/upic/2021/12/03/09/BMjAyMTEyMDMwOTM4MjRfMjE3NTU3MjEyNF82MTk4MDEzODcyOF8xXzM=_B8147c87a48dd9779db5b2e25867439cd.jpg?tag=1-1744979185-xpcwebsearch-0-fdev6dgncm-ab68c8f728096694&c' },
  { id: 33535, title: '我就能想到这么多了，大家还有啥补充的？#自驾游 #美女', likes: 19, cover: 'https://p5.a.yximgs.com/upic/2020/11/10/17/BMjAyMDExMTAxNzUyNDJfMTU4OTU3MjU2M18zOTAyNjY1Mjc4NV8wXzM=_Bbe41f18d2b5caf0a9575121ff4d20e10.jpg?tag=1-1744708351-xpcwebsearch-0-ecmjb6kioa-d0e84db2a1b860cf&c' },
  { id: 15899, title: '春游野餐你安排了吗？10分钟就搞定的午餐肉饭团，外带方便又美味！ #午餐肉饭团 #踏青赏花春游季', likes: 19, cover: 'https://p2.a.yximgs.com/upic/2021/03/13/21/BMjAyMTAzMTMyMTM3NDZfMTg5Mjk2NzU3MV80NTkzOTU2OTgyN18yXzM=_B32ccb9500fad1a89d52a0d1a1d77f8ee.jpg?tag=1-1744602107-xpcwebsearch-0-pk31utjc0q-2669dbfaa335f677&c' },
])

const viewRank = ref([
  { id: 32998, title: '#旅途中的我 #我的旅行故事', views: 85, cover: 'https://p1.a.yximgs.com/upic/2025/04/10/06/BMjAyNTA0MTAwNjUxMjhfOTIyNTI3Mjc5XzE2MTM1NTQxMDE1Nl8yXzM=_Bf1c5c40082a9a09fb3a11017a5bb46d3.jpg?tag=1-1744708212-xpcwebsearch-0-xkkufui0ci-8acf2783b20af0b8&c' },
  { id: 57675, title: '单腿深蹲的训练有助于让你膝盖越练越强 @快手热点(O3xddgkd5fav5if9) @快手粉条(O3xhcy6vhfzcu3qe) @快手健身(O3xq6pxy9umkct3w)', views: 80, cover: 'https://p4.a.yximgs.com/upic/2021/06/01/18/BMjAyMTA2MDExODA2NTBfMjI5NDI5ODQ1MV81MDYzNjQwNzQ4OV8xXzM=_Be42ad6b7b7319fcadc29d0479b973b8c.jpg?tag=1-1744983441-xpcwebsearch-0-1113styqkc-4cc29d81868675b1&c' },
  { id: 53689, title: '“我也是有极限的啊。拖着这样半龙半人的身体，为哥哥鞍前马后地跑，哥哥还不领情，总以为我给他的那些好处是自来的似的。”“哥哥，你淋雨，我就不打伞。” #配音 #路鸣泽', views: 76, cover: 'https://p2.a.yximgs.com/upic/2023/08/31/22/BMjAyMzA4MzEyMjExMDZfMTY4MTc5ODM1OV8xMTE4NDIzNTk4NjdfMV8z_ccc_Becc6a4562d6a003adace7f3417892bd3.jpg?tag=1-1744982237-xpcwebsearch-0-3donuea17z-7c197e428e5bd6' },
  { id: 360, title: '水下拍摄 海底世界 海洋生物 鲨鱼', views: 83, cover: 'https://picsum.photos/300/80?random' },
  { id: 57637, title: '深蹲你真的做对了吗？ #健身打卡燥一夏', views: 78, cover: 'https://p3.a.yximgs.com/upic/2021/05/31/21/BMjAyMTA1MzEyMTM0MTBfMTg0NDE1OTIzMl81MDU5MDM4NjEwNF8xXzM=_Bc57d614ecbaf7a9945b13e97436a86ed.jpg?tag=1-1744983435-xpcwebsearch-0-9jxgxkjigj-0b0dd0fe132b2e09&c' },
  { id: 32969, title: '#旅行中的风景 #我的旅行故事 #旅途中的我 #走喽', views: 75, cover: 'https://p2.a.yximgs.com/upic/2025/04/11/09/BMjAyNTA0MTEwOTMwNTNfMzg3MDg2OTUzXzE2MTQzMjAwNDUwOF8yXzM=_B32439b88fd8bc7859fdfc491dadfeae6.jpg?tag=1-1744708209-xpcwebsearch-0-x0y9reo0ac-d5e6493307f5c1fa&c' },
  { id: 57663, title: '如何正确的做好深蹲 #深蹲', views: 80, cover: 'https://p2.a.yximgs.com/upic/2021/12/17/22/BMjAyMTEyMTcyMjE1MDlfMjMwMjc3NDk3MV82Mjk3MDA0NjIyNl8xXzM=_Bd59d44cda64b0f8e67b73dc4cde98a7b.jpg?tag=1-1744983441-xpcwebsearch-0-ehjjectulu-07088e9be383421e&c' },
  { id: 24972, title: '#蓝心羽版淋一场雨 #就当作淋一场雨湿了眼睛 讨厌异地恋一万次，但我喜欢你一万零一次 #治愈系动画 #情侣日常', views: 75, cover: 'https://p1.a.yximgs.com/upic/2025/01/25/21/BMjAyNTAxMjUyMTE0MzhfMjgxODU4NjM5MF8xNTQ5NzUzNzM1MjZfMV8z_ccc_B4818adbd910cf56a1d9828dfced990bf.jpg?tag=1-1744705594-xpcwebsearch-0-7w6ohihuq7-50b4e8f57432a8' },
  { id: 53580, title: '@睡前动漫故事(O3xqvbh6ufd5e9ic) 的精彩视频', views: 80, cover: 'https://p5.a.yximgs.com/upic/2020/05/17/11/BMjAyMDA1MTcxMTM3MzZfODUxNzM3Njk5XzI4NzEzODc2NTY1XzJfMw==_Bb29196065aa03b73f98767bdebd24147.jpg?tag=1-1744982200-xpcwebsearch-0-bod9ajblgd-95798f1758fcda1f&c' },
  { id: 48474, title: '日常英语口语 英语日常口语 语言学习 零基础学英语', views: 77, cover: 'https://picsum.photos/300/80?random' },
  { id: 8389, title: '芒果台这次真的捡到宝了，《浪姐6》的唱跳黑马竟是搞笑女张小婉，张小婉凭实力打破了喜剧人的标签，直接飞升女神舞娘，真的惊艳了一眼又一眼。踢后腿，空中臀桥，舞蹈功底直接拉满。', views: 75, cover: 'https://p2.a.yximgs.com/upic/2025/04/13/09/BMjAyNTA0MTMwOTUxMjZfMjQ5NzcwNzIzNV8xNjE2MjUwMDQzMThfMV8z_B4a015a77c53dcfc28f8c2a0a57d11630.jpg?tag=1-1744604763-xpcwebsearch-0-kupknwk5iz-4e1a8366327769cd&c' },
  { id: 53572, title: '世界上最美妙的感觉就是 当你拥抱一个你爱的人时，他竟然把你抱得更紧. #治愈系动画 #情侣日常 #动画', views: 76, cover: 'https://p1.a.yximgs.com/upic/2025/02/16/15/BMjAyNTAyMTYxNTU2NDRfMjgxODU4NjM5MF8xNTcxMDc4NjA2OTJfMV8z_Bbebf141ce3153823a8e4743ba7d39895.jpg?tag=1-1744982200-xpcwebsearch-0-01nabh1i2g-2ed948b39d7766ee&c' },
  { id: 61704, title: '给大家爆料一下二手手机怎么选择 #手机使用小技巧 #手机小知识 #手机宝宝', views: 73, cover: 'https://p1.a.yximgs.com/upic/2023/10/05/16/BMjAyMzEwMDUxNjI5NDVfMTIxMDI4NTkzMl8xMTQzODQ1MzU5MjdfMl8z_B72aff057172751b0c71ad7b8934b3cfb.jpg?tag=1-1744984831-xpcwebsearch-0-nwjxxelale-05bba5b6a2789305&c' },
  { id: 2340, title: '同城好店推荐 四平电器 家电补贴 职场变迁 超实惠', views: 78, cover: 'https://picsum.photos/300/80?random' },
  { id: 8398, title: '一些有趣的瞬间📷', views: 76, cover: 'https://p1.a.yximgs.com/upic/2025/04/13/18/BMjAyNTA0MTMxODU3MTlfMTc3MzY1ODRfMTYxNjc1MTU1ODIzXzFfMw==_Bb074eae4b9d461529e5e8f9050622bb7.jpg?tag=1-1744604766-xpcwebsearch-0-swbpvw3aov-f3c8ef00584899fc&c' },
  { id: 53171, title: '熬过了异地恋 就是一辈子#以爱之名你还愿意吗#情侣日常#治愈系动画', views: 77, cover: 'https://p2.a.yximgs.com/upic/2024/10/18/13/BMjAyNDEwMTgxMzA1NTJfNDEyOTEyNjU4NF8xNDYzNDUwMzAzNjJfMF8z_Bfff59d4ca0c1a555623c2f35a024b452.jpg?tag=1-1744982095-xpcwebsearch-0-kxzzokn3ab-63336315dc5c002e&c' },
  { id: 53530, title: '拧巴的人需要一个赶不走的爱人 #治愈系动画 #情侣日常 #动画', views: 74, cover: 'https://p2.a.yximgs.com/upic/2025/01/19/18/BMjAyNTAxMTkxODExMDhfMjgxODU4NjM5MF8xNTQzNzg5NTc2MjRfMV8z_ccc_Bd885d5463e9794b71f7956133a6a40af.jpg?tag=1-1744982193-xpcwebsearch-0-pjq5iwywpj-4ea71ef1126889' },
  { id: 400, title: '白鲸吃鱼！', views: 74, cover: 'https://picsum.photos/300/80?random' },
  { id: 53548, title: '#请你别走太快 我可以不听，但你不能不管 #甜甜的恋爱 #情侣日常 #治愈系动画', views: 73, cover: 'https://p1.a.yximgs.com/upic/2025/03/19/18/BMjAyNTAzMTkxODUxMjZfMzM4NDUzMTk5OF8xNTk2MTc0MTI1MTVfMl8z_B802e5cae4864f0a61892ed94fde0f5ac.jpg?tag=1-1744982197-xpcwebsearch-0-ry3ygytzrf-4f09a9a2482a8b51&c' },
  { id: 46368, title: '一年四季常绿，而且很好养，浇浇水就可以', views: 76, cover: 'https://picsum.photos/300/80?random' },
  { id: 53699, title: '#配音 你已经长大了', views: 72, cover: 'https://p5.a.yximgs.com/upic/2024/07/28/15/BMjAyNDA3MjgxNTA0MzFfMjY2MDM3NzAyXzEzOTA1OTM0Nzg1OV8xXzM=_B952f5ca9777e3da8459d6cf9ccb2f739.jpg?tag=1-1744982237-xpcwebsearch-0-qv1xhrgzvm-ba4223d68690065c&c' },
  { id: 8376, title: '恭喜王珞丹姐姐在三公正式成为演员版唱跳歌手', views: 71, cover: 'https://p1.a.yximgs.com/upic/2025/04/12/22/BMjAyNTA0MTIyMjE0MzdfMTE2MDAwMzMyMV8xNjE1OTU0ODI5MjJfMl8z_B7e08b2eda4f5b81940ea8b58580bf776.jpg?tag=1-1744604763-xpcwebsearch-0-ccihfopxp6-669d1537230aa019&c' },
  { id: 57670, title: '深蹲教学', views: 72, cover: 'https://p2.a.yximgs.com/upic/2019/07/08/17/BMjAxOTA3MDgxNzI0MDNfMTUwNzU4OTQzXzE0OTM3Mjg3NTcwXzFfMw==_B865824e1ad50a84c5b631ca2b58c117d.jpg?tag=1-1744983441-xpcwebsearch-0-m8xreuhuyo-dcb315994fd9a156&c' },
  { id: 57668, title: '瘦大腿 宽距深蹲够厉害👍 也可以抱个壶铃！！！负重更难！也更酸爽！！！', views: 72, cover: 'https://p1.a.yximgs.com/upic/2024/10/29/21/BMjAyNDEwMjkyMTAzNDFfMTA4MzYzNzY2XzE0NzI4NjA4MDQ4MF8xXzM=_Bf77bb190fd6302a850e1a96397a3e726.jpg?tag=1-1744983441-xpcwebsearch-0-hjjttdvaki-2c494222c6fbda61&c' },
  { id: 57674, title: '做深蹲膝盖痛，你就这么练！一个月后整个人都🐮了 #深蹲', views: 70, cover: 'https://p1.a.yximgs.com/upic/2022/12/18/19/BMjAyMjEyMTgxOTU5MjJfMTE2MjM4NDA5NV85MTQyNzg0NDkwMV8yXzM=_Bbdae8adfbb471a9559002401236467a6.jpg?tag=1-1744983441-xpcwebsearch-0-hcheqyy0gb-67ff908bf8be447e&c' },
  { id: 8369, title: '唱跳 唱跳', views: 72, cover: 'https://p2.a.yximgs.com/upic/2025/04/12/13/BMjAyNTA0MTIxMzQxNTBfOTA1MzQ5NTI5XzE2MTUzOTU4MDc3OF8xXzM=_Bb0ce3015915376856ea46c8ac15e8625.jpg?tag=1-1744604760-xpcwebsearch-0-xkcqqsh12s-3622865a82ae6630&c' },
  { id: 8362, title: '最近被这土嗨土嗨的神曲洗脑了😂来！左边跟我一起……', views: 72, cover: 'https://p4.a.yximgs.com/upic/2019/09/13/19/BMjAxOTA5MTMxOTQxNDBfMjA0MjcyNjQzXzE3NDM5Nzk5ODMyXzFfMw==_Bc08cc162291e8fbebb3215f4fdb92ecd.jpg?tag=1-1744604760-xpcwebsearch-0-cveil1iqts-72227b85dedceb4c&c' },
  { id: 48593, title: '@红寺堡婉居软装(O3xhhjefkwte7sus) 的精彩视频', views: 71, cover: 'https://picsum.photos/300/80?random' },
  { id: 8397, title: '四舍五入我也是个唱跳博主了', views: 71, cover: 'https://p2.a.yximgs.com/upic/2020/03/15/18/BMjAyMDAzMTUxODA0MjFfMTYwNzg2NV8yNTAwNzY4NjQyN18xXzM=_B0e9d4a8a696dfc654eb90d94347b3005.jpg?tag=1-1744604766-xpcwebsearch-0-z9xnnmo2qw-87a48d595b05dcc9&clien' },
  { id: 53683, title: '哪一个是你最喜欢的角色声音', views: 69, cover: 'https://p2.a.yximgs.com/upic/2024/05/19/17/BMjAyNDA1MTkxNzA0MzFfOTc0NTQ5MTA1XzEzMjc5MjcxMzkzOF8xXzM=_ccc_B76d5371bbb74586a3f7ec6dc4eda4285.jpg?tag=1-1744982234-xpcwebsearch-0-w66gh5hplz-126333f3bc71a1' },
  { id: 57666, title: '这个视频有点长 请看到最后 能否换来一个小红心♥️ #客厅健身房 ##客厅挑战赛', views: 69, cover: 'https://p2.a.yximgs.com/upic/2020/01/22/14/BMjAyMDAxMjIxNDI2MzlfMjAyMjYzOTlfMjIyMjk2NzU5OTBfMV8z_B49050a4aa77cb4dc33b4abd753267ca3.jpg?tag=1-1744983441-xpcwebsearch-0-0y51whbchc-4f30abf5e746cb14&clien' },
  { id: 53680, title: '宇智波鼬的声优石川英郎本色出演，他真的 我哭死 @ㅤqi(O3xpb8442swnyade)', views: 72, cover: 'https://p2.a.yximgs.com/upic/2023/11/04/09/BMjAyMzExMDQwOTQxNTRfMjgxMzQyMTM4NV8xMTY0MDk0OTM5MDFfMl8z_B64511693fa142fe0eb2957df15ec3eb0.jpg?tag=1-1744982234-xpcwebsearch-0-qnqdzk5e8z-6f211955c9477583&c' },
  { id: 57645, title: '你练翘臀的深蹲姿势对了吗？正确的深蹲才能有效地臀桥，快点赞收藏练起来吧！#减肥', views: 69, cover: 'https://p3.a.yximgs.com/upic/2019/11/26/18/BMjAxOTExMjYxODM1NDlfMTM2OTY3Mjc4XzE5OTA5MjE1NzI0XzFfMw==_B1860ab7b499080df94f662bd3e89df95.jpg?tag=1-1744983438-xpcwebsearch-0-dojfh8slzj-41239490a6408ad2&c' },
  { id: 8402, title: '林莜作品，改编版练习。魔三斤的这种学唱，可谓是魔音绕梁啊。也能看出有些曲艺的功底', views: 67, cover: 'https://p1.a.yximgs.com/upic/2021/01/14/22/BMjAyMTAxMTQyMjE3MzhfMjE2NDU1MDUzMF80MjM4MTEzMjgwNV8yXzM=_B965a4d38bee6606ccedd109839f8416e.jpg?tag=1-1744604766-xpcwebsearch-0-lqmj6ykplg-3388e9019b77e759&c' },
  { id: 48452, title: '日常英语口语 英语口语练习打卡 英语日常口语 语言学习 零基础学英语', views: 68, cover: 'https://picsum.photos/300/80?random' },
  { id: 53557, title: '“我想你了”=你在身边就好了 #治愈系动画 #情侣日常 #动画', views: 72, cover: 'https://p1.a.yximgs.com/upic/2025/03/03/19/BMjAyNTAzMDMxOTA3MDdfMjgxODU4NjM5MF8xNTgzNDc2MTI0MzVfMV8z_B8601816474fa159bea756b0159ff7e17.jpg?tag=1-1744982197-xpcwebsearch-0-m0am5jlhri-47b1c62382ce50a2&c' },
  { id: 53187, title: '情侣之间频繁亲嘴的“后果”#治愈系动画 #情侣日常 #动画', views: 70, cover: 'https://p2.a.yximgs.com/upic/2025/02/04/18/BMjAyNTAyMDQxODE5MDNfMjgxODU4NjM5MF8xNTU5NTQ0MDUzODNfMV8z_ccc_Bcdd724d7ec0a66abe3e173e9f87a70fd.jpg?tag=1-1744982098-xpcwebsearch-0-pgwmxr1iw6-c08e6d910d5106' },
  { id: 48619, title: '婉居软装 我们家的婚庆款高端私人定制绝', views: 70, cover: 'https://picsum.photos/300/80?random' },
  { id: 8404, title: '不唱改跳了', views: 68, cover: 'https://p2.a.yximgs.com/upic/2025/04/12/20/BMjAyNTA0MTIyMDQxNDZfODY5MzczMjQ0XzE2MTU4NTM2NzczMl8yXzM=_B813f62aa20e5a00f05ede1bf5d5c1402.jpg?tag=1-1744604766-xpcwebsearch-0-hogambphfz-16e1ec7c3e04413b&c' },
  { id: 57661, title: '深蹲，深蹲跳 标准做法和训练方法', views: 69, cover: 'https://p4.a.yximgs.com/upic/2021/10/11/18/BMjAyMTEwMTExODU2MTJfNTQxODgwNzQ1XzU4ODgwNDkwNzk4XzFfMw==_B23e51e9dcd7803ab59491c85c687cec4.jpg?tag=1-1744983441-xpcwebsearch-0-sodtvllmey-48d206bfed5e7426&c' },
  { id: 8359, title: '学了半个月的舞，拿出来不丢人吧', views: 71, cover: 'https://p1.a.yximgs.com/upic/2019/10/24/19/BMjAxOTEwMjQxOTU2NTNfOTgxMzAwMjdfMTg4MzYzNTMwNzhfMV8z_Bfa2703b990ced81ea285ba4b30fb66d6.jpg?tag=1-1744604760-xpcwebsearch-0-564uzazite-3b1d0dce17f3b0ba&clien' },
  { id: 48446, title: '日常英语口语 英语口语练习打卡 英语日常口语 语言学习 零基础学英语', views: 71, cover: 'https://picsum.photos/300/80?random' },
  { id: 48459, title: '英语口语 英语听力 英语日常口语 语言学习 零基础学英语', views: 65, cover: 'https://picsum.photos/300/80?random' },
  { id: 8372, title: '@贞青春有你(O3x6e9bsyg92zf3e) 的精彩视频', views: 69, cover: 'https://p2.a.yximgs.com/upic/2025/04/12/15/BMjAyNTA0MTIxNTExMDhfNDQ1MjA2NTcwOF8xNjE1NDkxOTY0MjBfMF8z_Bad181bf8a0ef8dda1a74c5d81b576b03.jpg?tag=1-1744604760-xpcwebsearch-0-mm5cv5lcjj-d1480551c0bebf56&c' },
  { id: 57639, title: '把深蹲这个简单的动作练对，已经能够快速提升弹跳，找对方法永远大于盲目努力。', views: 64, cover: 'https://p1.a.yximgs.com/upic/2024/12/06/21/BMjAyNDEyMDYyMTE0MThfMzA5NzQ1NTQ3NV8xNTA0ODIwNDMwNTdfMF8z_Bdbad33ae22fed81ffef04a94b73f5f29.jpg?tag=1-1744983438-xpcwebsearch-0-5avfkqzdq3-15105d37ee419fb0&c' },
  { id: 57650, title: '深蹲，健身必练的一个动作，当之无愧的动作之王，你做对了吗？', views: 69, cover: 'https://p2.a.yximgs.com/upic/2019/05/15/11/BMjAxOTA1MTUxMTUwNTNfMTMzMjg5ODEyN18xMzA4MDI0NjkzMV8xXzM=_B89b1c33cc464beacc3748814778393c8.jpg?tag=1-1744983438-xpcwebsearch-0-maxbqkcpcy-60e01fb13dd75b39&c' },
  { id: 24987, title: '熬过异地恋，就是一辈子。#治愈系动画 #情侣日常 #异地恋 #熬过异地恋就是一辈子', views: 68, cover: 'https://p5.a.yximgs.com/upic/2025/02/23/22/BMjAyNTAyMjMyMjM0MTZfNDMxNzQ0ODE1XzE1NzcxMzgxNjc0Ml8xXzM=_B233ec1733ab0a39e7cc60446ed2f9d93.jpg?tag=1-1744705597-xpcwebsearch-0-xtcv3fdgng-cc37ec705bc38172&c' },
  { id: 53527, title: '宫崎骏笔下的动漫，很治愈。', views: 67, cover: 'https://p1.a.yximgs.com/upic/2025/02/23/23/BMjAyNTAyMjMyMzMzMzlfMTQyOTQ1MjI4NF8xNTc3MTcwMTkxNzhfMV8z_B044edcf313ebcf3226dcb40c4551676d.jpg?tag=1-1744982193-xpcwebsearch-0-f9egaztnpu-42098fc3b4a89ed3&c' },
  { id: 53547, title: '#侵袭式回忆 多抱抱你的男友吧 #情侣日常 #治愈系动画 #内容过于真实', views: 69, cover: 'https://p5.a.yximgs.com/upic/2025/03/21/23/BMjAyNTAzMjEyMzU5NTdfMjg1NTkzODUyNF8xNTk3ODY0NDE5MzJfMV8z_B99e49597c7a6670bdab5c0c02c5abc37.jpg?tag=1-1744982197-xpcwebsearch-0-popvckwzic-7a60d80dbb239326&c' },
  { id: 53539, title: '觉得像谁就艾特谁吧 #情侣日常 #治愈系动画', views: 66, cover: 'https://p3.a.yximgs.com/upic/2024/01/18/12/BMjAyNDAxMTgxMjA2MTRfMzg3MDM3MjI2NV8xMjIzOTAwNTA3MzlfMl8z_Bb136448ec76333a69a71e0014ae2281f.jpg?tag=1-1744982193-xpcwebsearch-0-mfxeg06hsf-69533492af12727f&c' },
  { id: 8371, title: '@海鸟短剧(O3xqwmsgy7tyevjc) 的精彩视频', views: 68, cover: 'https://p1.a.yximgs.com/upic/2025/04/14/10/BMjAyNTA0MTQxMDIyMzJfNDA1NzkwMTQ0NF8xNjE3MjA2ODgxMTJfMF8z_Baad444de458de37dc5171b55fa0415e8.jpg?tag=1-1744604760-xpcwebsearch-0-qgo6dosygb-a4ca22de5789b0a3&c' },
  { id: 53681, title: '#配音 #鬼灭之刃 #动漫 @快手粉条(O3xhcy6vhfzcu3qe) @快手热点(O3xddgkd5fav5if9) @快手用户1733684182531(O3x8tieq2gk9hk2y) 鬼灭合集来了！', views: 69, cover: 'https://p1.a.yximgs.com/upic/2024/07/10/10/BMjAyNDA3MTAxMDUzMDNfNTQ3NzE0NjgxXzEzNzI3MTAxNzk0N18xXzM=_B3922f0978d80fc2aef1af37de964da96.jpg?tag=1-1744982234-xpcwebsearch-0-8hfqcunymm-c159be6b6b907087&c' },
  { id: 32945, title: '#旅行中的风景 #旅途中的我 #我的旅行故事 #我们公司旅行可以不', views: 64, cover: 'https://p1.a.yximgs.com/upic/2025/04/13/09/BMjAyNTA0MTMwOTE2NDlfMTI5Njk4MTgwOF8xNjE2MjIzOTMzNjJfMl8z_B1a6aefc6ec2b97dad1aefb6a8a89a5f0.jpg?tag=1-1744708205-xpcwebsearch-0-fiugh8ikj0-bcdd491f410a2de4&c' },
  { id: 48448, title: '日常英语口语 英语日常口语 语言学习 零基础学英语', views: 67, cover: 'https://picsum.photos/300/80?random' },
  { id: 62775, title: '选择题万能通用口诀！让考试超简单！轻松拿高分～', views: 65, cover: 'https://picsum.photos/300/80?random' },
  { id: 8385, title: '宿舍唱跳版恋爱告急来啦，哈哈 #鞠婧祎恋爱告急', views: 66, cover: 'https://p4.a.yximgs.com/upic/2023/03/12/14/BMjAyMzAzMTIxNDAxMzRfMTEyMTUyMTM0OF85ODA1NTIwNTQ5Ml8xXzM=_B2d200bceb20d030d731dd8bb64cb99c1.jpg?tag=1-1744604763-xpcwebsearch-0-1nbk7aaxng-4ff8609bfd304c63&c' },
  { id: 8401, title: '那边的朋友 让我听到你们的声音！', views: 63, cover: 'https://p2.a.yximgs.com/upic/2019/09/21/18/BMjAxOTA5MjExODEyMjhfMzY1NzgyOV8xNzcwNzk4MjY5NF8xXzM=_Ba6d8a59e8421278458d37a5a8ac68fc9.jpg?tag=1-1744604766-xpcwebsearch-0-kwsifufk5m-1082cfcdcf92e5f6&clien' },
  { id: 32968, title: '#旅途中的我 #我的旅行故事 #旅行中的风景', views: 66, cover: 'https://p1.a.yximgs.com/upic/2025/02/02/09/BMjAyNTAyMDIwOTQxNTdfMTQwNDcyNjM1OV8xNTU3MTgwMzU4MDFfMl8z_B8ae7d90041e2ef13677dd7bd5ae43e07.jpg?tag=1-1744708209-xpcwebsearch-0-pms6l6gisn-9b62b972f6e55934&c' },
  { id: 8391, title: '自编自跳 小哥真行', views: 62, cover: 'https://p1.a.yximgs.com/upic/2025/04/12/20/BMjAyNTA0MTIyMDQ5NDJfNDEwNDYxODkzXzE2MTU4NjMzOTcwM18yXzM=_Ba2b60c5c0dce0e67249cfb8cf95de989.jpg?tag=1-1744604763-xpcwebsearch-0-lfzwsa6axc-360a146b4b8cf301&c' },
  { id: 8392, title: '后面直接唱错了😓', views: 62, cover: 'https://p3.a.yximgs.com/upic/2025/04/13/22/BMjAyNTA0MTMyMjUzMDdfMTc3Nzk5ODE1OV8xNjE2OTc2Nzc1NDZfMl8z_ccc_Bd9090c71b72927b112a280afe1b8d5cb.jpg?tag=1-1744604763-xpcwebsearch-0-pfosb4jrqy-6572e2e3961291' },
  { id: 48599, title: '为新家添置的每一件单品都是我精心挑选的，装修新家耗尽了我所有的心思，在自己能力范围之内想把所有喜欢的东西都买回来放在我的新家，或许这是对理想生活的一种期待。', views: 61, cover: 'https://picsum.photos/300/80?random' },
  { id: 48592, title: '香薰', views: 64, cover: 'https://picsum.photos/300/80?random' },
  { id: 57664, title: '如何练习深蹲 这几个动作要领你记住了吗？ #健身', views: 60, cover: 'https://p1.a.yximgs.com/upic/2019/12/12/11/BMjAxOTEyMTIxMTQ3MDlfMTYxNjM2NjU1NV8yMDQyNDY3NjM3NV8yXzM=_B16607a17558210829eef5119c27c93cc.jpg?tag=1-1744983441-xpcwebsearch-0-rceypsdjzy-7d07e82fc5de2012&c' },
  { id: 48604, title: '@红寺堡婉居软装(O3xhhjefkwte7sus) 的精彩视频', views: 64, cover: 'https://picsum.photos/300/80?random' },
  { id: 53578, title: '感情出现裂痕后还可以复原吗 #治愈系动画 #情侣日常 #内容启发分享计划', views: 58, cover: 'https://p1.a.yximgs.com/upic/2025/03/18/18/BMjAyNTAzMTgxODAxNTRfMjgxODU4NjM5MF8xNTk1NDcwNTYzMjdfMV8z_ccc_B45b9e81b915e85a74d4cb4d8104ca9f3.jpg?tag=1-1744982200-xpcwebsearch-0-lnfgp8wvyv-287ec598d4f66c' },
  { id: 19229, title: '初高中学习资料分享 学霸秘籍 高分秘籍', views: 64, cover: 'https://picsum.photos/300/80?random' },
  { id: 32978, title: '#旅途中的我 #旅行中的风景 #我的旅行故事 #出来放风', views: 60, cover: 'https://p2.a.yximgs.com/upic/2025/04/10/17/BMjAyNTA0MTAxNzQ3MTRfNDE0MDM2MzI2XzE2MTM4OTUyMzQxM18yXzM=_B5c22fc4723d46beba1e800d611dc4c7b.jpg?tag=1-1744708209-xpcwebsearch-0-urmsg5wffd-c81dd78aa28ea823&c' },
  { id: 32965, title: '#旅途中的我 #我的旅行故事 #旅行中的风景', views: 63, cover: 'https://p5.a.yximgs.com/upic/2025/04/13/07/BMjAyNTA0MTMwNzQ3NDhfODAyMTYxNzMxXzE2MTYxNzEwNzEyMV8yXzM=_Bf906746386a9c5df34c6133a7bf3d988.jpg?tag=1-1744708209-xpcwebsearch-0-nnif1eviaw-094c06fb953e2b3d&c' },
  { id: 48455, title: '英语口语练习 英语听力 英语日常口语 语言学习 零基础学英语', views: 58, cover: 'https://picsum.photos/300/80?random' },
  { id: 8407, title: '请你忘掉我的模样 ， 新雨摇舞蹈挑战 怎么唱情歌变妆挑战 😭', views: 65, cover: 'https://p3.a.yximgs.com/upic/2025/04/13/18/BMjAyNTA0MTMxODExMDlfOTUxNzMwNTUxXzE2MTY3MDIwNTE5MF8yXzM=_ccc_Bf120d31a0ea7a64c02e9ca6517c03bcc.jpg?tag=1-1744604766-xpcwebsearch-0-jvyphkommn-0069a2a1d59ade' },
  { id: 57045, title: '踩雷之后留下的家庭版健身好物！ #健身器材#减脂健身', views: 38, cover: 'https://p2.a.yximgs.com/upic/2024/06/17/17/BMjAyNDA2MTcxNzMyMzVfMTg5MzYwMjExNl8xMzUyMzY1NDg4OTRfMF8z_ccc_Bffe72b6eccc211896493a8cc0b9d4ac4.jpg?tag=1-1744983268-xpcwebsearch-0-apir0xgwtu-e15faf07bb5016' },
  { id: 10987, title: '摄影眼大挑战，如何构图才能拍出氛围感和故事感#自然风景手机随拍 #摄影技巧#培养摄影眼#摄影构图技巧', views: 37, cover: 'https://p4.a.yximgs.com/upic/2025/04/01/20/BMjAyNTA0MDEyMDE2MDlfMTcyNTQ1OTAxOF8xNjA2NjE3MDkwNDRfMF8z_ccc_B0350762625d3dd72bb2ceead12794224.jpg?tag=1-1744605003-xpcwebsearch-0-uhf8vxwfg8-ee4e1616a6ac0b' },
  { id: 12712, title: '盘点沙雕农村搞笑视频 #农村搞笑视频逗乐每一天 #纯属搞笑让大家乐一乐 #老爷们儿是不是这样的呢 幽默搞笑农村人分享快乐生活每一天', views: 36, cover: 'https://p1.a.yximgs.com/upic/2024/08/30/07/BMjAyNDA4MzAwNzQ5MzVfNDYzMzc1OTA1XzE0MjI2MTE5OTEwOF8yXzM=_Bb9881efb4f2b49514f166db4220601e3.jpg?tag=1-1744603431-xpcwebsearch-0-i2klbpnogu-269b574112b49fba&c' },
  { id: 35783, title: '“只有狗狗不在乎你在外面混的好或坏，当它见到你的时候，开心的像个孩子” #可爱狗 #修勾 #久别重逢', views: 35, cover: 'https://p4.a.yximgs.com/upic/2025/01/07/08/BMjAyNTAxMDcwODA4MDZfMTE4MjIzMjFfMTUzMjcxMzM4OTM0XzJfMw==_B5845f223b09f30056946b0adad142cf5.jpg?tag=1-1744709113-xpcwebsearch-0-katndc5w9i-89f15b79315ac29c&c' },
  { id: 36314, title: '#爱合拍 #合拍同框 #小动物叫声', views: 36, cover: 'https://p2.a.yximgs.com/upic/2024/04/15/20/BMjAyNDA0MTUyMDI0NDNfMjcxNzk5MDgxMF8xMjk5NDYwNzg5NTdfMV8z_Be8fc559dcb8ff6e4639d54d53f3e719f.jpg?tag=1-1744709299-xpcwebsearch-0-c3xmg9ydaj-54517774687f7e68&c' },
  { id: 24679, title: '厨房防潮没做好，到处发霉湿哒哒！这几个防潮小技巧 ，一个都不能少！【厨房防水层】地面全防水墙面要做30cm高水槽区域要做到水龙头往上10cm的高度❗【台盆】选台下盆，再搭配抽卡水龙头日常清扫hin方便【注意❗】很多宝宝会担心台下盆会掉，但是现在技术很发达啦，金牌刚拿到了『台盆下扣技术』的专利，台盆上站人都没关系~【挡水条】说过hin多遍~一定一定，要做挡水条前挡水5mm，后挡水4cm高度【水槽柜】底板单单用铝箔纸包覆是不够的🏻底板的材料可以用雪弗板（雪弗板是运用在轮船底部的材料）水槽柜四周可以用不锈钢板，更加防水防潮！【踢脚线】一般做10-15cm，更好地保护柜体呀！', views: 35, cover: 'https://p2.a.yximgs.com/upic/2021/06/15/15/BMjAyMTA2MTUxNTE2MDhfMTE4MDQ0MDE0NF81MTQ2MjYzODU3NF8wXzM=_Be838695e79ffd91a7b0882503ed83da7.jpg?tag=1-1744705497-xpcwebsearch-0-gky7ganh5g-ebb9d5d829341bb3&c' },
  { id: 33247, title: '#婚纱照 #变装 #氛围感', views: 35, cover: 'https://p1.a.yximgs.com/upic/2025/01/10/22/BMjAyNTAxMTAyMjAyMzlfMTEzMTU0NzcxM18xNTM1Njk2MTY1OThfMV8z_B415371d90fa438a61e28150ff4dcf144.jpg?tag=1-1744708265-xpcwebsearch-0-dng395jwce-6f3a5a8a5a3c1a0d&c' },
  { id: 18486, title: '这右眼跳得好啊～', views: 35, cover: 'https://picsum.photos/300/80?random' },
  { id: 58065, title: '#一个人的夜晚 #深夜的孤独 #情绪释放', views: 35, cover: 'https://p5.a.yximgs.com/upic/2024/11/20/02/BMjAyNDExMjAwMjI3NThfMjEyNjEyMzIxM18xNDkwNjYxNzg3OTVfMl8z_Ba597b96f3ca119e9afca41ae48f0abb0.jpg?tag=1-1744983561-xpcwebsearch-0-wjejbbvqoj-c7a7cd39180d2d28&c' },
  { id: 34396, title: '#爆款选题创作计划', views: 35, cover: 'https://p2.a.yximgs.com/upic/2025/04/15/14/BMjAyNTA0MTUxNDU4NDhfMjA5NzkwMDYyNF8xNjE4MDY0MDkzNTNfMV8z_B5373e51b126e18b348f5b322fe412eb8.jpg?tag=1-1744703102-xpcwebsearch-0-m6glxnqdzk-ca1aa2d122906c94&c' },
  { id: 9276, title: '家庭养的花卉绿植如何做到正确浇水呢？今天就教大家~ #分享养花知识 #花卉绿植 #养花达人 @快手服务号(O3xb7u4siymccsza) @快手平台帐号(O3xa3cpv8sghbu8m)', views: 37, cover: 'https://p4.a.yximgs.com/upic/2021/09/23/17/BMjAyMTA5MjMxNzAzMDVfMjM0NDE5NDA4Ml81NzcxNDA1ODc1OF8wXzM=_B03becc9f61a7c6507eebd00c6a47475a.jpg?tag=1-1744604721-xpcwebsearch-0-ieedyld2ne-b3f88af588f8745e&c' },
  { id: 8100, title: '你的美一缕飘散 去到我去不了的地方 #青花瓷 #吉他 #伴奏', views: 37, cover: 'https://p1.a.yximgs.com/upic/2021/06/23/20/BMjAyMTA2MjMyMDQ5MTZfNDIyMzg1NTlfNTE5MzgyNzIwMjZfMV8z_B7de501050bf38fdeece780a6c55af35a.jpg?tag=1-1744604652-xpcwebsearch-0-ovvwcma0wt-6b63622d8e0a1551&clien' },
  { id: 9408, title: '#真实还原系列 琪燕妈妈又出来嘚瑟了', views: 33, cover: 'https://p5.a.yximgs.com/upic/2024/04/01/19/BMjAyNDA0MDExOTIxMjFfMTc0NjA0NjA0M18xMjg3NTAyNjM3MDBfMV8z_Bd0e87855a84a0c27566331b5a2b96261.jpg?tag=1-1744603952-xpcwebsearch-0-cnysekha8g-d8575e7929b966ef&c' },
  { id: 13356, title: '是龙你就盘着，是虎你就卧着，做事低调，活的心安。#快手小剧场 #天天拍好剧#快手短剧冷启大赏', views: 33, cover: 'https://p1.a.yximgs.com/upic/2023/06/21/13/BMjAyMzA2MjExMzQ0MTFfMzUwMTQ0OTcwN18xMDYwNTQyMDUyMjNfMF8z_Bba2b7bd5e5560d7f801aacd5574c5942.jpg?tag=1-1744603526-xpcwebsearch-0-xbk8hnngzh-5de61527f179020c&c' },
  { id: 16033, title: '新手也能零失败，法式硬欧轻松搞定 #太原烘焙培训 #教学员技术配方毫无保留 #烘焙培训', views: 33, cover: 'https://p5.a.yximgs.com/upic/2025/03/19/22/BMjAyNTAzMTkyMjMyNDlfOTc4MDgyMTAxXzE1OTYzNTMwOTcwOV8yXzM=_B562718972581f9ddee42bae17e6354c3.jpg?tag=1-1744602063-xpcwebsearch-0-4dwlyajq9y-3dbd06247bade7ae&c' },
  { id: 37966, title: '难怪鲁迅当时逮谁喷谁，原来大家都有瓜，看完网友分享太震惊了 #网友神评', views: 36, cover: 'https://p2.a.yximgs.com/upic/2025/04/14/17/BMjAyNTA0MTQxNzMyMDlfNjk0ODEwOTdfMTYxNzQ1MjIyMTQ5XzJfMw==_B7e7685283afc66ea726e96c37335faf8.jpg?tag=1-1744703372-xpcwebsearch-0-dllpyybjh2-0f6b4c00a8aa4a9e&c' },
  { id: 17890, title: '@有妈真好🌈(O3x2p3e2fm2rmrva) 的精彩视频', views: 33, cover: 'https://picsum.photos/300/80?random' },
  { id: 33207, title: '终于用上这个BGM喽👰 #婚纱照拍摄花絮', views: 33, cover: 'https://p1.a.yximgs.com/upic/2024/01/02/17/BMjAyNDAxMDIxNzQ4NDNfMTk0MjM2MDhfMTIxMjEwNDI4MDY0XzFfMw==_B28a5b5f46f4d1d03617150c9a810ebaf.jpg?tag=1-1744708259-xpcwebsearch-0-vehlzgvzrb-ec275863748df1ae&c' },
  { id: 2857, title: '新生儿第一个月护理要点，新手妈妈们赶紧收藏起来！', views: 35, cover: 'https://picsum.photos/300/80?random' },
  { id: 65001, title: '星际探索 人类发现.首个外星访客 #宇宙未解之谜 #快影', views: 37, cover: 'https://p1.a.yximgs.com/upic/2021/03/16/18/BMjAyMTAzMTYxODI5NTJfMjIwNTg1NTc2M180NjEwMzE2MjM3NV8yXzM=_B8fd8b9fe713340394e5c8a8ebddb3045.jpg?tag=1-1744978384-xpcwebsearch-0-sdwovg4sfu-94ba9dfe9d377f80&c' },
  { id: 38731, title: '不要因为别人的一句话，就丢掉一整天的快乐，任何让你不舒服的关系，都要适可而止。 你不用去做别人认可的人，也不用去迎合任何人的标准，只要仰起头来做自己，自然会有人来爱你。 在心里种花，人生才不会荒芜来日方长，你要活成一束光，够光亮，就福泽四方;光微弱，就惠及身旁。 目光所及皆是所爱，心之所向皆是美好，愿你用力爱过，也用力生活着。 #今日书摘 #生活感悟 #领悟人生', views: 35, cover: 'https://p2.a.yximgs.com/upic/2023/09/21/20/BMjAyMzA5MjEyMDAwMzFfMjAxNjI2NDc1OF8xMTMyNjQ0NDQ2MDRfMl8z_B2f62fef72ad8da316a81331866e6e35c.jpg?tag=1-1744702630-xpcwebsearch-0-wcg10oxbdf-752e3c16a2bbf73c&c' },
  { id: 50042, title: '欢迎收看我的早间日常 #我的vlog日常 #日常vlog原创作品 #小洛很乖 #感谢快手平台记录每天的自己 关注小号 @xiao洛的碎碎念✨(O3xahjyprkp5ytzs)', views: 32, cover: 'https://p5.a.yximgs.com/upic/2025/03/27/18/BMjAyNTAzMjcxODI1MzVfNTQ2NDg2Mzk4XzE2MDIzNzIwOTM0NV8xXzM=_B3fbbe5a2187e65f8d749943fff4121da.jpg?tag=1-1744977210-xpcwebsearch-0-vj8vqu09g9-c68132255bc54bac&c' },
  { id: 41178, title: '美国中央司令部公布战斗机起飞视频证实美军对也门发动袭击', views: 32, cover: 'https://picsum.photos/300/80?random' },
  { id: 46482, title: '春蚕到死丝方尽，蜡炬成灰泪始干', views: 32, cover: 'https://picsum.photos/300/80?random' },
  { id: 41460, title: '中国书法简史 中国文字博大精深 书法历史 热爱书法支持正能量', views: 34, cover: 'https://picsum.photos/300/80?random' },
  { id: 35472, title: '男子给狗子洗澡，结果狗子一动不动还以为狗子…来源：@卢姥爷 #万万没想到 #狗狗能有什么坏心思 #澄江观察', views: 34, cover: 'https://p2.a.yximgs.com/upic/2024/05/11/16/BMjAyNDA1MTExNjUzMzNfNjczMzE4NjcwXzEzMjExMDUwMzIwMl8xXzM=_B46cc330ff08876ba2d0d2a180a5407b7.jpg?tag=1-1744709038-xpcwebsearch-0-oeawckcy30-6b5ec999f33af35c&c' },
  { id: 55813, title: '饭后消食操，饭后扭一扭，摆脱大肚腩 #健身挑战 #健身操 #我的健身挑战', views: 34, cover: 'https://p1.a.yximgs.com/upic/2024/05/09/18/BMjAyNDA1MDkxODIwNTFfMTQwODQzMDA1OF8xMzE5NjYyMDMzODhfMV8z_B2406451f4e17ae23a24b4fffdacdcd2d.jpg?tag=1-1744982893-xpcwebsearch-0-fia6lpw8gj-d435c52a1c44ebff&c' },
  { id: 33259, title: '和好朋友一起变装吧 #姐妹合拍 #梁山V1婚纱摄影', views: 33, cover: 'https://p2.a.yximgs.com/upic/2024/11/30/17/BMjAyNDExMzAxNzU0MDJfMjA4MTgwNjYxNF8xNDk5NjI1MTkyOThfMV8z_B67f934640f2dc5e87c7879ce5b15e225.jpg?tag=1-1744708265-xpcwebsearch-0-2gttpetei5-151d3731087f9b6a&c' },
  { id: 55882, title: '我要拉伸100遍！！拉伸完太舒服了！不愧是mizi姐的动态拉伸！全程站立轻松拉完！运动必备！姐妹们收藏起来～ #运动拉伸', views: 33, cover: 'https://p2.a.yximgs.com/upic/2024/08/09/15/BMjAyNDA4MDkxNTE1MTJfMTY3ODg3MjM5MV8xNDAyNTMxNDE2NThfMV8z_Bcb5ccd34675453fe8db89c8c60647af6.jpg?tag=1-1744982915-xpcwebsearch-0-tgkxiwpcbk-a8c98f3494c4b45a&c' },
  { id: 57708, title: '@小路..。(O3x2h27rhg8a686k) 的精彩视频', views: 35, cover: 'https://p4.a.yximgs.com/upic/2021/10/11/17/BMjAyMTEwMTExNzU4NTNfMTE1NzU5NTZfNTg4NzczODI1NzdfMV8z_Bebe68bc4f84907e2508def31f27d6eee.jpg?tag=1-1744983457-xpcwebsearch-0-ph7dryrvjk-7481b624b7144f42&clien' },
])

// 分页控制
const pageSize = 4
const currentLikePage = ref(1)
const currentViewPage = ref(1)

const paginatedLikeRank = computed(() => {
  const start = (currentLikePage.value - 1) * pageSize
  return likeRank.value.slice(start, start + pageSize)
})

const paginatedViewRank = computed(() => {
  const start = (currentViewPage.value - 1) * pageSize
  return viewRank.value.slice(start, start + pageSize)
})

// 跳转播放页
function viewVideo(id) {
  router.push(`/player/${id}`)
}

const recommendedHistory = new Set()

function refresh() {
  // 候选视频 = allVideos 中尚未推荐过的
  let candidates = allVideos.value.filter(v => !recommendedHistory.has(v.id))

  // 如果剩下不足4个，就清空历史，重新开始（实现循环）
  if (candidates.length < 4) {
    console.warn("剩余视频不足，重置推荐历史以支持循环刷新")
    recommendedHistory.clear()
    candidates = [...allVideos.value]
  }

  // 加权选择 4 个视频
  const weights = candidates.map((_, index) => 1 / (index + 1))
  const selected = []
  const usedIndexes = new Set()

  while (selected.length < 4) {
    const totalWeight = weights.reduce((sum, w, i) => usedIndexes.has(i) ? sum : sum + w, 0)
    let r = Math.random() * totalWeight

    for (let i = 0; i < weights.length; i++) {
      if (usedIndexes.has(i)) continue
      r -= weights[i]
      if (r <= 0) {
        selected.push(candidates[i])
        usedIndexes.add(i)
        recommendedHistory.add(candidates[i].id) // 记录推荐历史
        break
      }
    }
  }

  // 更新推荐
  recommendVideos.value = selected
}

function search() {
  router.push(`/videos?keyword=${searchKeyword.value}`)
}


function getTopBadge(index) {
  if (index === 0) return '🏆'
  if (index === 1) return '🥈'
  if (index === 2) return '🥉'
  return index + 1
}

function loadUserInfo() {
  const userId = localStorage.getItem('userId')
  if (userId) {
    username.value = `用户${userId}`
  }
}

async function loadRecommend() {
  const userId = localStorage.getItem('userId')
  if (!userId) return

  try {
    const res = await api.get(`/user/${userId}/recommend`)
    const raw = res.data || []

  // 去重 + 过滤无效项
  const uniqueMap = new Map()
  raw.forEach(v => {
    if (v.id && v.title && v.cover && !uniqueMap.has(v.id)) {
      uniqueMap.set(v.id, v)
    }
  })

  allVideos.value = Array.from(uniqueMap.values())
  recommendVideos.value = [...allVideos.value]
  .sort(() => Math.random() - 0.5)  // 打乱顺序
  .slice(0, 4)                      // 取前 4 个
  } catch (error) {
    console.error('加载推荐失败', error)
  }
}

function goProfile() {
  const userId = localStorage.getItem('userId')
  if (!userId) {
    alert('请先登录')
    return
  }
  router.push(`/user/${userId}`)
}

function logout() {
  localStorage.removeItem('userId')
  router.push('/login')
}

onMounted(() => {
  loadUserInfo()
  loadRecommend()
})


</script>

<style scoped>
.home-container {
  padding: 20px;
}

.top-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.search-input {
  width: 300px;
}

.main-content {
  margin-top: 20px;
}

.section-card {
  padding: 20px;
}

.recommend-list {
  margin-top: 10px;
}

.video-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px;
  border-radius: 10px;
  overflow: hidden;
  text-align: center
}

.video-cover {
  width: 300px;
  height: 180px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 10px;
  text-align: center;
}

.video-info {
  width: 100%;
}

.video-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  padding: 0 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}


.rank-item {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding: 8px;
  border-radius: 10px;
  cursor: pointer;
  transition: background 0.3s;
}

.rank-item:hover {
  background: #f0f2f5;
}

.rank-cover {
  width: 80px;
  height: 50px;
  object-fit: cover;
  border-radius: 8px;
  margin-right: 12px;
}

.rank-info {
  flex: 1;
}

.rank-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rank-sub {
  font-size: 13px;
  color: #999;
  margin-top: 4px;
}


.home-container {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.top-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}

.search-input {
  width: 300px;
}

.main-content {
  margin-top: 20px;
}

.section-card {
  margin-bottom: 24px;
  padding: 24px;
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.section-title {
  font-size: 22px;
  font-weight: bold;
  margin-bottom: 20px;
  color: #333;
}

.recommend-list {
  margin-top: 10px;
}

.video-card {
  padding: 10px;
  cursor: pointer;
  transition: all 0.3s;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #fefefe;
}

.video-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  background: #f0f2f5;
}

.video-cover {
  width: 300px;
  height: 180px;
  object-fit: cover;
  border-radius: 10px;
  margin-bottom: 12px;
}

.video-info {
  width: 100%;
  text-align: center;
}

.video-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rank-item {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding: 8px;
  border-radius: 10px;
  cursor: pointer;
  transition: background 0.3s;
}

.rank-item:hover {
  background: #f0f2f5;
}

.rank-cover {
  width: 80px;
  height: 50px;
  object-fit: cover;
  border-radius: 8px;
  margin-right: 12px;
}

.rank-info {
  flex: 1;
}

.rank-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.rank-sub {
  font-size: 13px;
  color: #999;
  margin-top: 4px;
}

.pagination {
  margin-top: 20px;
  text-align: center;
}


.rank-item {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding: 8px;
  border-radius: 10px;
  cursor: pointer;
  transition: background 0.3s;
  position: relative;
}

.rank-item:hover {
  background: #f0f2f5;
}

.rank-badge {
  width: 24px;
  height: 24px;
  font-size: 16px;
  font-weight: bold;
  color: #fff;
  background: linear-gradient(135deg, #409eff, #66b1ff);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  position: absolute;
  left: -12px;
  top: 50%;
  transform: translateY(-50%);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

.rank-cover {
  width: 80px;
  height: 50px;
  object-fit: cover;
  border-radius: 8px;
  margin-left: 20px;
  margin-right: 12px;
}

.rank-info {
  flex: 1;
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
}

.rank-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.rank-sub {
  font-size: 13px;
  color: #999;
  margin-top: 4px;
}

.pagination {
  margin-top: 20px;
  text-align: center;
}



.user-info {
  margin-left: auto; /* ✨ 让它自动推到最右 */
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 16px;
  color: #333;
  background: #fff;
  padding: 6px 12px;
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.user-info span {
  font-weight: 500;
}

.user-info .el-button {
  height: 30px;
  line-height: 30px;
  padding: 0 12px;
  font-size: 14px;
}
</style>

