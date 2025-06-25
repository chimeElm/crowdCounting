<template>
	<view>
		<camera device-position="back" flash="off" resolution="high" @error="error"
			style="width: 100%; height: 500upx;">
			<cover-image src="../../static/scan-frame/scan-img.png" class="scan-img"></cover-image>
		</camera>
		<button type="primary" @click="switchs">{{isRecord ? '停止计数' : '开始计数'}}</button>
		<image mode="widthFix" class="photos-box"
			:src="start ? 'http://127.0.0.1:5000/static/teme/' + ts + '.jpg' : '../../static/scan-frame/mask.jpg'">
		</image>
		<view class="count">{{count === '' ? '待估计' : '当前估计人数：' + count + '人'}}</view>
	</view>
</template>

<script>
	export default {
		data() {
			return {
				count: '',
				ts: '',
				isRecord: false,
				intervalTimer: null,
				start: false
			}
		},
		methods: {
			switchs() {
				if (this.isRecord) {
					clearInterval(this.intervalTimer);
					this.count = '';
				} else {
					this.intervalTimer = setInterval(() => {
						this.takePhotos();
					}, 500);
					this.start = true;
				}
				this.isRecord = !this.isRecord;
			},
			takePhotos() {
				const ctx = uni.createCameraContext();
				ctx.takePhoto({
					quality: 'high',
					success: (res) => {
						this.src = res.tempImagePath;
						uni.uploadFile({
							url: 'http://127.0.0.1:5000/uploadImg',
							filePath: res.tempImagePath,
							name: 'file',
							success: (uploadFileRes) => {
								let data = JSON.parse(uploadFileRes.data);
								console.log(data);
								if (data.ts > this.ts) {
									this.count = data.count;
									this.ts = data.ts;
								} else {
									console.log('drop data');
								}
							}
						});
					}
				});
			},
			error(e) {
				console.log(e.detail);
			}
		}
	}
</script>

<style>
	.scan-img {
		opacity: 0.4;
		width: 100%;
		height: 500upx;
	}

	.photos-box {
		width: 100%;
		height: 500upx;
	}

	button {
		width: 70%;
		height: 70upx;
		margin: 15upx auto;
		font-size: 30upx;
		line-height: 70upx;
		border-radius: 15px;
	}

	.count {
		width: 70%;
		height: 70upx;
		margin: 15upx auto;
		font-size: 30upx;
		line-height: 70upx;
		text-align: center;
		background: #1972ad;
		color: #fff;
		border-radius: 15px;
	}
</style>