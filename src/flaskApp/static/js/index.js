const vm = new Vue({
    el: '#app',
    delimiters: ['{[', ']}'],
    data: {
        dataLs: []
    },
    methods: {
        prepare(file) {
            this.$message({
                message: '上传并解析视频中......',
                type: 'warning',
                duration: 2000,
                offset: 100
            });
        },
        finish(response, file, fileList) {
            this.$message({
                message: '视频帧计数分析成功!',
                type: 'success',
                duration: 2000,
                offset: 100
            });
            this.dataLs = response.dataLs;
        }
    }
})