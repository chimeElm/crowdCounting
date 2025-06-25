import cv2
import os


def extract_frames(video_path='static/video.mp4', output_folder='static/img', interval_sec=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            os.remove(file_path)
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print('无法打开视频文件:', video_path)
        return
    # 获取视频帧率
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # 计算每隔多少帧提取一张图像
    interval_frames = int(fps * interval_sec)
    frame_count = 0
    sec_count = 0
    sec_list = []
    # 读取视频帧
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # 每隔 interval_frames 帧保存一张图像
        if frame_count % interval_frames == 0:
            output_path = os.path.join(output_folder, f'{sec_count}.jpg')
            sec_list.append(sec_count)
            cv2.imwrite(output_path, frame)
            print(f'已保存帧 {output_path}')
            sec_count += interval_sec
        frame_count += 1
    # 释放视频捕获对象
    video_capture.release()
    return sec_list


if __name__ == '__main__':
    print(extract_frames())
