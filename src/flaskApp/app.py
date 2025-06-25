from flask import Flask, render_template, request, jsonify
from extract import extract_frames
from count import estimate_density_map
import time
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.debug = True

for filename in os.listdir('static/temp'):
    file_path = os.path.join('static/temp', filename)
    os.remove(file_path)
for filename in os.listdir('static/teme'):
    file_path = os.path.join('static/teme', filename)
    os.remove(file_path)


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/upload')
def upload_file():
    try:
        file = request.files['file']
        file.save('static/video.mp4')
        print('video save success')
        sec_list = extract_frames('static/video.mp4', 'static/img', 1)
        data_list = []
        for sec in sec_list:
            num = estimate_density_map(f'static/img/{sec}.jpg')
            data_list.append({
                'name': f'第 {sec} 秒',
                'img': f'/static/img/{sec}.jpg',
                'res': f'/static/res/{sec}.jpg',
                'count': f'共 {num} 人'
            })
        return jsonify({'msg': 'success', 'dataLs': data_list}), 200
    except Exception as e:
        print(e)
        return jsonify({'msg': str(e), 'dataLs': []}), 400


@app.post('/uploadImg')
def upload_img():
    try:
        print(request.files)
        ts = int(time.time() * 1000)
        path = f'static/temp/{ts}.jpg'
        img = request.files['file']
        img.save(path)
        num = estimate_density_map(path)
        print('处理时间:', int(time.time() * 1000) - ts, 'ms')
        return jsonify({'msg': 'success', 'count': num, 'ts': ts}), 200
    except Exception as e:
        print(e)
        return jsonify({'msg': str(e), 'count': 0, 'ts': 0}), 400


if __name__ == '__main__':
    app.run('127.0.0.1', 5000)
