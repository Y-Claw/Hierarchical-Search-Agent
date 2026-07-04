import os
import json
import argparse  # Import argparse module
from flask import Flask, render_template, send_from_directory, request
import webbrowser
from threading import Timer

app = Flask(__name__)

# Add argparse to handle command-line arguments
parser = argparse.ArgumentParser(description='Run the Flask application.')
parser.add_argument('--save_path', type=str, default="results/example_run", help='Path to save JSON files')
parser.add_argument('--port', type=int, default=8088, help='Port to run the Flask application on')
args = parser.parse_args()

save_path = os.path.abspath(args.save_path)  # Use the argument value for save_path

# 创建保存目录
# os.makedirs(save_path, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list')
def list_files():
    files = [f for f in os.listdir(save_path) if f.endswith('.json')]
    return {"files": files}

@app.route('/data/<path:filename>')
def get_data(filename):
    return send_from_directory(save_path, filename)

@app.route('/key/<path:filename>/<key>')
def show_key(filename, key):
    # 读取 JSON 文件
    filepath = os.path.join(save_path, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 检查 key 是否存在
    if key not in data:
        return f"Key '{key}' not found in {filename}", 404
    
    # 渲染 key 的详细页面
    return render_template('key.html', filename=filename, key=key, value=data[key])

@app.route('/next')
def next_file():
    files = [f for f in os.listdir(save_path) if f.endswith('.json')]
    # 获取当前文件的索引
    current_file = request.args.get('current', files[0])
    try:
        current_index = files.index(current_file)
    except ValueError:
        current_index = -1

    # 获取下一个文件
    next_index = (current_index + 1) % len(files)
    next_file = files[next_index]

    # 读取下一个文件的数据
    filepath = os.path.join(save_path, next_file)
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data

def open_browser():
    webbrowser.open_new(f'http://localhost:{args.port}')

if __name__ == '__main__':
    # 生成示例数据（测试用）
    sample_data = {
        "sample_1.json": {"name": "测试1", "data": [1, 2, 3], "info": {"age": 20}},
        "sample_2.json": {"name": "测试2", "data": {"a": 1, "b": 2}}
    }
    for filename, data in sample_data.items():
        with open(os.path.join(save_path, filename), 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=args.port)  # 绑定到所有网络接口
