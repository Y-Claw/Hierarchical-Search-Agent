import os
import json
import argparse
from flask import Flask, render_template, send_from_directory, request
import webbrowser
from threading import Timer

app = Flask(__name__)

# Add argparse to handle command-line arguments
parser = argparse.ArgumentParser(description='Run the Flask application.')
parser.add_argument('--dir1', type=str, required=True, help='First directory to compare')
parser.add_argument('--dir2', type=str, required=True, help='Second directory to compare')
parser.add_argument('--port', type=int, default=8088, help='Port to run the Flask application on')
args = parser.parse_args()

dir1 = os.path.abspath(args.dir1)
dir2 = os.path.abspath(args.dir2)

@app.route('/')
def index():
    return render_template('compare.html')

@app.route('/list/<int:dir_num>')
def list_files(dir_num):
    directory = dir1 if dir_num == 1 else dir2
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return {"files": files}

@app.route('/data/<int:dir_num>/<path:filename>')
def get_data(dir_num, filename):
    directory = dir1 if dir_num == 1 else dir2
    return send_from_directory(directory, filename)

def open_browser():
    webbrowser.open_new(f'http://localhost:{args.port}')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=args.port)
