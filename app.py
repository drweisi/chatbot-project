# app.py - 本地开发启动文件
from api.index import app

if __name__ == '__main__':
    app.run(debug=True, port=8888)
    