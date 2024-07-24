## 使用方法
```bash
conda activate team_d # 激活环境
cd /home/team_d/src # 进入项目目录
/bin/bash start.sh # 启动服务器与前端框架， ctrl + c 退出
```
启动服务后，打开浏览器访问 http://localhost:8000/ 即可进入聊天界面。（如果不行，需要进行端口转发，将远程机的 8000 端口转发到本地机）



## 目录结构
- UI/ # 前端框架
    - api.py # 前端接口
    - item.py # 数据结构定义
    - ui_main.py # 前端主界面
    - ui_chat.py # 聊天界面
    - ui_upload.py # 文件上传界面

- model/ # 模型代码
    - chat_model.py # 聊天模型
    - embedding_model.py # 文本嵌入模型

- *_svc.py # 后端服务代码
    - chat_svc.py # 聊天服务
    - db_svc.py # 数据库服务(包含向量嵌入功能)

- model_download.sh # 模型下载脚本

- start.sh # 启动脚本

## 使用的库
- streamlit 前端框架
- fastapi 后端框架
- langchain llm调用框架
- langchain-community llm调用框架
- chromadb 向量数据库，同faiss
- pypdf pdf文件解析器

