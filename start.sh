#!/bin/bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=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
export TZ='Asia/Shanghai'
DATE=$(date +'%Y%m%d_%H%M%S')

# 启动fastapi应用，日志写入到log_<name>.txt文件中
mkdir -p logs
log_chat="logs/${DATE}_log_rag1_chat.txt"
uvicorn chat_svc:app --port 8001 > "$log_chat" 2>&1 &
echo "RAG-1 chat started"

# 启动数据库应用，日志写入到log_<name>.txt文件中
log_db="logs/${DATE}_log_rag1_database.txt"
uvicorn db_svc:app --port 8002 > "$log_db" 2>&1 &
echo "RAG-1 database started"


streamlit run UI/ui_main.py --server.port 8000

# 获取进程PID 位于log文件第一行中，用[]包裹起来, 只提取数字
# PID_8001=$(head -n 1 "$log_chat" | awk '{print}' | sed 's/[^0-9]//g')
# PID_8002=$(head -n 1 "$log_db" | awk '{print}' | sed 's/[^0-9]//g')
PID_8001=$(pgrep -f "uvicorn chat_svc:app --port 8001")
PID_8002=$(pgrep -f "uvicorn db_svc:app --port 8002")


kill -9 $PID_8001
echo "RAG-1 chat stopped"

kill -9 $PID_8002
echo "RAG-1 database stopped"