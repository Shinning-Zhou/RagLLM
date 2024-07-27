import streamlit as st

from item import Dialog
from api import *
from langchain_core.documents import Document
from pydantic import BaseModel
import json

with st.sidebar:  # 侧边栏
    _session_id = st.text_input(
        "session_id", value=st.session_state.get("session_id", "test")
    )
    _clear_btn = st.button("clear session", use_container_width=True)
    _is_responsing = st.status("wating for input...", state="complete")


_message_box = st.container(border=True)  # 聊天区域


for chat_type, json_load in load_dialogs(_session_id):  # 加载历史对话
    _message_box.chat_message(chat_type).write(json_load)  # 绘制聊天框并显示历史对话

im_cnt = 0
im_grid = st.columns(2)

try:

    def callback(chunk: str):  # 回调函数，用于处理图片消息
        global im_cnt  # 全局变量，用于给图片分2栏
        data = json.loads(chunk)
        source = data["metadata"]["source"]
        ftype = source.split(".")[-1]

        if ftype in ["png", "jpg", "jpeg", "gif"]:  # 判断图片
            im_uri = f"http://localhost:8002/image?file_path={source}"  # 访问图片地址，get方法
            im_grid[im_cnt % 2].image(im_uri, use_column_width=True)
            im_cnt += 1

    if _message := st.chat_input("你好！", disabled=_is_responsing.state == "running"):  # 输入消息，信息为_message，:=表示赋值给变量_message
        _message_box.chat_message("user").write_stream(stream_str(_message))
        _is_responsing.update(state="running", label="Responsing...")
        response = send_msg(
            Dialog(
                session_id=_session_id,
                message=_message,
            ),
        )

        _message_box.chat_message("assistant").write_stream(
            stream_resp(response, callback=callback)
        )

        _is_responsing.update(state="complete", label="Waiting for input...")

    if _clear_btn:
        _is_responsing.update(state="running", label="Cleaning session...")
        clean_session(_session_id)
        _is_responsing.update(state="complete", label="Waiting for input...")
        st.rerun()
except Exception as e:
    st.write(e)
