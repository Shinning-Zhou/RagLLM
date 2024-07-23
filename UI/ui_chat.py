import streamlit as st

from item import Dialog
from api import *
from langchain_core.documents import Document
from pydantic import BaseModel
import json

with st.sidebar:
    _session_id = st.text_input(
        "Enter a session ID", value=st.session_state.get("session_id", "test")
    )
    _clear_btn = st.button("Clear session", use_container_width=True)
    _is_responsing = st.status("Waiting for input...", state="complete")


_message_box = st.container(border=True)


for chat_type, json_load in load_dialogs(_session_id):
    _message_box.chat_message(chat_type).write(json_load)

im_cnt = 0
im_grid = st.columns(2)

try:
    def callback(chunk: str):
        global im_cnt
        data = json.loads(chunk)
        source = data['metadata']['source']
        ftype = source.split('.')[-1]
        if ftype in ['png', 'jpg', 'jpeg', 'gif']:
            im_uri = f'http://localhost:8002/image?file_path={source}'
            im_grid[im_cnt % 2].image(im_uri, use_column_width=True)
            im_cnt+=1

    if _message := st.chat_input("你好！", disabled=_is_responsing.state == "running"):
        _message_box.chat_message("user").write_stream(stream_str(_message))
        _is_responsing.update(state="running", label="Responsing...")
        response = send_msg(
            Dialog(
                session_id=_session_id,
                message=_message,
            ),
        )

        _message_box.chat_message("assistant").write_stream(stream_resp(response, callback=callback))

        _is_responsing.update(state="complete", label="Waiting for input...")
        _is_responsing.update(state="complete", label="Waiting for input...")

    if _clear_btn:
        _is_responsing.update(state="running", label="Cleaning session...")
        clean_session(_session_id)
        _is_responsing.update(state="complete", label="Waiting for input...")
        st.rerun()
except Exception as e:
    st.write(e)
