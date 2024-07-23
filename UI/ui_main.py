import streamlit as st
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

pages = [
    st.Page(f"{ROOT}/ui_chat.py", title="Chat"),
    st.Page(f"{ROOT}/ui_upload.py", title="Embedding"),
]

pg = st.navigation(pages)
pg.run()
