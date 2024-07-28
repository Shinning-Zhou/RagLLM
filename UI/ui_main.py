import streamlit as st
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

pages = [
    st.Page(f"{ROOT}/ui_chat.py", title="Chat"),
    st.Page(f"{ROOT}/ui_upload.py", title="Embedding"),
    st.Page(f"{ROOT}/ui_embedding.py", title="Embedding DEBUG")
]

pg = st.navigation(pages)
pg.run()
