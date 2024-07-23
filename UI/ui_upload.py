import os
from typing import List

import streamlit as st
from api import send_request
from item import Query, Dialog, UploadPacket
import pandas as pd

if "dir_path" not in st.session_state:
    st.session_state["dir_path"] = os.getcwd()

_query = st.chat_input("Query")
_intent = st.selectbox('Intent', ['问答', '表情包'])
_dir_path = st.text_input("Directory Path", value=st.session_state["dir_path"])
if os.path.isdir(_dir_path):
    st.session_state["dir_path"] = _dir_path
    files = [os.path.join(_dir_path, file) for file in os.listdir(_dir_path)]
else:
    files = []
st.table(files)

_upload_btn = st.button("Upload")

if _query:
    resp = send_request(
            "embedding/query",
            port=8002,
            json=Query(query=_query,intent=_intent).model_dump(),
            method="POST",
        ).json()
    st.write(f'{_intent}:\n')
    if _intent == '问答':
        records = [{'source': r['metadata']['source'], 'page_content': r['page_content']} for r in resp]
        df = pd.DataFrame.from_records(records)

        st.table(df)

    
    elif _intent == '表情包':
    
            # "image:\n",
            # *[f'![{im["page_content"]}]({im["metadata"]["source"]}) "{im["page_content"]}")\n' for im in resp]
        grid = st.columns(2)
        cnt = 0
        for im in resp:
            im_uri = f'http://localhost:8002/image?file_path={im["metadata"]["source"]}'
            # st.markdown(f'![{im["page_content"]}]({im_uri})', unsafe_allow_html=True)
            grid[cnt].image(im_uri, use_column_width=True)
            cnt+=1
            cnt = cnt%2

if _upload_btn:
    upload_packet = UploadPacket(dir_path=_dir_path)

    st.write("response from server:")
    if _intent == '问答':
        st.write(
            "pdf:\n",
            send_request("database/pdf", port=8002, json=upload_packet.model_dump(), method="POST").json()
        )
    
    elif _intent == '表情包':
        st.write(
            "image:\n",
            send_request("database/img", port=8002, json=upload_packet.model_dump(), method="POST").json()
        )
