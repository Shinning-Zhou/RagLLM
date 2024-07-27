import streamlit as st
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

with st.sidebar:
    page_num = st.number_input("page_num", value=0, min_value=0, max_value=10, step=1)
    chunk_size = st.slider("chunk_size", value=512, min_value=1, max_value=1024, step=1)
    chunk_overlap = st.slider("chunk_overlap", value=0, min_value=0, max_value=chunk_size, step=1)
    keep_separator = st.selectbox("保留分隔符", options=[True, False, 'start', 'end'], index=3)
    separators = st.text_area("请输入分隔符", value=json.dumps(["\n\n", "。", "!", "！", "?", "？", ";", "；", "."], ensure_ascii=False))
    strip_whitespace = st.toggle("是否去除空白字符", value=True)
    
cols = st.columns(2)
with cols[0]:
    file_path = st.text_input("请输入文件路径", value="/home/team_d/data/history.pdf")

    pdf_loader = PyPDFLoader(file_path)
    text = pdf_loader.load()
    st.write(f"原始文本：")
    all_text = "\n\n".join([doc.page_content for doc in text if doc.metadata["page"] == page_num])
    st.write(all_text)

with cols[1]:
    splitter = RecursiveCharacterTextSplitter(  # 词语切分分词器
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=keep_separator,
        strip_whitespace=strip_whitespace,
        separators=json.loads(separators),
    )

    splits = splitter.split_text(all_text)
    
    st.write(f"分割结果：")
    for s in splits:
        st.write(s)
        st.divider()
