import os
ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
import json
sys.path.append(f'{ROOT}/../model/')
from embedding_model import EmbeddingModel
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from shutil import rmtree

@st.cache_resource
def load_model(device_name, model_name, path, collection_name='team_d'):
    model = EmbeddingModel(device_name, model_name)
    retriever = Chroma(collection_name, model, persist_directory=path).as_retriever()
    return model, retriever

@st.cache_data
def get_splitter(chunk_size, chunk_overlap, keep_separator, separators, strip_whitespace):
    return RecursiveCharacterTextSplitter(  # 词语切分分词器
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=keep_separator,
        strip_whitespace=strip_whitespace,
        separators=json.loads(separators),
    )

with st.expander("模型配置", expanded=False):
    device_name = st.selectbox("Select device", ["cuda:0", "cuda:2", "cuda:4"])
    model_name = st.selectbox("Select model", ["Mistral-7B-Instruct-v0.1"])
    vdb_path = st.text_input("Enter path to VDB", value=(os.path.abspath(f"{ROOT}/../chromadb/")))
    search_type = st.selectbox('搜索类型', ['similarity', 'mmr'], index=1)
os.makedirs(vdb_path, exist_ok=True)

model, retriever = load_model(model_name, device_name, vdb_path)

    
upload_tab, query_tab, splitter_tab = st.tabs(["Upload", "Query", "Splitter"])

with splitter_tab:
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
        splitter = get_splitter(  # 词语切分分词器
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=keep_separator,
            strip_whitespace=strip_whitespace,
            separators=separators,
        )

        splits = splitter.split_text(all_text)
        
        st.write(f"分割结果：")
        for s in splits:
            st.write(s)
            st.divider()


with upload_tab:
    dir_path = st.text_input("上传文件目录路径", value='/home/team_d/data/')
    if os.path.isdir(dir_path):
        st.table(os.listdir(dir_path))
    
    cols = st.columns(2)
    if cols[0].button("Upload",use_container_width=True):
        for file in os.listdir(dir_path):
            if file.endswith('.pdf'):
                file_path = os.path.join(dir_path, file)
                docs = PyPDFLoader(file_path).load_and_split(splitter)
                
                st.success(retriever.add_documents(docs))
    
    if cols[1].button("Delete all documents",use_container_width=True):
        rmtree(vdb_path)
    
with query_tab:
    query = st.chat_input("Enter query")


if query and model and retriever:
    st.write(retriever.invoke(query, search_type=search_type))