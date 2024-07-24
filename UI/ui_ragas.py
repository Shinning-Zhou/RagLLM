from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.langchain.evalchain import RagasEvaluatorChain
from api import send_msg, stream_resp
from item import Dialog
import streamlit as st

if st.button("Send"):
    response = send_msg(
        Dialog(
            session_id="test",
            message="问答：中国近代革命的社会原因是什么？",
        ),
    )

    result = ""
    for ch in stream_resp(response):
        result += ch
    st.write(result)
    
    # create evaluation chains
    faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
    answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
    context_rel_chain = RagasEvaluatorChain(metric=context_precision)
    context_recall_chain = RagasEvaluatorChain(metric=context_recall)

    # 获取结果
    eval_result = faithfulness_chain(result)
    st.write(eval_result)
    eval_result = answer_rel_chain(result)
    st.write(eval_result)
    eval_result = context_rel_chain(result)
    st.write(eval_result)
    eval_result = context_recall_chain(result)
    st.write(eval_result)
