"""
前端页面用到的接口函数
"""
import json
import requests
import time


def stream_str(s):
    """
    字符串流式传输，没有打字间隔
    """
    for char in s:
        yield char


def stream_resp(resp: requests.Response, typing_delay=0.02, callback=None):
    """
    响应流式传输
    """
    for chunk_full in resp.iter_content(chunk_size=1024, decode_unicode=True):  # 1024字节一块，当chunk_size偏小时会将返回结果切分
        split = chunk_full.split("|", 1)  # 将"{cate}|{chunk}"格式的字符串拆分
        if len(split) == 1:  # 有时候会收到空字符串，这时切分后长度为1
            cate = split[0]
            chunk = ""
        else:
            cate, chunk = split
        print(f"{cate}: {chunk}")

        if cate == "context":   # context表示引用的文献
            if callback:
                callback(chunk)
            continue
        elif cate != "answer":  # answer表示回答的文本
            continue

        # yield chunk
        for char in chunk:
            yield char
            time.sleep(typing_delay)  # 按字符间隔延时，模拟打字的效果


def send_request(path, port: int = 8000, json={}, method="GET", files=None):
    """
    发送请求的封装
    """
    url = f"http://localhost:{port}/{path}"

    headers = {
        "User-Agent": "apifox/1.0.0 (https://www.apifox.cn)",
        "Content-Type": "application/json",
    }
    return requests.request(
        method, url, headers=headers, files=files, json=json, stream=True
    )


def send_msg(model):
    """
    发送消息的接口
    """
    return send_request("rag/", port=8001, json=model.model_dump(), method="POST")


def session_list():
    """查询聊天记录的统计结果"""
    return send_request(f"session", port=8001).content.decode("utf-8")


def session_show(session_id):
    """查询单个聊天记录详情"""
    return send_request(f"session/{session_id}", port=8001).json()


def clean_session(session_id):
    """清空聊天记录"""
    # body = send_request(f"chat/clear/{user_id}?partition={session_id}").json()
    # return f'{body["code"]}: {body["message"]}'
    return send_request(f"session/{session_id}", port=8001, method="DELETE").json()


def load_dialogs(session_id):
    """加载聊天记录，返回每一句"""
    dialogs = session_show(session_id)
    if dialogs != {}:
        for dialog_json in dialogs[session_id]:
            ind, json_load = dialog_json[0], json.loads(dialog_json[1])
            yield json_load["type"], json_load["data"].get(
                "content", "KeyError: 'content'"
            )
