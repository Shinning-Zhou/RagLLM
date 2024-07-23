import json
import requests
import time


def stream_str(s):
    for char in s:
        yield char


def stream_resp(resp: requests.Response, typing_delay=0.02, callback=None):
    for chunk_full in resp.iter_content(chunk_size=1024, decode_unicode=True):
        split = chunk_full.split("|", 1)
        if len(split) == 1:
            cate = split[0]
            chunk = ""
        else:
            cate, chunk = split
        print(f"{cate}: {chunk}")

        if cate == "context":
            if callback:
                callback(chunk)
            continue
        elif cate != "answer":
            continue

        # yield chunk
        for char in chunk:
            yield char
            time.sleep(typing_delay)


def send_request(path, port: int = 8000, json={}, method="GET", files=None):
    url = f"http://localhost:{port}/{path}"

    headers = {
        "User-Agent": "apifox/1.0.0 (https://www.apifox.cn)",
        "Content-Type": "application/json",
    }
    return requests.request(
        method, url, headers=headers, files=files, json=json, stream=True
    )


def send_msg(model):
    return send_request("rag/", port=8001, json=model.model_dump(), method="POST")


def session_list():
    return send_request(f"session", port=8001).content.decode("utf-8")


def session_show(session_id):
    return send_request(f"session/{session_id}", port=8001).json()


def clean_session(session_id):
    # body = send_request(f"chat/clear/{user_id}?partition={session_id}").json()
    # return f'{body["code"]}: {body["message"]}'
    return send_request(f"session/{session_id}", port=8001, method="DELETE").json()


def load_dialogs(session_id):
    dialogs = session_show(session_id)
    if dialogs != {}:
        for dialog_json in dialogs[session_id]:
            ind, json_load = dialog_json[0], json.loads(dialog_json[1])
            yield json_load["type"], json_load["data"].get(
                "content", "KeyError: 'content'"
            )
