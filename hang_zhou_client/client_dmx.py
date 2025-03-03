
"""
    这边的话，容易报错是其实还是由于之前运行的时候没把进程杀干净， 
    kill掉就不报错了，我猜报错核心dump 就是 两个进程 同时写文件导致的
"""


import pyaudio
import wave
import time
from zhipuai import ZhipuAI
import pyttsx3
import socket
import base64
import urllib
import requests
import json
import os
from filelock import FileLock
import traceback
LOCK_FILE_PATH = "/tmp/ltl.lock" 

audio_lock = FileLock(LOCK_FILE_PATH, timeout=40) # 如果拿不到就一直等

API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"


file_path = "output.wav"

def baidu_wav_to_words():

    url = "https://vop.baidu.com/server_api"

    speech = get_file_content_as_base64(file_path, False)
    payload = json.dumps({
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": "vks6nBUXlchi2SekxmPHOuFoqW0UpeMe",
        "dev_pid": 1537,
        "speech": speech,
        "len": os.path.getsize(file_path),
        "token": get_access_token()
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    return(response.json().get('result')[0])


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))



def demo_api(input_txt):
    conversation_id = None
    output=input_txt
    api_key = "299adac92d9b98c139f22fa1e22a8b2c.t7LzNyfNX49gsShG"
    url = "https://open.bigmodel.cn/api/paas/v4"
    client = ZhipuAI(api_key=api_key, base_url=url)
    prompt = output
    generate = client.assistant.conversation(
        assistant_id="659e54b1b8006379b4b2abd6",
        conversation_id=conversation_id,
        model="glm-4-assistant",
        messages=[
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            }
        ],
        stream=True,
        attachments=None,
        metadata=None
    )
    output = ""
    for resp in generate:
        if resp.choices[0].delta.type == 'content':
            output += resp.choices[0].delta.content
            conversation_id = resp.conversation_id
    return output




host = '192.168.1.103'
socket_port = 12345
golobal_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
golobal_client_socket.connect((host, socket_port))




def lu_yin_and_save():
    print("yes1")
    with audio_lock:
        time.sleep(2)
        print("yes2")
        # 配置录音参数

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        RECORD_SECONDS = 2 # 录音的时间
        WAVE_OUTPUT_FILENAME = "output.wav"

        audio = pyaudio.PyAudio()

        # 打开音频流
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK, input_device_index=28)

        print("录音中...")

        frames = []

        # 录制音频
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            # 如果 锁来了， 就break
            data = stream.read(CHUNK)
            frames.append(data)

        print("录音结束")

        # 停止和关闭音频流

        audio.terminate()
        stream.stop_stream()
        stream.close()
        

        # 将录制的音频保存为wav文件
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("yes3")

def run():
    while True:
        try:


            lu_yin_and_save() # wav

            text = baidu_wav_to_words() # wav 
            print(text) 


            text  = "从一个小狗的角度，判断下面这段话属于，1表扬我，2批评我，3正常谈话， 当中的哪个种类，只需要用编号1或2或3来回答我。" + text
            print(text)
            web_text = demo_api(input_txt=text) # 

            print(web_text)

            if web_text == "1" or  web_text == "2" or web_text == "3":
                print("send")
                data = "dmx " + web_text + " " + "0"
                golobal_client_socket.sendall(data.encode('utf-8'))
                # shuo_zhong_wen(web_text)
                time.sleep(1)
            time.sleep(1) # 大模型不是一直在录音 也会有间断的
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
            # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))
            print("error")
            continue
            




if __name__ == '__main__':
    run()