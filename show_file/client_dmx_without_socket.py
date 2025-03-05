"""
    data = "dmx " + str(args1) + " " + str(args2) + " " + str(args3)

    args1 指令 

    args2 语义

    args3 null
"""

import pyaudio
import wave
import time
from zhipuai import ZhipuAI
# import pyttsx3
import socket
import base64
import urllib
import requests
import json
import os
# from filelock import FileLock
import traceback
import sys

import threading    
import numpy as np

API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"

WAVE_OUTPUT_FILENAME = "output.wav"

record_input_th = 3000

RECORD_STATE = 0 #

RECORD_STATE_DICT = {"Waiting_Input":0, "Input_Now":1, "Waiting_End":2}




Dmx_Positive = 1            # 积极语义
Dmx_Negative = 2            # 消极语义 * 2



Cmd_LieDown = 1             # 趴下指令
Cmd_StandUp = 2             # 站起来指令
Cmd_GoAhead = 3             # 向前走指令
Cmd_GoBack = 4              # 向后走指令
Cmd_Woof = 5                # 往往叫指令

def baidu_wav_to_words(file_name):

    def get_access_token():
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))


    url = "https://vop.baidu.com/server_api"

    
    speech = get_file_content_as_base64(file_name, False)
    sp_len = os.path.getsize(file_name)
        
    payload = json.dumps({
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": "vks6nBUXlchi2SekxmPHOuFoqW0UpeMe",
        "dev_pid": 1537,
        "speech": speech,
        "len": sp_len,# os.path.getsize(file_path),
        "token": get_access_token()
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.json())
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

def dmx_api(input_txt):
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


def flow_record():    
    global WAVE_OUTPUT_FILENAME, record_input_th, RECORD_STATE
    time.sleep(2)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024 * 8 # 大改一个chunk 是 0.5s
    
    audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=28)  # windows 改成0 是可以工作的

    frames = []
    cnt = 0 # 用于标记这是第几个交给大模型的音频
    args1 = 0   # socket 第一个参数
    args2 = 0   # socket 第二个参数
    args3 = 0   # socket 第三个参数

    # 录制音频
    print("开始录音...")
    while(True):



        data = stream.read(CHUNK) # 录1个chunk

        data_int = np.abs(np.frombuffer(data, dtype=np.int16))
        # print(len(data), len(data_int)) 由于数据是16位，因此data的len是 8192 的double
        max_data_int = np.max(data_int)
        
        # print("max is: ", max_data_int)

        if RECORD_STATE == RECORD_STATE_DICT['Waiting_Input']:
            print("waiting for input...")
            if max_data_int < record_input_th:
                continue 
            else:
                frames.append(data)
                RECORD_STATE = RECORD_STATE_DICT["Input_Now"]
        
        if RECORD_STATE == RECORD_STATE_DICT["Input_Now"]:
            frames.append(data)
            
            if max_data_int < record_input_th:
                RECORD_STATE = RECORD_STATE_DICT["Waiting_End"]
            else:
                continue
        
        if RECORD_STATE == RECORD_STATE_DICT["Waiting_End"]:
            frames.append(data)

            if max_data_int > record_input_th:
                RECORD_STATE = RECORD_STATE_DICT["Input_Now"]
            else:
                # 第二次小于则 则把数据提交给 dmx
                # 先转为 wav 文件，提交给百度
                wf_name = str(cnt) + WAVE_OUTPUT_FILENAME
                wf = wave.open(wf_name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                frames.clear() # 数据列表清空

                text = baidu_wav_to_words(file_name=wf_name) # 百度转文字
                
                if "起来" in text:
                    print("######### stand up!!!")
                    args1 = Cmd_StandUp
                elif "趴下" in text:
                    print("######### lie down!!!")
                    args1 = Cmd_LieDown
                elif "过来" in text:
                    print("######### lie down!!!")
                    args1 = Cmd_GoAhead
                else:
                    print(text)
                    text  = "你是我的宠物小狗,判断下面这句话是属于哪一类? 1表扬我、2批评我、3与我无关。用编号回答: " + text
                    
                    web_text = dmx_api(input_txt=text) # 
                    print(web_text)
                    if web_text == "1":
                        print("yes")
                        # data = "dmx " + "1 0"  
                        # client_socket.sendall(data.encode('utf-8'))
                        args2 = Dmx_Positive

                    elif web_text == "2":
                        print("no")
                        # data = "dmx " + "2 0"  
                        # client_socket.sendall(data.encode('utf-8'))
                        args2 = Dmx_Negative
                    else:
                        print("puzzled")
                    cnt += 1
                if args1 != 0 or args2 != 0:
                    data = "dmx " + str(args1) + " " + str(args2) +  " 0" # 第三个参数总是0
                    # client_socket.sendall(data.encode('utf-8'))
                    print(data)
                    
                    # 清空
                    args1 = 0   # socket 第一个参数
                    args2 = 0   # socket 第二个参数
                    args3 = 0   # socket 第三个参数

                RECORD_STATE = RECORD_STATE_DICT["Waiting_Input"]



    
if __name__ == '__main__':
    flow_record()