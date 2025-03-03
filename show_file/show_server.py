"""
    为了25号的展示， 这里写一个小的 server  只接受 imu 和 dmx 两种 客户端的输入

    承接于找到的之前的一个版本的代码 https://github.com/superbignut/emo_alcohol/blob/master/qian_dao_hu_spaic/show_server.py
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
from Controller import Controller
import threading    
import numpy as np

API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"

WAVE_OUTPUT_FILENAME = "output.wav"

record_input_th = 3000

RECORD_STATE = 0 #

RECORD_STATE_DICT = {"Waiting_Input":0, "Input_Now":1, "Waiting_End":2}

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





def _bo_fang(index):
    # with audio_lock: # 在多线程之前上锁
    try:
        if index == 1:
            file_name = "wang_wang.wav"
        elif index == 2:
            file_name = "woof_sad.wav"

        # 打开.wav文件
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        wf = wave.open(file_path, 'rb')

        # 创建PyAudio对象
        p = pyaudio.PyAudio()

        # 打开音频流
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # 播放数据
        data = wf.readframes(1024)

        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # 停止音频流
        stream.stop_stream()
        stream.close()

        # 关闭 PyAudio
        p.terminate()
        wf.close()
    except:
        print("error")
    finally:
        print("over")

class Gouzi:

    def __init__(self) -> None:

        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5

        self.dmx = 0 # 0 1 2 3 # 无，积极，消极                  3

        self.controller = None
        
        self.action_socket_init() # 初始化controller

        self.is_moving = False


    def action_socket_init(self):

        server_address = ("192.168.1.120", 43893)  # 运动主机端口
        self.controller = Controller(server_address) # 创建 控制器
        self.controller.heart_exchange_init() # 初始化心跳
        time.sleep(2)
        self.controller.stand_up()
        print('stand_up')
        # pack = struct.pack('<3i', 0x21010202, 0, 0)
        # controller.send(pack) # 
        self.controller.not_move() # 进入 静止状态
        print("动作socket初始化")
    
    def show_stand(self):
        self.controller.do_move()
        self.say_something(index=1) # 汪一下
        time.sleep(0.5)
        self.controller.stand_up()
        self.controller.not_move()
        # self.say_something(index=1) # 汪一下


    def _start_server_thread(self, host='192.168.1.103', port=12345):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5) # 等待数， 连接数
        print(f"Server listening on {host}:{port}...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            client_handler = threading.Thread(target=self.handle_client_thread, args=(client_socket,))
            client_handler.start()


    def start_server(self, host='192.168.1.103', port=12345):
        temp_thread = threading.Thread(target=self._start_server_thread, name="_start_server_thread")
        temp_thread.start()


    def say_something(self, index):    
        #  temp_p = multiprocessing.Process(target=_bo_fang, args=(index, ))
        temp_t = threading.Thread(target=_bo_fang, name="bo_fang_thread", args=(index, ))
        temp_t.start() # 这里 因为麦克风是 io 且是独占的，所有 多线程可以加速， 并且 需要join
        temp_t.join()
        # _bo_fang(index=index) # 干掉多线程
        print("播放结束")


    def positive_action_with_flag_changed(self): # 做一些积极的动作， 并标志flag
        #########################################
        self.is_moving = True
        #########################################

        self.controller.fuyang_or_qianhou()
        time.sleep(1)
        self.controller.thread_active = False
        
        #########################################
        # 执行完所有动作后
        self.is_moving = False
        #
        ########################################
    def negative_action_with_flag_changed(self): # 做一些消极的动作， 并标志flag
        #########################################
        self.is_moving = True
        #########################################

        self.controller.pian_hang() # 偏航角
        time.sleep(1)
        self.controller.thread_active = False
        time.sleep(1)

        #########################################
        # 执行完所有动作后
        self.is_moving = False
        #
        ########################################


    def do_action_from_input(self):
        # 这里只要状态使用过一次之后 ， 就会被 置零
        while True:

            if self.imu == 1:
                
                # self.controller.light_eyes() # 等待的时候闪闪眼睛 或者
                time.sleep(1)
                #########################################
                self.controller.thread_active = False # 先关掉所有的动作
                self.is_moving = True 
                #########################################

                self.controller.zuo_you_huang()
                # self.say_something(index=4) # 3wang
                
                time.sleep(1)
                self.controller.thread_active = False
                
                #########################################
                # 执行完所有动作后
                self.is_moving = False
                ########################################
                
            elif self.imu == 2:
                # self.controller.light_eyes() # 等待的时候闪闪眼睛 或者
                time.sleep(1)
                #########################################
                self.controller.thread_active = False # 先关掉所有的动作
                self.is_moving = True 
                #########################################

                # self.controller.low_height_of_dog()
                self.say_something(index=1)
                time.sleep(1)
                self.controller.thread_active = False

                #########################################
                # 执行完所有动作后
                self.is_moving = False
                ########################################
            

            self.clear()
            # time.sleep(1) #  统统给我 等待一秒#
            break

            
    def clear(self):
        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
        self.dmx = 0 # 0 1 2 3 # 无，积极，消极                  3



    def handle_client_thread(self, client_socket):
        # 处理线程
        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    continue
                # print(data)
                command, args1, args2 = data.decode('utf-8').split()  # 假设数据格式为 "COMMAND arg1 arg2"
                # print(command, args1, args2)
                args1 = int(args1)
                # args2 = int(args2)
                print(f"Received command: {command}, args: {args1}, {args2}")

                """
                        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
                        self.color = 0 #  0 1 2   无， 蓝色， 红色               3
                        self.alcohol = 0 #  0 1 2   无， 酒精 酒精               3
                        self.dmx = 0 #    1 2 3 #  积极，消极, 正常              3
                        self.gesture = 0 # 0 1 2 3 无  上，下，挥手              4
                        self.power = 0 # 0 1 # 20% 以下， 和以上的电量           2
                """
                if self.is_moving:
                    time.sleep(0.2)
                    continue


                elif command == "imu":
                    if args1 == 1: # 抚摸
                        print("touching_imu")
                        self.imu = 1
                    elif args1 == 2: # 敲打
                        print("was kicked_imu")
                        self.imu = 2
                    else:
                        self.imu = 0


        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
            # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))

        finally:
            client_socket.close()


def flow_record(dog:Gouzi):    
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
                        frames_per_buffer=CHUNK, input_device_index=27)  # windows 改成0 是可以工作的

    frames = []
    cnt = 0 # 用于标记这是第几个交给大模型的音频

    # 录制音频
    print("开始录音...")
    while(True):
        data = stream.read(CHUNK) # 录1个chunk

        data_int = np.abs(np.frombuffer(data, dtype=np.int16))
        # print(len(data), len(data_int)) 由于数据是16位，因此data的len是 8192 的double
        max_data_int = np.max(data_int)
        print("max is: ", np.max(data_int))

        if RECORD_STATE == RECORD_STATE_DICT['Waiting_Input']:
            print("waiting for input...")
            if max_data_int < record_input_th:


                # 这里如果最一开始 什么都没有 就 检测一下 动作输入
                dog.do_action_from_input()
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

                text  = "你是我的一个喜欢吃肉、不喜欢吃菜的宠物小狗，从下面三种选择中做出回答，1是的，2不是，3其他， 只需要用编号1或2或3来回答我:" + text
                print(text)
                web_text = dmx_api(input_txt=text) # 
                print(web_text)
                if web_text == "1":
                    print("yes")
                    dog.positive_action_with_flag_changed()
                elif web_text == "2":
                    print("no")
                    dog.negative_action_with_flag_changed()
                else:
                    print("puzzled")
                cnt += 1
                RECORD_STATE = RECORD_STATE_DICT["Waiting_Input"]


def run():

    ysc_dog=Gouzi()
    ysc_dog.start_server() #  启动任务的监视进程
    
    # check_state_0(dog=ysc_dog)

    flow_record(ysc_dog)
    
    # ysc_dog.do_action_from_input() # 开始循环了

if __name__ == '__main__':
    run()