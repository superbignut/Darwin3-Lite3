"""
    这个文件相较于 main.py 目的在于把原来用  spaic 的网络传播 改成用 Darwin3 的完全替代 

    并增添了指令部分 修改了整个情感对四足机器人影响的方式

    能成功启动一次挺不容易的

    首先要把 darwin 的 ssh 跑通 这里会遇到的问题是 NX的网络总跳, 但是接上显示器等到 右上wifi不再闪烁 即为ok

    其次是 麦克风的 ID 要先检测 再跑，再其次是 代理要配好

    摄像头 和 imu 问题最小， cv 包装上了，大概率没问题了

    电量的需要重新编译

    酒精的也没啥问题
"""
import os
import sys
import numpy as np
import torch
import time
# sys.path.append("../scripts")
from darwin3_runtime_api import darwin3_device
import numpy as np
from tqdm import tqdm
import os
import sys
import random
import torch
from enum import Enum
import torch.nn.functional as F
import multiprocessing
import csv
import pandas as pd
from collections import deque
import traceback
import threading
import socket
from Controller import Controller
import time
import subprocess
import wave
import pyaudio

EMO = {"POSITIVE":0, "NEGATIVE":1, "ANGRY":2, "NULL_P":3, "NULL_N":4} # NULL(只在没有输入的时候使用 ), 积极，消极，愤怒

INTERACT = {"POSITIVE":0, "NEGATIVE":1, "ANGRY":2} # 用于对交互结果进行 积极和消极的判定 # 这里还要加一个

model_path = 'save/ysc_model'

buffer_path = 'ysc_buffer.pth'

device = torch.device("cuda:0")

input_node_num_origin = 16

input_num_mul_index = 16 #  把输入维度放大16倍

assign_label_len = 10 # 情感队列有效长度

input_node_num = input_node_num_origin * input_num_mul_index #  把输入维度放大16倍

output_node_num = 3 # 情感输出种类

label_num = 100 # 解码神经元数目


def _bo_fang(index):
    try:
        if index == 1:
            file_name = "wang_wang.wav"
        elif index == 2:
            file_name = "woof_sad.wav"
        
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        wf = wave.open(file_path, 'rb')        

        p = pyaudio.PyAudio()           # 创建PyAudio对象
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=wf.getnchannels(),rate=wf.getframerate(),output=True)        
        data = wf.readframes(1024)      # 播放数据
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
    except:
        print("audio played error!")
    finally:
        print("audio played over!")

class Darwin_Net():
    def __init__(self):
        super().__init__()

        self.buffer = [[] for _ in range(output_node_num)] # 这里不能写成 [[]] * 4 的形式， 否则会右拷贝的问题

        self.assign_label = None # 统计结束的解码层神经元的的情感分组

        self.time_step = 25

        self.board = self.ysc_darwin_init() # 初始化板子

    def ysc_darwin_init(self):
        # darwin 板子初始化 ，参考运行时手册
        board = darwin3_device.darwin3_device(app_path='API_4.0/apps/', step_size=1000_000, ip=['172.31.111.35'], spk_print=True) # 172.31.111.35

        time.sleep(1)
        board.reset() # 重置
        time.sleep(1)
        board.darwin3_init(333) # 时钟频率初始化
        time.sleep(1)
        board.deploy_config() # 板子权重、连接初始化
        time.sleep(1)

        return board

    def ysc_darwin_step(self, input_ls):
        # 一个时间步周期的运行， 输入数据维度 1 * 256，也即需要 [[1,2,3,4,...]] 这种
        # 返回脉冲输出，维度与输入一致 1 * 256
        # 最外层似乎的一层没什么用，只是和 pytorch 的历史一致
        ls = np.zeros(100,)
        print("input ls is: ", input_ls)
        for _ in range(self.time_step):
            a = self.input_func(input_ls)
            # print("a is: ", a)
            out = self.board.run_darwin3_withoutfile(spike_neurons=[a])
            # print("out is:", out)   
            for i in range(len(out[0])):
                index = out[0][i][1]
                ls[index] += 1
        # print(ls.nonzero())
        # print(ls[ls.nonzero()])
        return np.array(ls.nonzero())  # 

    def input_func(self, input_ls, unit_conversion=0.6, dt=0.1):# 这连个参数对标spaic的possion_encoder
        # 这里改成 numpy 的输入, 输入维度 1 * input_node_num
        # 返回 一个 list 也即发放脉冲的神经元编号
        a = (np.random.rand(*input_ls.shape) < input_ls * unit_conversion * dt)

        return np.nonzero(a[0].astype(int))[0]


    def run(self, data, reward=1): 
        # 对接 darwin 
        # Todo reward 暂未使用
        return self.ysc_darwin_step(data)
    

    
    def new_check_label_from_data(self, data):
        # 这里暗含了优先级的概念在里面, 但要是能真正影响 情绪输出的还得是 权重
        
        if data[0][2] == 1 or data[0][5] == 1 or data[0][10] == 1 or data[0][12] == 1:
            return EMO["ANGRY"] # 
        
        elif data[0][15] == 1 or  data[0][14] == 1 or data[0][7] == 1 or data[0][6] == 1 or data[0][4] == 1 or data[0][1] == 1:
            return EMO['NEGATIVE']
        
        elif data[0][0] == 1 or data[0][3] == 1 or data[0][8] == 1 or data[0][9] == 1 or data[0][11] == 1 or data[0][13] == 1:
            return EMO['POSITIVE']
        
        else:
            raise NotImplementedError


    def assign_label_update(self, newoutput=None, newlabel=None, weight=0):
        # 如果没有新的数据输入，则就是对 assign_label 进行一次计算，否则 会根据权重插入新数据，进而计算

        if newoutput != None:
            self.buffer[newlabel].append(newoutput)
        try:
            avg_buffer = [sum(self.buffer[i][-assign_label_len:]) / len(self.buffer[i][-assign_label_len:]) for i in range(len(self.buffer))] # sum_buffer 是一个求和之后 取平均的tensor  n * 1 * 100
            
            assign_label = torch.argmax(torch.cat(avg_buffer, 0), 0) # n 个1*100 的list在第0个维度合并 -> n*100的tensor, 进而在第0个维度比哪个更大, 返回一个1维的tensor， 内容是index，[0,n)， 目前是012
            
            self.assign_label = assign_label # 初始化结束s
        except ZeroDivisionError:
            # 如果分母是零 说明是刚开始数据还不够的时候，就暂时不管
            return     
          
    def predict_with_no_assign_label_update(self, output):
        # 根据输出 返回模型的预测
        if self.assign_label == None:
            raise ValueError("predict_with_no_assign_label_update error!")
        
        temp_cnt = [0 for _ in range(len(self.buffer))]
        temp_num = [0 for _ in range(len(self.buffer))]

        for i in range(len(self.assign_label)):
            # print(i)
            temp_cnt[self.assign_label[i]] += output[0, i]  # 第一个维度是batch, 
            temp_num[self.assign_label[i]] += 1
    
        predict_label = np.argmax(np.array(temp_cnt) / np.array(temp_num)) # 有待验证
        
        return predict_label
    
    def influence_all_buffer(self, interact, temp_output):
        # interact ： 0 积极交互 1 消极交互 2 愤怒交互，
        # 
        if interact == EMO['POSITIVE']:

            self.buffer[EMO['POSITIVE']].append(temp_output)
            
            self.buffer[EMO['NEGATIVE']][-1] += -1 * temp_output 

            self.buffer[EMO['ANGRY']][-1] += -1 * temp_output 
            
        elif interact == EMO["NEGATIVE"]:

            self.buffer[EMO['NEGATIVE']].append(temp_output)

            self.buffer[EMO['POSITIVE']][-1] += -1 * temp_output

            self.buffer[EMO['ANGRY']][-1] += -1 * temp_output 
        else:
            self.buffer[EMO['ANGRY']].append(temp_output)

            self.buffer[EMO['POSITIVE']][-1] += -1 * temp_output

            self.buffer[EMO['NEGATIVE']][-1] += -1 * temp_output 
            
        self.assign_label_update() # 施加了积极和消极得影响后 重新 assign label


    def single_test(self):
        
        print(self.assign_label)
        t = 1
        while t < 20:
            t+=1
            result_list = [0.0] * input_node_num_origin
            for i in range(input_node_num_origin):
                if i == 4: # 1 红， 9 10 抚摸
                    result_list[i] = 1.0
                else:
                    result_list[i] = random.uniform(0, 0.2)
            result_list = result_list * input_num_mul_index
            # print(result_list)
            
            # Todo 有待补充



class Gouzi:
    class Sensor(Enum):
        # 各种检测到的传感器 编码输入数据 和 指令数据的状态

        IMU_NUM = 2                 # IMU 传感器个数
        COLOR_NUM = 3               # COLOR 传感器个数
        DMX_NUM = 2                 # DMX 传感器个数
        GESTURE_NUM = 3             # GESTURE 传感器个数
        CMD_NUM = 5                 # CMD 指令个数

        Null = 0                    # 这个就是 各个信号0的状态

        IMU_Touching = 1            # 抚摸 * 2
        IMU_Hit = 2                 # 拍打

        Color_Red = 1               # 红颜色
        Color_Blue = 2              # 蓝颜色
        Color_Black = 3             # 黑颜色

        Dmx_Positive = 1            # 积极语义
        Dmx_Negative = 2            # 消极语义 * 2

        Other_Power = 1             # 电量低 * 3
        Other_Alcohol = 1           # 酒精浓度高

        Gesture_Like = 4            # 点赞手势
        Gesture_Dislike = 5         # 拳头手势
        Gesture_Palm = 6            # 手掌手势

        # 总体输入编码的维度是 16, 除此之外，下面的信号被用来作为四足机器人的指令变量

        Cmd_LieDown = 7             # 趴下指令
        Cmd_StandUp = 8             # 站起来指令
        Cmd_GoAhead = 9             # 向前走指令
        Cmd_GoBack = 10             # 向后走指令
        Cmd_Woof = 11               # 往往叫指令
        
        # Todo 当然还有更多样的指令，扭一扭、跟随之类的，先不弄

    def __init__(self) -> None:

        self.imu = self.Sensor.Null
        self.color = self.Sensor.Null
        self.alcohol = self.Sensor.Null
        self.dmx = self.Sensor.Null
        self.gesture = self.Sensor.Null
        self.power = self.Sensor.Null
        self.cmd = self.Sensor.Null

        self.robot_net = Darwin_Net() # 情感模型网络
        
        self.state_update_lock = threading.Lock() # 这个lock 使用来检测 狗的状态的更新的， 在检测线程 和 clear 线程中使用

        self.cmd_thread = threading.Thread(target=self.cmd_waiting_thread, name="action_thread") # 命令线程

        self.controller = None # 控制器
        
        self.action_socket_init() # 初始化 Controller

        self.is_moving = False

    def test_darwin(self):
        # 这个函数是用来测试达尔文的 输入和 输出的函数
        # 实际不使用
        temp_input = np.zeros(input_node_num_origin) # 初始传感器维度, 传感器初始化

        temp_input[1] = 1 
        temp_input[3] = 1
        temp_input[2] = 1
        temp_input[4] = 1

        temp_input = np.array([np.tile(temp_input, input_num_mul_index)])       # 增加了一个维度

        temp_output = self.robot_net.run(data=temp_input)          # 前向传播

        print("output is : ", temp_output)

    
    def action_socket_init(self):
        # 运动主机初始化、创建运动控制器、建立心跳
        server_address = ("192.168.1.120", 43893)  # 运动主机端口
        self.controller = Controller(server_address) # 创建 控制器

        self.controller.heart_exchange_init() # 初始化心跳
        time.sleep(2)
        self.controller.stand_up() # 站起来
        print('stand_up')
        # pack = struct.pack('<3i', 0x21010202, 0, 0)
        # controller.send(pack) # 
        time.sleep(2)
        self.controller.not_move() # 进入 静止状态
        print("action socket init...")

    def start(self):
        # 外部调用，启动指令监听线程
        # Gouzi 启动完毕

        self.say_something(index=1) # 汪汪一下

        self.cmd_thread.start() # 启动

        print("Gouzi into socket server...")
        
        self.start_server() # 启动监听线程 ， 线程中不断获取传感器数据


    def start_server(self, host='192.168.1.103', port=12345):
        # 启动client 监听线程
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5) # 等待数， 连接数
        print(f"Server listening on {host}:{port}...")

        while True:
            client_socket, addr = server_socket.accept() # 这里会阻塞
            print(f"Connection from {addr}")
            client_handler = threading.Thread(target=self.client_handle_thread, args=(client_socket,))
            client_handler.start()

    def say_something(self, index):    
        # 调用外部函数，播放音频
        temp_t = threading.Thread(target=_bo_fang, name="bo_fang_thread", args=(index, ))
        temp_t.start() # 这里 因为麦克风是 io 且是独占的，所有 多线程可以加速， 并且 需要join
        temp_t.join()
        print("播放结束")

    
    def clear_sensor_status_with_lock(self):
        # 清除传感器状态变量，清除前加锁
        
        with self.state_update_lock:
            
            self.imu = self.Sensor.Null
            self.color = self.Sensor.Null
            self.alcohol = self.Sensor.Null
            self.dmx = self.Sensor.Null
            self.gesture = self.Sensor.Null
            self.power = self.Sensor.Null
            
            self.cmd = self.Sensor.Null

    def cmd_plus_emotion(self, cmd, emo):

        # 将 命令 和 情感 融合后输出
        pass

    def _check_emotion_input(self):
        
        # 这里对有明显情感导向的动作 进行检测， 如果有的话，对应的式后续的情感调节

        pass

    def cmd_waiting_thread(self):
        # Gouzi 交互逻辑主线程

        # 0 每次检测各个传感器信号， 判断是不是指令
        # 1 如果是指令，则找到情感输出
        # 2 融合指令和情感后 动作输出
        # 3 动作输出后 进入窗口期 等待倒计时结束
            # 4 期间如果有评价信号，则调节情感模型buffer
            # 5 期间如果有指令信号，则进入下一次指令周期
            # 6 倒计时结束
            
        """
            实际的传感器排序就如下面所示，输入给模型的编码输入就是 下面的 16 * 16

            | 0     1     2    3      |   4    5   |  6     |  7     8     9   |   10    11   |  12    |  13    14    15  |
            | 蓝    红    黑  表扬语义 |  批评  批评 | 酒精高  | 点赞  点踩   手掌 |  抚摸  抚摸  |   拍打  | 电低  电低  电低  |     
        """
        def _check_sensor_input(temp_ls):
            # 传感器信号 检测 并转为 编码输入

            # 颜色
            if self.color == self.Sensor.Color_Blue:
                temp_ls[0] = 1
            elif self.color == self.Sensor.Color_Red:
                temp_ls[1] = 1
            elif self.color == self.Sensor.Color_Black:
                temp_ls[2] = 1
            
            # 语义
            if self.dmx == self.Sensor.Dmx_Positive:
                temp_ls[3] = 1
            elif self.dmx == self.Sensor.Dmx_Negative:
                temp_ls[4] = 1
                temp_ls[5] = 1

            # 酒精
            if self.alcohol == self.Sensor.Other_Alcohol:
                temp_ls[6] = 1                    
            
            # 手势
            if self.gesture == self.Sensor.Gesture_Like:
                temp_ls[7] = 1
            elif self.gesture == self.Sensor.Gesture_Dislike:
                temp_ls[8] = 1
            elif self.gesture == self.Sensor.Gesture_Palm:
                temp_ls[9] = 1

            # IMU
            if self.imu == self.Sensor.IMU_Touching:
                temp_ls[10] = 1
                temp_ls[11] = 1
            elif self.imu == self.Sensor.IMU_Hit:
                temp_ls[12] = 1
            
            # 电量
            if self.power == self.Sensor.Other_Power:
                temp_ls[13] = 1
                temp_ls[14] = 1
                temp_ls[15] = 1
        
        while True:
            
            temp_input = np.zeros(input_node_num_origin) # 初始传感器维度, 传感器初始化

            if self.cmd != self.Sensor.Null:                                    
                # 如果有指令输入，从darwin得到情感输出， 进而机器狗动作输出

                _check_sensor_input(temp_ls=temp_input)                             # 传感器输入检测

                print("current cmd is: ", self.cmd)
                print("current temp_input is:", temp_input)                         # 查看修改后的输入数据
                
                temp_input = np.array([np.tile(temp_input, input_num_mul_index)])   # 输入维度扩充，第二种扩充
                
                temp_emotion = self.robot_net.run(data=temp_input)                  # 情感输出

                self.cmd_plus_emotion(cmd=self.cmd, emo=temp_emotion)               # 这里进行的动作输出
                
            
            else:
                
                # 这里其实没有必要 做一个 时间窗口， 直接对有明确感情的动作也做一个 if 进入
                
                emo_input = self._check_emotion_input() # 检测到的具有明显情感倾向的输入

                if emo_input != None:
                    # 如果有明确的情感输入, 没有指令就只迭代情感变化

                    pass
                # 调节 buffer
            
            self.clear_sensor_status_with_lock()                                # 清除状态
            
            time.sleep(0.5)                                                     # 这里确实是要 等一下，相当于 每次留给 client_handle_thread 的传感器观察时间

            

    def client_handle_thread(self, client_socket):
        # 处理不同客户端上报的传感器数据，这里需要一个通信格式的定义
        # 这里会不断更新各个 传感器状态 和 接收到的指令的状态
        """
            "COMMAND args1 args2"

                COMMAND 是代表不同命令的字符串

                args1 代表 Sensor 的各个传感器数值

                args2 代表 收到的指令，语音指令优先
        
        """
        try:
            while True:
                data = client_socket.recv(1024)
                if not data: 
                    continue 
                if self.is_moving:                                                              # 正在运动则跳过
                    continue
                
                command, args1, args2, args3 = data.decode('utf-8').split()                     # 数据格式为 "COMMAND arg1 arg2 args3"
                
                args1, args2, args3 = int(args1), int(args2), int(args3)                        # 转换为整数
                
                # print(f"Received command: {command}, args: {args1}, {args2}, {args3}")

                with self.state_update_lock:  # 修改状态 上锁
                    if command == "video":

                        if args1 != 0:
                            self.color = args1

                        if args2 != 0:
                            self.gesture = args2

                        if args3 != 0:
                            self.cmd = args3

                    elif command == "imu":

                        if args1 != 0:
                            self.imu = args1

                    elif command == "alcohol":
                        if args1 == self.Sensor.Other_Alcohol:         
                            self.alcohol = args1
                        else:
                            self.alcohol = self.Sensor.Null

                    elif command == "power":
                        if args1 == self.Sensor.Other_Power:         
                            self.power = args1
                        else:
                            self.power = self.Sensor.Null

                    elif command == "dmx":
                        if args1 != 0:         
                            self.cmd = args1

                        if args2 != 0:
                            self.dmx = args2
                        
                    
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


if __name__ == "__main__":
    
    # 如果需要重新构造数据集的话，需要重新打开这个函数， 把其余部分注释掉
    # 这个文件需要在 API_4.0 外面执行
    # ysc_darwin_init()
    xiaobai = Gouzi()
    # xiaobai.test_darwin()
    # xiaobai.test_darwin()
    # xiaobai.test_darwin()
    xiaobai.start()



