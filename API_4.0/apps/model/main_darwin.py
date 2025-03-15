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

develop_mode = False
# print(os.getenv("MY_FLAG"))
if os.getenv("MY_FLAG") == "dev":
    develop_mode = True
    print("develop mode....")
else:
    develop_mode = False
    print("not develop mode....")

# sys.exit()
# 开发模式，不创建socket， 不做动作

EMO = {"POSITIVE":0, "NEGATIVE":1, "ANGRY":2, "NULL":3} # NULL(只在没有输入的时候使用 ), 积极，消极，愤怒"

INTERACT = {"POSITIVE":0, "NEGATIVE":1, "ANGRY":2} # 用于对交互结果进行 积极和消极的判定 # 这里还要加一个

model_path = 'save/ysc_model'

buffer_path = 'ysc_buffer.pth'

device = torch.device("cpu")

input_node_num_origin = 16

input_num_mul_index = 16 #  把输入维度放大16倍

assign_label_len = 10 # 情感队列有效长度

input_node_num = input_node_num_origin * input_num_mul_index #  把输入维度放大16倍

output_node_num = 3 # 情感输出种类

label_num = 100 # 解码神经元数目


def _bo_fang(index):
    # 把狗上的Sound 输出改成 Analog Output USB Audio Device, 则可以在不影响dmx 的前提下 实现播放
    try:
        if index == 1:
            file_name = "wang_wang.wav"
        elif index == 2:
            file_name = "woof_sad.wav"
        elif index == 3:
            file_name = "woof_sad_new.wav" # 声音更短
        
        if not develop_mode:
            
            def play_audio_t():
                os.system(f'aplay "{file_name}"') # 
            
            play_thread = threading.Thread(target=play_audio_t, name="play_audio")
            play_thread.start()
            # 使用线程播放，就可以和动作同时进行

        else:
            import winsound
            winsound.PlaySound(file_name, winsound.SND_FILENAME)

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

        # if not develop_mode:

        self.board = self.ysc_darwin_init() # 初始化板子

        self.load_buffer() # 把 buffer 权重 load 进来
        
        if develop_mode:
            print("0 is index :", np.where(self.assign_label.detach().cpu().numpy()==0))
            print("1 is index :", np.where(self.assign_label.detach().cpu().numpy()==1))
            print("2 is index :", np.where(self.assign_label.detach().cpu().numpy()==2))


    def load_buffer(self, buffer_path='real_ysc_buffer_400_mic_cpu.pth'):
        """
            def load_buffer(buffer_path='real_ysc_buffer_400_mic.pth'):
    
                buffer = torch.load(os.path.join(os.path.dirname(__file__), buffer_path))

                for i in range(len(buffer)):
                    for j in range(len(buffer[i])):

                        buffer[i][j] = buffer[i][j].cpu()
                
                torch.save(buffer, f='real_ysc_buffer_400_mic_cpu.pth')
            这里需要手动把 cuda 的tensor 转为 cpu 是由于在训练的时候导致的，如果可以的话,直接cpu训练就没这么麻烦了
        
        """
        try:
            self.buffer = torch.load(os.path.join(os.path.dirname(__file__), buffer_path))
            self.assign_label_update()

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
            print("load buffer no error...", len(self.buffer[0]))    
            print("Net buffer loaded...")
        # print("assign lable is :", self.assign_label)
        # pass

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
        ls = np.zeros((1, label_num))
        
        if False:
            print("input ls is: ", input_ls)

        for _ in range(self.time_step):
            a = self.input_func(input_ls)
            # print("a is: ", a)
            out = self.board.run_darwin3_withoutfile(spike_neurons=[a])
            # print("out is:", out)   
            for i in range(len(out[0])):
                index = out[0][i][1]
                ls[0][index] += 1
                
        if False:
            print("ls nozeros is : " ,ls[0].nonzero())
        # print(ls[ls.nonzero()])
        return ls  # 

    def input_func(self, input_ls, unit_conversion=0.6, dt=0.4):
        # 这两个参数对标spaic的possion_encoder，但由于我的抑制层的权重放大的较小，所以这里可以适当调节 这两个超参数，从而得到合适的脉冲输出
        # 这里改成 numpy 的输入, 输入维度 1 * input_node_num
        # 返回 一个 list 也即发放脉冲的神经元编号
        a = (np.random.rand(*input_ls.shape) < input_ls * unit_conversion * dt)

        return np.nonzero(a[0].astype(int))[0]


    def run(self, data, reward=1): 
        # 对接 darwin 
        # Todo reward 暂未使用
        return self.ysc_darwin_step(data)
    


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
        if develop_mode:
            if False:
                print("shape is: ", output.shape)
                print("output is: ", output)
                print("assign label is :", self.assign_label)
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

class fake_controller:
    def __init__(self):
        
        self.thread_active = False
    
    def zuo_you_huang(self):
        pass

    def stand_up(self):
        pass

    def di_tou_new(self):
        pass

    def low_height_of_dog(self):
        pass

    def fuyang_or_qianhou(self):
        pass

    def pian_hang(self):
        pass

    def niu_yi_niu(self):
        pass

    def di_tou_new(self):
        pass


    
class Gouzi:
    class Sensor():
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
    
    class State(Enum):
        stand_up = 1
        lie_down = 2
        # low_height = 3

        
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

        self.cmd_lock = threading.Lock()

        self.cmd_thread = threading.Thread(target=self.action_from_cmd_thread, name="cmd_thread") # 命令线程

        self.emo_thread = threading.Thread(target=self.emo_from_input_thread, name="emo_thread")

        self.controller = None # 控制器

        self.current_state = self.State.stand_up # 初始状态 设为站立状态
        
        if not develop_mode:
            # 如果是在windows上进行调试，这个运动主机的接口是 需要完全关闭的
            self.action_socket_init() # 初始化 Controller
        else:
            self.controller = fake_controller() #

        self.is_moving = False

        self.emo = (EMO["NULL"], time.time()) # 第二个参数 用于标记当前emo 的时效性

        self.emo_check_window = 10.0 # 检测窗口时间间隔， 也是上一个情感输出 在没有明显情感动作输入后的 持续时间
    """
        实际的传感器排序就如下面所示，输入给模型的编码输入就是 下面的 16 * 16

        | 0     1     2    3      |   4    5   |  6     |  7     8     9   |   10    11   |  12    |  13    14    15  |
        | 蓝    红    黑  表扬语义 |  批评  批评 | 酒精高  | 点赞  手掌  拳头  |  抚摸  抚摸  |   拍打  | 电低  电低  电低  |     
    """
    def test_darwin(self):
        # 这个函数是用来测试达尔文的 输入和 输出的函数
        # 实际不使用
        temp_input = np.zeros(input_node_num_origin) # 初始传感器维度, 传感器初始化

        temp_input[0] = 1  # 使用颜色衣服进行测试

        temp_input = np.array([np.tile(temp_input, input_num_mul_index)])       # 增加了一个维度

        temp_output = self.robot_net.run(data=temp_input)          # 前向传播

        temp_emotion = self.robot_net.predict_with_no_assign_label_update(output=temp_output)

        print("output is : ", temp_output)


        print("emotion output is :", temp_emotion)

    
    def action_socket_init(self):
        # 运动主机初始化、创建运动控制器、建立心跳
        server_address = ("192.168.1.120", 43893)  # 运动主机端口
        self.controller = Controller(server_address) # 创建 控制器

        self.controller.heart_exchange_init() # 初始化心跳
        time.sleep(1)
        self.controller.not_move() # 进入 静止状态
        time.sleep(1)
        self.controller.stand_up() # 站起来
        print('stand_up')
        # pack = struct.pack('<3i', 0x21010202, 0, 0)
        # controller.send(pack) # 
        # time.sleep(2)
        
        print("action socket init...")

    def start(self):
        # 外部调用，启动指令监听线程
        print("start function....")
        self.cmd_thread.start() # 指令线程启动

        print("cmd thread....")
        self.emo_thread.start() # 情感线程启动

        print("emo thread....")
        self.m_wang_wang() # 汪汪一下

        print("Gouzi into socket server...")
        
        # if not develop_mode:
        self.start_server() # 启动监听线程 ， 线程中不断获取传感器数据


    def start_server(self, host='192.168.1.103', port=12345):
        # 启动client 监听线程
        # global develop_mode
        if develop_mode:
            host = "172.31.111.211"
            print("host is ", host)
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
        # obsolote 改成直接调用播放替代
        raise ValueError("OBSOLOTE FUNCTION...")
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
            
            # self.cmd = self.Sensor.Null

    def clear_cmd(self):
        with self.cmd_lock:
            self.cmd = self.Sensor.Null

    def m_lie_down(self):
        
        if self.current_state != self.State.lie_down:
            
            self.controller.thread_active = False

            self.controller.stand_up()

            self.current_state = self.State.lie_down
        
            time.sleep(2)
        else:
            print( "dog is not stand up now!")

    def m_stand_up(self):

        if self.current_state != self.State.stand_up:
            
            self.controller.thread_active = False
            self.controller.stand_up()

            self.current_state = self.State.stand_up
            
            time.sleep(2)
        else:
            print( "dog is not lie down now!")

        


    def m_low_height(self):

        # Todo 把这个改成 可以高度调节的那种
        # assert self.current_state != self.State.low_height, "dog is low height now!"

        if self.current_state == self.State.stand_up:
        
            self.controller.low_height_of_dog()

            # self.current_state = self.State.low_height

            time.sleep(2)
            
            self.controller.thread_active = False

        else:
            print("dog is already stand up now!")

    def restore_height(self):
        # Todo as before
        # assert self.current_state == self.State.low_height, "dog is not low height now!"

        if self.current_state == self.State.low_height:

            self.controller.thread_active = False

            self.current_state = self.State.stand_up

            time.sleep(1)
        else:
            print("dog is not low height now!")

    def m_happy_rotate(self):

        self.controller.zuo_you_huang()

        time.sleep(2)

        self.controller.thread_active = False


    def m_nod_head(self):

        self.controller.fuyang_or_qianhou()

        time.sleep(2)

        self.controller.thread_active = False

    def m_shake_head(self):

        self.controller.pian_hang()

        time.sleep(2)

        self.controller.thread_active = False

    def m_niu_yi_niu(self):
        
        self.controller.niu_yi_niu()
        time.sleep(4)

    def m_di_tou(self):

        self.controller.di_tou_new()

        time.sleep(2)

    def m_wang_wang(self):
        print("wang wang wang......")
        _bo_fang(index=1)

        # time.sleep(0.5)
    
    def m_wu_wu(self):
        
        _bo_fang(index=3)
        
        # time.sleep(1.5)


    def look_left(self):
        # 不要用
        self.controller.look_left()
        print("look left....")
        time.sleep(2) # Todo
    
    def look_right(self):
        # 不要用
        self.controller.look_right()
        print("look right....")
        time.sleep(2)
    
    def m_dian_tou(self):
        self.controller.fuyang_diantou()
        time.sleep(2)
        self.controller.thread_active = False


    """
        实际的传感器排序就如下面所示，输入给模型的编码输入就是 下面的 16 * 16

        | 0     1     2    3      |   4    5   |  6     |  7     8     9   |   10    11   |  12    |  13    14    15  |
        | 蓝    红    黑  表扬语义 |  批评  批评 | 酒精高  | 点赞  手掌  拳头  |  抚摸  抚摸  |   拍打  | 电低  电低  电低  |     
    """
    def cmd_plus_emotion(self, cmd, emo=None):

        # 将 命令 和 情感 融合后输出

        # 其实我感觉真正应该 清楚状态的 是在执行完动作之后, 有道理 说不定 还要再清除一下 emo

        if emo == EMO["NULL"]:
            if self.cmd == self.Sensor.Cmd_GoAhead:                
                    # self.look_left()
                    self.m_niu_yi_niu()
                    print("go ahead null...")
            elif self.cmd == self.Sensor.Cmd_GoBack:
                    self.m_wu_wu() # 播放的时间太长了 # 声音会被自己录到
                    self.m_dian_tou()   
                    print("go back null...") # 
            elif self.cmd == self.Sensor.Cmd_LieDown:

                    print("lie down null...")
                    self.m_wang_wang()
                    # self.m_shake_head()
                    self.m_lie_down()
            elif self.cmd == self.Sensor.Cmd_StandUp:
                    print("stand up null...")
                    # self.m_wu_wu() 
                    self.m_shake_head()
                    self.m_stand_up()
        
        elif emo == EMO["POSITIVE"]:
            
            if cmd == self.Sensor.Cmd_GoAhead:
                print("go ahead happy...")
            elif self.cmd == self.Sensor.Cmd_GoBack:
                
                print("go abck happy...")
            elif self.cmd == self.Sensor.Cmd_LieDown:
                
                print("lie down happy...")
                self.m_niu_yi_niu()
                self.m_lie_down()
                
            elif self.cmd == self.Sensor.Cmd_StandUp:

                print("stand up happy...")
                self.m_stand_up()
                self.m_happy_rotate()

        elif emo == EMO["NEGATIVE"]:

            self.m_shake_head() # 没好像和下一个叠在一起了 # 这里最好能 低头摇

            if cmd == self.Sensor.Cmd_GoAhead:
                print("go ahead sad...")

            elif self.cmd == self.Sensor.Cmd_GoBack:
                
                print("go abck sad...")
                self.m_wu_wu() # 没声音

            elif self.cmd == self.Sensor.Cmd_LieDown:
                self.m_happy_rotate() # 会重复
                print("lie down sad...")
                

                self.m_lie_down()
                
            elif self.cmd == self.Sensor.Cmd_StandUp:
                
                print("stand up sad...")
                self.m_stand_up()
                # self.m_low_height()
                
        
        elif emo == EMO["ANGRY"]:

            self.m_wang_wang()


            if cmd == self.Sensor.Cmd_GoAhead:
                print("go ahead angry...")
                
                # self.m_di_tou()
                # self.m_wang_wang()

            elif self.cmd == self.Sensor.Cmd_GoBack:
                
                print("go abck angry...")
                # self.m_wang_wang()
            elif self.cmd == self.Sensor.Cmd_LieDown:
                
                print("lie down angry...")
                # self.m_wang_wang()
                # self.m_shake_head()

            elif self.cmd == self.Sensor.Cmd_StandUp:
                
                self.m_shake_head()
                print("stand up angry...")

            
        else:

            raise ValueError("we get an error emotion !")

        


    def action_from_cmd_thread(self):
        # 根据情感和动作 融合得到 最终的动作输出   动作线程

        while True:

            if self.cmd != self.Sensor.Null:                                    
                    
                print("current cmd is: ", self.cmd)

                if time.time() - self.emo[1] < self.emo_check_window: 
                    # 如果在emo窗口之内
                    temp_emotion = self.emo[0]
                else:
                    # 在emo 窗口之外说明 长时间没有有效输入
                    temp_emotion = EMO["NULL"]
                
                # print("current emotion output with cmd is :", temp_emotion)
                    
                self.cmd_plus_emotion(cmd=self.cmd, emo=temp_emotion)                   # 这里进行的动作输出
            
                self.clear_cmd() # 清空
                self.clear_sensor_status_with_lock() # 指令执行完 最好也能清空状态
            
            else:

                if time.time() - self.emo[1] < self.emo_check_window: 
                    # 窗口内
                    pass


            time.sleep(0.5) # 慢一点
        
    def emo_from_input_thread(self):
        # 检测外界环境 得到 情感输出    情感线程

        """
            实际的传感器排序就如下面所示，输入给模型的编码输入就是 下面的 16 * 16

            | 0     1     2    3      |   4    5   |  6     |  7     8     9   |   10    11   |  12    |  13    14    15  |
            | 蓝    红    黑  表扬语义 |  批评  批评 | 酒精高  | 点赞  手掌  拳头  |  抚摸  抚摸  |   拍打  | 电低  电低  电低  |     
        """
        def _check_sensor_input(temp_ls):
            # 传感器信号 检测 并转为 编码输入

            # 颜色
            if self.color == self.Sensor.Color_Blue:
                temp_ls[0]= temp_ls[1] = temp_ls[2] = 0 # 每次同属性赋值的时候，清空其他输入
                temp_ls[0] = 1
            elif self.color == self.Sensor.Color_Red:
                temp_ls[0]= temp_ls[1] = temp_ls[2] = 0 
                temp_ls[1] = 1
            elif self.color == self.Sensor.Color_Black:
                temp_ls[0]= temp_ls[1] = temp_ls[2] = 0 
                temp_ls[2] = 1
            
            # 语义
            if self.dmx == self.Sensor.Dmx_Positive:
                temp_ls[3] = temp_ls[4] = temp_ls[5] = 0
                temp_ls[3] = 1
            elif self.dmx == self.Sensor.Dmx_Negative:
                temp_ls[3] = temp_ls[4] = temp_ls[5] = 0
                temp_ls[4] = 1
                temp_ls[5] = 1

            # 酒精
            if self.alcohol == self.Sensor.Other_Alcohol:
                temp_ls[6] = 1                    
            
            # 手势
            if self.gesture == self.Sensor.Gesture_Like:
                temp_ls[7] = temp_ls[8] = temp_ls[9] = 0
                temp_ls[7] = 1
            elif self.gesture == self.Sensor.Gesture_Palm:
                temp_ls[7] = temp_ls[8] = temp_ls[9] = 0
                temp_ls[8] = 1
            elif self.gesture == self.Sensor.Gesture_Dislike:
                temp_ls[7] = temp_ls[8] = temp_ls[9] = 0
                temp_ls[9] = 1


            # IMU
            if self.imu == self.Sensor.IMU_Touching:
                temp_ls[10] = temp_ls[11] = temp_ls[12] = 0
                temp_ls[10] = 1
                temp_ls[11] = 1
            elif self.imu == self.Sensor.IMU_Hit:
                temp_ls[10] = temp_ls[11] = temp_ls[12] = 0
                temp_ls[12] = 1
            
            # 电量
            if self.power == self.Sensor.Other_Power:
                temp_ls[13] = 1
                temp_ls[14] = 1
                temp_ls[15] = 1

            # 不仅能修改 temp_ls 


            if self.dmx == self.Sensor.Dmx_Positive or self.imu == self.Sensor.IMU_Touching:                
                
                temp_emo = EMO["POSITIVE"]
            
            elif self.dmx == self.Sensor.Dmx_Negative:                
                temp_emo = EMO["NEGATIVE"]
            
            elif self.imu == self.Sensor.IMU_Hit:
                temp_emo = EMO["ANGRY"]
            
            else:
                temp_emo = EMO["NULL"]

            if self.color != self.Sensor.Null or self.gesture != self.Sensor.Null:
                
                if_normal_input = True
            else:
                
                if_normal_input = False
            
            # self.clear_sensor_status_with_lock()                                                                  # 这里不去清除状态，把清除操作留给检测到有效输入之后

            return if_normal_input, temp_emo                                                                        # 返回 是否有常规输入、是否有情感因素的外界输入
        
        # temp_time = time.time()


        temp_input = np.zeros(input_node_num_origin) # 初始传感器维度, 传感器初始化， 
        
                                                                                                                    # 这里并不是每个 while 都进行清空，那样的话频率太快
                                                                                                                    # 而且 在 _check_sensor_input 中 同种类的输入是互斥的，所以不用担心 同种类多输入的情况
        
        while True:
            
            if_normal_input, emo_input = _check_sensor_input(temp_ls=temp_input)

            if emo_input != EMO["NULL"] or (if_normal_input and time.time() - self.emo[1] > self.emo_check_window): # 超参数 有待调节
                # 更新 self.emo

                print("input list is: ", temp_input)

                temp_input = np.array([np.tile(temp_input, input_num_mul_index)])                                   # 输入维度扩充，第二种扩充
                
                temp_output = self.robot_net.run(data=temp_input)                                                   # 网络输出
                
                # print("temp output is ", type(temp_output))

                if self.is_moving == False:
                                                                                                                    # 只有不在运动的时候， 才会进行情感计算 # Todo
                    pass

                self.emo = (self.robot_net.predict_with_no_assign_label_update(output=temp_output), time.time())    # 情感输出

                print("emo changed to ", self.emo)

                self.clear_sensor_status_with_lock()                                                                # 清除状态

                temp_input = np.zeros(input_node_num_origin)                                                        # 每当一次有效输入，才会去清空输入

                if emo_input != EMO["NULL"] and if_normal_input:
                                                                                                                    # 这里进行在线调节          
                    self.robot_net.influence_all_buffer(interact=emo_input, temp_output= temp_output)
                    print("There is an influence buffer progress...")
                    # 调用调节函数
            
            elif time.time() - self.emo[1] > self.emo_check_window:
                
                print("long time no meaningful input!!!!")

            time.sleep(0.5)
                

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
                            # print("color received is:", args1)
                        if args2 != 0:
                            self.gesture = args2 # 手势 

                        if args3 != 0:
                            with self.cmd_lock:
                                self.cmd = args3 # 指令

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
                            with self.cmd_lock:
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
            print("a socket is going to close....")
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



