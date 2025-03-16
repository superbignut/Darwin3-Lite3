import os
import sys
import numpy as np
import torch
import time
from Controller import Controller
# sys.path.append("../scripts")
""" from darwin3_runtime_api import darwin3_device

# test = darwin3_device.darwin3_device(app_path='../', step_size=10000, ip=['192.168.1.90']) # 172.31.111.35
board = darwin3_device.darwin3_device(app_path='API_4.0/apps/', step_size=1000_000, ip=['172.31.111.35'], spk_print=True) # 172.31.111.35

time.sleep(1)
board.reset()


time.sleep(1)
board.darwin3_init(333)

time.sleep(2)
board.deploy_config()

time.sleep(1)


print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]]))
print(board.run_darwin3_withoutfile(spike_neurons=[[0,1,2,3,4]])) """


import torch, os
# 这个函数用来做把 cuda 的 buffer 转为 cpu 的 buffer
def load_buffer(buffer_path='real_ysc_buffer_400_mic.pth'):
    
    buffer = torch.load(os.path.join(os.path.dirname(__file__), buffer_path))

    for i in range(len(buffer)):
        for j in range(len(buffer[i])):

            buffer[i][j] = buffer[i][j].cpu()
    
    torch.save(buffer, f='real_ysc_buffer_400_mic_cpu.pth')

    # load_buffer()

    torch.load(os.path.join(os.path.dirname(__file__), 'real_ysc_buffer_400_mic_cpu.pth'))


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
    
class Gouzi:

        

    def __init__(self) -> None:

        self.controller = None # 控制器

        self.action_socket_init() # 初始化 Controller

    def test_action(self):
        # 测试狗的动作
        
        pass

    
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

        # if self.current_state == self.State.stand_up:
        
        self.controller.low_height_of_dog()

            # self.current_state = self.State.low_height

        # time.sleep(2)
            
        # self.controller.thread_active = False

        

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

    def m_tai_tou_wang(self):

        self.controller.tai_tou()

        _bo_fang(index=1)

        time.sleep(1)
        
        self.controller.thread_active = False

    def m_yao_tou_wu(self):

        self.controller.pian_hang()

        _bo_fang(index=3)

        time.sleep(2)
        
        self.controller.thread_active = False

    def m_wang_wang(self):
        print("wang wang wang......")
        _bo_fang(index=1)

        # time.sleep(0.5)
    
    def m_wu_wu(self):
        
        _bo_fang(index=3)
        
        # time.sleep(1.5)


    def look_left(self):
        # 不要用
        pass
    
    def look_right(self):
        # 不要用
        pass
    
    def m_dian_tou(self):
        self.controller.fuyang_diantou()
        time.sleep(2)
        self.controller.thread_active = False

    def m_shake_hands(self):
        self.controller.da_zhao_hu()
        time.sleep(6) # 大概 6 s 完成动作

    def m_follow(self):
        # 不要用
        # print("begin follow....")
        self.controller.follow()

    def m_close_follow(self):
        # 不要用
        self.controller.close_ai()

    def m_move_f(self):
        self.controller.move_forward()
        time.sleep(2)
        self.controller.thread_active = False

    def m_move_low_f(self):
        self.controller.move_low_forward()
        time.sleep(2)
        self.controller.thread_active = False

    def m_move_b(self):
        self.controller.move_backward()
        time.sleep(2)
        self.controller.thread_active = False
    
    def m_move_f_low_and_keep_low(self):

        self.controller.move_low_forward()
        time.sleep(3)
        self.controller.thread_active = False

        self.m_low_height()


    def m_move_b_low_and_keep_low(self):

        self.controller.move_low_backward()
        time.sleep(3)
        self.controller.thread_active = False
        
        self.m_low_height()
        # time.sleep(1)

    def low_height_all_the_time(self):
        # 这里可以标记一个变量 ，来表示 全局的 身体高度
        # 发现 如果 先切到 动作模式  在 切成 匍匐步态  再切回静止 模式 
        # 就会 让身体 全局降低一个高度，而且不影响 静止状态下 其他动作的执行
        # 比 m_low_height 函数要好用很多
        self.controller.do_move_low()
        time.sleep(0.2)


if __name__ == '__main__':

    xiaobai = Gouzi()

    # xiaobai.controller.do_move_low()
    
    
    # time.sleep(1)

    xiaobai.m_low_height()

    while True:
        
        
        # xiaobai.m_low_height()

        # xiaobai.low_height_all_the_time()

        # xiaobai.m_yao_tou_wu()
        # xiaobai.m_low_height()

        # xiaobai.m_shake_hands()

        time.sleep(3)
