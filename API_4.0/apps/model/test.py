import os
import sys
import numpy as np
import torch
import time
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