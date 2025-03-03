"""
    相较于stdplif_ts 增加权重的修改， 体现在汇编和 “配置寄存器” 的配置上

    

"""

from hang_zhou_spaic.SPAIC import spaic
# from hang_zhou_spaic.SPAIC.spaic
import torch
import math
from hang_zhou_spaic.SPAIC.spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum
from darwin3_deployment.core_config.core_config import CoreConfig
from darwin3_deployment.ir.dinst import * 


def test_lif(timestep=25, th_inc=25, th_sub=1, vreset=-100): # 电压每衰减？？？？？？   周期没有用  ？？？？
    """
        这里的timestep 实际运行多少次，就给多少就ok, 目前已经可以做到 在timestep结束后进行 vt 清零

        每timestep 次后 电压清空
    
    """
    len_of_update = 6

    core_config = CoreConfig(
        inference_state_settings={
             "res" : "vth_theta", # 用作变化的阈值
             "vth": "my_vth", # 阈值存储器
            # 打算把i 用做大周期的循环，比如每25个输入进行一次 电压的重置操作
             "i":"my_loop_index",

        },
        learning_parameter_settings={
            # "PRT1A-X":"1ax", # 这里不清零 会出问题
        },


        assembly_program=[

            ADDI(rs=2, imme=1),
            NPC(),

            ADDI(rs=3, imme=1),
            NPC(),  

            # 更新 迹 阶段

        ]
    )


    core_config.set_register("CR_VTDEC", int(hex((vreset & 0xffff)<<16), 16)) 
    core_config.set_register("CR_LI", 0x00) # 每个时刻都进行 更新权重 和 更新学习参数
    core_config.set_learning_mode(True) # 
    # 迹的量程是7位，所以最大127
    core_config.set_register("CR_LPARXY", 0x0E | 0x0E<<16) # LPAR0 = 15 # 不衰减 LPAR2 = 15
    core_config.set_register("CR_LPARR", 0x0E << 8 | 0x0E << 16) # LPAR5 = 15 防止 右移9取整约没了； LPAR6=15 # 脉冲系数
    core_config.set_register("CR_WPARA", 0x02 | int(hex((-1 & 0xff)<<8), 16)) # wpar0 = 1 wpar1 = -1 # 这里的5 和 -5 应该改成 1 1奥 
    core_config.set_register("CR_STATE", 0x02)  # 学习状态存储器 清零
    core_config.set_register("CR_QA", (0x4 << 8)) # 状态更新阶段精度 16位 随机取整 *15/16 右移 4位 
    # 每次执行完 需要 用clear_neurons_states 将学习状态 清空，否则 会迹的存在 会影响接下来的权重更新
    # core_config.set_register("CR_LPARXY", )
    # core_config.initial_inference_state_memory()



    # 这么写真抽象

    return core_config
