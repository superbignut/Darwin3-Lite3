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

register = {
    "vt":0,
    "vth":1,
    "i":2,
    'rpd':3,
    'wgtsum':4,
    'res':5,
    'rand':6,
    'wgt':7,
    'pra01':0xD,
    'pra23':0xE,
    'pra45':0xF
}
""" for i in range(len(register)):
    register.i =  """
def spaic_stdpexlif_config(vth=0, tauM=3/4):

    core_config = CoreConfig(
        inference_state_settings={
            "res" : "vth_theta", # 用作变化的阈值
            "vth": "vth", # 阈值存储器
            # 打算把i 用做大周期的循环，比如每25个输入进行一次 电压的重置操作

        },

        assembly_program=[
            LSIS(ls=LSIS.LS.LOAD, nph=0b0110_0100), # vt vth weightsum
            SUB(rs=6, rt=6, ns=0), # r6 = 0
            ADDI(rs=6, imme=1), # r6 = 1
            SUB(rs=3, rt=3, ns=0), # r3 = 0
            ADD(rs=3, rt=0, ns=0), # r3 = vt
            ADD(rs=3, rt=0, ns=0), # r3 = 2vt
            ADD(rs=3, rt=0, ns=0), # r3 = 3vt
            SFTI(rs=3, sf=SFTI.SF.RIGHT, ns=0, imme=8), # r3 >> 8: r3 = 3/256 vt = 0.012vt
            SUB(rs=0, rt=3, ns=0), # vt' = vt - 0.01vt = 0.99vt
            LSIS(ls=LSIS.LS.LOAD, nph=0b0000_0010), # 加载res 到寄存器，存储的是我的vth_theta
            ADD(rs=1, rt=5, ns=0), # vth += res # vth  = vth + vth_theta # 这里的vth 被更改了
            UPTVT(fcs=1, fcph=0), # 更新电压 vt += wgtsum
            CMP(rs=0, rt=1, func=CMP.Func.GE),# if vt >= vth : vth_theta + 0.05
            JC(cmpf=1, cmpr=1, addr=15),# CMP = True 不jmp 否则 jmp
            ADDI(rs=5, imme=2),
            CMP(rs=5, rt=6, func=CMP.Func.GE),  # 如果vtheta >= 1
            JC(cmpf=1, cmpr=1, addr=18), # CMP = True 不jmp 否则jmp
            ADDI(rs=5, imme=-1), # 减1   13 <-# else vth_theta - 0.01
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=1, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位:是 发放脉冲：是 与阈值比较：是 重置到：rst
            LSIS(ls=LSIS.LS.STORE, nph=0b0100_0010), # 把vt存回去 vth_theta存回去 vth 被修改了这里不存回去
            NPC(),
        ]
    )
    """     if vth >= 0:
        core_config.set_register("R1", int(vth)) # 这么写应该是 只配置了寄存器，但是没配 存储器
    else:
        core_config.set_register("R1", int(hex(-52 & 0xffff), 16)) """
    
    # vth 的话 用LSIS 直接加载 就不用手动去设置寄存器了

    # 每次脉冲后衰减到 -100
    core_config.set_register("CR_VTDEC", int(hex((-100 & 0xffff)<<16), 16)) 

    # core_config.initial_inference_state_memory()



    # 这么写真抽象

    return core_config