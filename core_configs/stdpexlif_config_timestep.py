"""
    相较于stdplif 增加了 一个时间周期， 更符合spaic 的 timestep 和 dt 的关系

    这里打算用推理状态存储器的 电流 i 来做一个定时

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
def spaic_stdpexlif_ts_config(timestep=25, th_inc=25, th_sub=1, vreset=-100): # 电压每衰减？？？？？？   周期没有用  ？？？？
    """
        这里的timestep 实际运行多少次，就给多少就ok, 目前已经可以做到 在timestep结束后进行 vt 清零

        每timestep 次后 电压清空
    
    """
    core_config = CoreConfig(
        inference_state_settings={
            "res" : "vth_theta", # 用作变化的阈值
            "vth": "my_vth", # 阈值存储器
            # 打算把i 用做大周期的循环，比如每25个输入进行一次 电压的重置操作
            "i":"my_loop_index",

        },






        assembly_program=[
            LSIS(ls=LSIS.LS.LOAD, nph=0b0111_0100), # vt vth i wgtsum # 电流的也拿出来了
            ADDI(rs=2, imme=-timestep), # 与立即数相比， 如果是 >= 25 则
            CMP(rs=2, rt=2, func=CMP.Func.GE_0), 
            JC(cmpf=1, cmpr=1, addr=7),
            SUB(rs=0, rt=0, ns=0), # >= 25 执行  vt = 0 清空电压
            SUB(rs=2, rt=2, ns=0), # 清空loop_index
            ADDI(rs=2, imme=-(timestep)), # i = -ts-1
            ADDI(rs=2, imme=(timestep+1)), #   < 25 执行 # 小于的话 + 1
            SUB(rs=6, rt=6, ns=0), # r6 = 0 
            ADDI(rs=6, imme=1), # r6 = 1
            SUB(rs=3, rt=3, ns=0), # r3 = 0
            ADD(rs=3, rt=0, ns=0), # r3 = vt
            ADD(rs=3, rt=0, ns=0), # r3 = 2vt
            ADD(rs=3, rt=0, ns=0), # r3 = 3vt
            SFTI(rs=3, sf=SFTI.SF.RIGHT, ns=0, imme=8), # r3 >> 8: r3 = 3/256 vt = 0.012vt, 这里有问题，如果是负数的话 会越减越大，应该要加
            ADD(rs=0, rt=3, ns=0), # vt' = vt - 0.01vt = 0.99vt 所以改成 ADD
            LSIS(ls=LSIS.LS.LOAD, nph=0b0000_0010), # 加载res 到寄存器，存储的是我的vth_theta  # 这里第一次 也不能加   vth theta
            ADD(rs=1, rt=5, ns=0), # vth += res # vth  = vth + vth_theta # 这里的vth 被更改了
            UPTVT(fcs=1, fcph=0), # 更新电压 vt += wgtsum
            CMP(rs=0, rt=1, func=CMP.Func.GE),# if vt >= vth : vth_theta + 0.05
            JC(cmpf=1, cmpr=1, addr=22),# CMP = True 不jmp 否则 jmp
            ADDI(rs=5, imme=th_inc), # 每次增加 th_inc
            CMP(rs=5, rt=6, func=CMP.Func.GE),  # 如果vtheta >= 1
            JC(cmpf=1, cmpr=1, addr=25), # CMP = True 不jmp 否则jmp
            ADDI(rs=5, imme=-th_sub), # 减th_sub   13 <-# else vth_theta - 0.01   
            CMP(rs=2, rt=6, func=CMP.Func.EQ),  # 如果rs=2 == 1 # 这里应该是 如果 rs=2 是 1 就不脉冲 gsp参数 是 0
            JC(cmpf=1, cmpr=1, addr=30), # CMP = True 不jmp 否则jmp
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=0, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位:是 发放脉冲：否 与阈值比较：是 重置到：rst
            LSIS(ls=LSIS.LS.STORE, nph=0b0101_0010), # 把vt存回去 vth_theta存回去 vth 被修改了这里不存回去 loop_i也存回去 
            JC(cmpf=0, cmpr=0, addr=32), # 直接跳转 NPC
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=1, vcp=1, rsm=GSPRS.RSM.ZERO), # 正常的脉冲的情况
            LSIS(ls=LSIS.LS.STORE, nph=0b0101_0010), # 把vt存回去 vth_theta存回去 vth 被修改了这里不存回去 loop_i也存回去 
            NPC(),

        ]
    )
    """     if vth >= 0:
        core_config.set_register("R1", int(vth)) # 这么写应该是 只配置了寄存器，但是没配 存储器
    else:
        core_config.set_register("R1", int(hex(-52 & 0xffff), 16)) """
    
    # vth 的话 用LSIS 直接加载 就不用手动去设置寄存器了

    # 每次脉冲后衰减到 -100
    core_config.set_register("CR_VTDEC", int(hex((vreset & 0xffff)<<16), 16)) 

    # core_config.initial_inference_state_memory()



    # 这么写真抽象

    return core_config