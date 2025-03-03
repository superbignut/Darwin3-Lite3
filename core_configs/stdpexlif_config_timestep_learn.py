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


def spaic_stdpexlif_ts_learn_config(timestep=25, th_inc=25, th_sub=1, vreset=-100): # 电压每衰减？？？？？？   周期没有用  ？？？？
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


            # 更新权重阶段 这里每一个神经元都执行了256次
            LSSYN(ls=LSSYN.LS.LOAD),                        # NC_CONF_CR_WORKMODE.E 是 0 不使用拓展
            LSLS(ls=LSLS.LS.LOAD, pre=0, nph=0b0110_0101),  # 把迹 load 下来
            UPTSYN(pno=0, nph=UPTSYN.NPH.PRT1A_X_PRT0_Y),   # w' = w + wpar0 * p1a-x * prto-y 输入迹 * 输出脉冲 * 1
            UPTSYN(pno=1, nph=UPTSYN.NPH.PRT1A_Y_PRT0_X),   # w' = w + wpar1 * p1a-y * prt0-x 输出迹 * 输入脉冲 * -1
            LSSYN(ls=LSSYN.LS.STORE),
            NPC(),



            # 推理阶段
            LSIS(ls=LSIS.LS.LOAD, nph=0b0111_0100), # vt vth i wgtsum # 电流的也拿出来了
            ADDI(rs=2, imme=-timestep), # 与立即数相比， 如果是 >= 25 则
            CMP(rs=2, rt=2, func=CMP.Func.GE_0), 
            JC(cmpf=1, cmpr=1, addr=7+len_of_update),
            SUB(rs=0, rt=0, ns=0), # >= 25 执行  vt = 0 清空电压
            SUB(rs=2, rt=2, ns=0), # 清空loop_index
            ADDI(rs=2, imme=-(timestep)), # i = -ts-1
            ADDI(rs=2, imme=(timestep+1)), #   < 25 执行 # 小于的话 + 1
            SUB(rs=6, rt=6, ns=0), # r6 = 0 
            ADDI(rs=6, imme=1), # r6 = 1
            SUB(rs=3, rt=3, ns=0), # r3 = 0     # 这里试一下 1 》 10 位的效果 vt' = 1.001 vt
            ADD(rs=3, rt=0, ns=0), # r3 = vt
            ADD(rs=3, rt=0, ns=0), # r3 = 2vt
            ADD(rs=3, rt=0, ns=0), # r3 = 3vt
            SFTI(rs=3, sf=SFTI.SF.RIGHT, ns=0, imme=8), # r3 >> 8: r3 = 3/256 vt = 0.012vt, 这里有问题，如果是负数的话 会越减越大，应该要加
            ADD(rs=0, rt=3, ns=0), # vt' = vt - 0.01vt = 0.99vt 所以改成 ADD
            LSIS(ls=LSIS.LS.LOAD, nph=0b0000_0010), # 加载res 到寄存器，存储的是我的vth_theta  # 这里第一次 也不能加   vth theta
            ADD(rs=1, rt=5, ns=0), # vth += res # vth  = vth + vth_theta # 这里的vth 被更改了
            UPTVT(fcs=1, fcph=0), # 更新电压 vt += wgtsum
            CMP(rs=0, rt=1, func=CMP.Func.GE),# if vt >= vth : vth_theta + 0.05
            JC(cmpf=1, cmpr=1, addr=22+len_of_update),# CMP = True 不jmp 否则 jmp
            ADDI(rs=5, imme=th_inc), # 每次增加 th_inc
            CMP(rs=5, rt=6, func=CMP.Func.GE),  # 如果vtheta >= 1
            JC(cmpf=1, cmpr=1, addr=25+len_of_update), # CMP = True 不jmp 否则jmp
            ADDI(rs=5, imme=-th_sub), # 减th_sub   13 <-# else vth_theta - 0.01   
            CMP(rs=2, rt=6, func=CMP.Func.EQ),  # 如果rs=2 == 1 # 这里应该是 如果 rs=2 是 1 就不脉冲 gsp参数 是 0
            JC(cmpf=1, cmpr=1, addr=30+len_of_update), # CMP = True 不jmp 否则jmp
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=0, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位:是 发放脉冲：否 与阈值比较：是 重置到：rst
            LSIS(ls=LSIS.LS.STORE, nph=0b0101_0010), # 把vt存回去 vth_theta存回去 vth 被修改了这里不存回去 loop_i也存回去 
            JC(cmpf=0, cmpr=0, addr=32+len_of_update), # 直接跳转 NPC
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=1, vcp=1, rsm=GSPRS.RSM.ZERO), # 正常的脉冲的情况
            LSIS(ls=LSIS.LS.STORE, nph=0b0101_0010), # 把vt存回去 vth_theta存回去 vth 被修改了这里不存回去 loop_i也存回去 
            NPC(),


            LSLS(ls=LSLS.LS.LOAD, pre=0, nph=0b0110_0101), # prt1a-x 输入迹、prt1a-y 输出迹 、prt0-x 输入脉冲、prt0-y输出脉冲 取出 这里可以可以进行衰减操作，反正其余的量我也不用
            UPTLS(unph=0b0101), # prt1a-x 更新输入迹, prt1a-y 更新输出迹,
            LSLS(ls=LSLS.LS.STORE, pre=0, nph=0b0000_0101), # 输入迹保存  由于这个过程在 脉冲发放之后，所以相当于可以拿到这个时刻的脉冲输出， 也可以拿到之前的脉冲输出
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
    core_config.set_register("CR_WPARA", 0x03 | int(hex((-1 & 0xff)<<8), 16)) # wpar0 = 1 wpar1 = -1 # 这里的5 和 -5 应该改成 1 1奥 
    core_config.set_register("CR_STATE", 0x02)  # 学习状态存储器 清零
    core_config.set_register("CR_QA", (0x4 << 8)) # 状态更新阶段精度 16位 随机取整 *15/16 右移 4位  # 这里要 将 0-2 config.dwnc 中该称 0 write 0 2 0x0008 0x00000400 
    # 每次执行完 需要 用clear_neurons_states 将学习状态 清空，否则 会迹的存在 会影响接下来的权重更新
    # core_config.set_register("CR_LPARXY", )
    # core_config.initial_inference_state_memory()



    # 这么写真抽象

    return core_config


"""
            



"""