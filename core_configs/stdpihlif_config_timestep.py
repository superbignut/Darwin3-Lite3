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
# from darwin3_deployment.ir.dinst import ADD as _ADD

"""
class LIFModel(NeuronModel):
    
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm) # tag
    O^n[t] = spike_func(V^n[t-1])
    

    def __init__(self, **kwargs):
        super(LIFModel, self).__init__()
        # initial value for state variables
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0


        self._constant_variables['Vth'] = kwargs.get('v_th', 1.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

        self._tau_variables['tauM'] = kwargs.get('tau_m', 8.0)

        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]'))

NeuronModel.register("lif", LIFModel)

"""


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


def spaic_stdpihlif_ts_config(timestep=25, vreset=-100):
    # 输入量化后的阈值， 衰减固定是15/16 在spaic 中需要 修改lif 的 tau_m 为 tau_m=(-0.1 / np.log(3 / 4)) 因为 spaic 自带一层 np.exp(-dt / tau)
    # 用于实现spaic的LIF 神经元
    """
    vt = vt * tauM
    vt = vt + wgsum
    O = func(vt)
    
    vt = vt + wgtsum if FCS == 1
    vt = p1 + vt x p0 + wgtsum x c6 + i x c7 if FCS ==0 其中 c6 = CPAR6 c7 = CPAR7
    """
    """ inference_state_settings={
        "vth": "vth",
    }, """




    core_config = CoreConfig(

        inference_state_settings={
            "vth": "my_vth", # 阈值存储器
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
            SFTI(rs=3, sf=SFTI.SF.RIGHT, ns=0, imme=5), # r3 >> 5: r3 = 3/32 vt = 0.093vt
            ADD(rs=0, rt=3, ns=0), # vt' = vt - 0.01vt = 0.907vt
            UPTVT(fcs=1, fcph=0), # 更新电压 vt += wgtsum
            CMP(rs=2, rt=6, func=CMP.Func.EQ),  # 如果rs=2 == 1 # 这里应该是 如果 rs=2 是 1 就不脉冲 gsp参数 是 0
            JC(cmpf=1, cmpr=1, addr=22), # CMP = True 不jmp 否则jmp
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=0, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位:是 发放脉冲：否 与阈值比较：是 重置到：rst
            LSIS(ls=LSIS.LS.STORE, nph=0b0101_0000), # 把vt存回去 vth_theta存回去 vth 被修改了这里不存回去 loop_i也存回去 
            JC(cmpf=0, cmpr=0, addr=24), # 直接跳转 NPC
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=1, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位:是 发放脉冲：是 与阈值比较：是 重置到：rst
            LSIS(ls=LSIS.LS.STORE, nph=0b0101_0000), # 把vt存回去 vth 被修改了这里不存回去 loop_i也存回去 
            NPC(),
        ]
    )
    
    core_config.set_register("CR_VTDEC", int(hex((vreset & 0xffff)<<16), 16)) 
    return core_config